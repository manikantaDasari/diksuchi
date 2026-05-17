"""
router_engine.py — Routing decision logic for the Local-First AI API Router.

Evaluates a chain of rules (defined in config.yaml) against each incoming
request and returns a RoutingDecision (LOCAL or CLOUD) plus the matched rule,
human-readable reason, and optional provider override.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import zoneinfo

# tiktoken is optional — falls back to whitespace-split estimation if absent
# Also catches OSError/ProxyError at encoder load time (e.g. offline environments)
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        # ~0.75 tokens per word is a reasonable approximation
        return max(1, int(len(text.split()) / 0.75))


# ─── Enums & Data-classes ─────────────────────────────────────────────────────

class Backend(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass
class RoutingDecision:
    backend: Backend
    rule_name: str
    reason: str
    token_count: int
    prompt_preview: str          # first 120 chars of the assembled prompt
    model_requested: str
    provider: str = ""           # provider override from rule (empty = use active)
    latency_ms: float = 0.0      # set by the proxy after the decision is made


# ─── Helper: assemble prompt text from OpenAI messages array ─────────────────

def messages_to_text(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):                   # vision / multi-modal
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ─── Rule evaluators ─────────────────────────────────────────────────────────

def _eval_rule(rule: dict, *, prompt_text: str, token_count: int,
               model_name: str, headers: dict[str, str]) -> bool:
    """Return True if the rule condition matches."""
    condition = rule.get("condition", "")

    if condition == "header_equals":
        # Normalise header name to lowercase so config can use any casing
        h = headers.get(rule["header"].lower(), "")
        return h.strip().lower() == rule["value"].strip().lower()

    if condition == "model_name_contains":
        low = model_name.lower()
        return any(v.lower() in low for v in rule.get("values", []))

    if condition == "prompt_contains_any":
        low = prompt_text.lower()
        return any(kw.lower() in low for kw in rule.get("keywords", []))

    if condition == "prompt_matches_regex":
        pattern = rule.get("pattern", "")
        flags = re.IGNORECASE | re.MULTILINE
        return bool(re.search(pattern, prompt_text, flags))

    if condition == "token_count_gte":
        return token_count >= rule.get("threshold", 0)

    if condition == "token_count_lte":
        return token_count <= rule.get("threshold", 9_999_999)

    if condition == "token_count_between":
        lo, hi = rule.get("min", 0), rule.get("max", 9_999_999)
        return lo <= token_count <= hi

    if condition == "time_of_day":
        # Rule fields:
        #   timezone: "Asia/Kolkata"          (IANA tz name, default UTC)
        #   start:    "09:00"                  (HH:MM, 24-hr, inclusive)
        #   end:      "18:00"                  (HH:MM, 24-hr, exclusive)
        #   days:     [0,1,2,3,4]              (0=Mon … 6=Sun, default all)
        tz_name = rule.get("timezone", "UTC")
        try:
            tz = zoneinfo.ZoneInfo(tz_name)
        except Exception:
            tz = timezone.utc
        now = datetime.now(tz)
        start_str = rule.get("start", "00:00")
        end_str   = rule.get("end",   "23:59")
        sh, sm = map(int, start_str.split(":"))
        eh, em = map(int, end_str.split(":"))
        start_mins = sh * 60 + sm
        end_mins   = eh * 60 + em
        now_mins   = now.hour * 60 + now.minute
        allowed_days: list[int] = rule.get("days", list(range(7)))
        in_window = start_mins <= now_mins < end_mins
        on_day    = now.weekday() in allowed_days
        return in_window and on_day

    return False


# ─── Main routing function ────────────────────────────────────────────────────

def decide(
    messages: list[dict],
    model: str,
    headers: dict[str, str],
    routing_cfg: dict,
) -> RoutingDecision:
    """
    Evaluate rules top-to-bottom, return first match.
    Falls back to routing_cfg['default_backend'] if nothing matches.
    Rules may optionally carry a `provider` key to pin a specific provider.
    """
    t0 = time.perf_counter()

    prompt_text = messages_to_text(messages)
    tokens = count_tokens(prompt_text)
    preview = prompt_text[:120].replace("\n", " ")

    rules: list[dict] = routing_cfg.get("rules", [])
    default_backend = Backend(routing_cfg.get("default_backend", "local"))

    for rule in rules:
        if not rule.get("name"):
            continue
        try:
            matched = _eval_rule(
                rule,
                prompt_text=prompt_text,
                token_count=tokens,
                model_name=model,
                headers=headers,
            )
        except Exception:
            matched = False

        if matched:
            return RoutingDecision(
                backend=Backend(rule["route_to"]),
                rule_name=rule["name"],
                reason=rule.get("reason", "Rule matched"),
                token_count=tokens,
                prompt_preview=preview,
                model_requested=model,
                provider=rule.get("provider", ""),   # optional provider pin
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    # No rule matched — use default
    return RoutingDecision(
        backend=default_backend,
        rule_name="default",
        reason=f"No rule matched — using default backend ({default_backend.value})",
        token_count=tokens,
        prompt_preview=preview,
        model_requested=model,
        provider="",
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
