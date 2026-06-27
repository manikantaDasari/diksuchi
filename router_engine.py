"""
router_engine.py — Routing decision logic for the Local-First AI API Router.

Evaluates a chain of rules (defined in config.yaml) against each incoming
request and returns a RoutingDecision (LOCAL or CLOUD) plus the matched rule,
human-readable reason, and optional provider override.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import zoneinfo

log = logging.getLogger("diksuchi")

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
    bucket: str = ""             # centroid classifier result (simple/coding/reasoning/creative/critical)
    bucket_confidence: float = 0.0  # cosine similarity score (0-1)


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
            route_to  = rule["route_to"]
            rule_name = rule["name"]

            # ── Post-rule allowed-tiers policy check ─────────────────────────
            # Header overrides (force_local_header / force_cloud_header) are
            # explicit user intent — never subject to the routing policy.
            # All other rules (keyword, regex, token-count) ARE policy-checked:
            # if the rule wants to send to local (tier 0) but the user has
            # blocked tier 0 for this bucket via allowed_tiers, we run the
            # centroid classifier to identify the bucket and reroute.
            _HARD_OVERRIDES = {"force_local_header", "force_cloud_header"}
            allowed_tiers_cfg: dict = routing_cfg.get("bucket_allowed_tiers", {})

            if (route_to == "local"
                    and rule_name not in _HARD_OVERRIDES
                    and allowed_tiers_cfg):
                try:
                    from centroid_classifier import classify_prompt, get_classifier_tier
                    _bucket, _conf = classify_prompt(prompt_text)
                    _bucket_allowed = allowed_tiers_cfg.get(_bucket)

                    if _bucket_allowed is not None and 0 not in _bucket_allowed:
                        # Tier 0 is blocked for this bucket — find lowest allowed tier
                        _rec_tier   = get_classifier_tier(_bucket)
                        _lower      = [t for t in _bucket_allowed if t <= _rec_tier]
                        _adjusted   = max(_lower) if _lower else min(_bucket_allowed)
                        _chain: list[str] = routing_cfg.get("fallback_chain", ["local"])
                        _provider   = _chain[min(_adjusted, len(_chain) - 1)]
                        _backend    = Backend.LOCAL if _provider == "local" else Backend.CLOUD
                        _cloud_prov = "" if _backend == Backend.LOCAL else _provider

                        log.info(
                            "🎯  [policy-override] rule=%s blocked by allowed_tiers for "
                            "bucket=%s (allowed=%s) → tier %d (%s)",
                            rule_name, _bucket, _bucket_allowed, _adjusted, _provider,
                        )
                        return RoutingDecision(
                            backend=_backend,
                            rule_name=f"policy:{_bucket}",
                            reason=(
                                f"Rule '{rule_name}' → local, but allowed-tiers policy "
                                f"blocks tier 0 for '{_bucket}' bucket "
                                f"(allowed={_bucket_allowed}) → {_provider}"
                            ),
                            token_count=tokens,
                            prompt_preview=preview,
                            model_requested=model,
                            provider=_cloud_prov,
                            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                            bucket=_bucket,
                            bucket_confidence=_conf,
                        )
                except Exception:
                    pass   # classifier unavailable — respect original rule

            return RoutingDecision(
                backend=Backend(route_to),
                rule_name=rule_name,
                reason=rule.get("reason", "Rule matched"),
                token_count=tokens,
                prompt_preview=preview,
                model_requested=model,
                provider=rule.get("provider", ""),   # optional provider pin
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    # ── Centroid classifier fallback ─────────────────────────────────────────
    # No explicit rule matched. Use the embedding centroid classifier to
    # determine capability bucket, then map bucket → tier → provider.
    #
    # Confidence threshold: if the best cosine similarity score is below the
    # threshold, the classifier isn't sure — stay local (cheapest, safest).
    # Default threshold 0.35 chosen to catch genuinely ambiguous prompts.
    try:
        from centroid_classifier import classify_prompt, get_classifier_tier
        bucket, confidence = classify_prompt(prompt_text)

        confidence_threshold: float = routing_cfg.get(
            "classifier_confidence_threshold", 0.35
        )

        if confidence < confidence_threshold:
            # Low-confidence classification — default to local to avoid
            # unnecessarily spending cloud quota on a guess.
            return RoutingDecision(
                backend=Backend.LOCAL,
                rule_name="centroid:low_confidence",
                reason=(
                    f"Centroid classifier → '{bucket}' bucket "
                    f"(confidence {confidence:.0%} < threshold {confidence_threshold:.0%}) "
                    f"— defaulting to local"
                ),
                token_count=tokens,
                prompt_preview=preview,
                model_requested=model,
                provider="",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                bucket=bucket,
                bucket_confidence=confidence,
            )

        tier = get_classifier_tier(bucket)
        original_tier = tier
        policy_note: str = ""

        # ── Per-bucket allowed-tiers policy ──────────────────────────────
        # Users can restrict which tiers are usable per bucket via the
        # dashboard routing-policy grid (multi-select checkboxes).
        #
        # Stored as:  bucket_allowed_tiers: {reasoning: [1, 2], simple: [0, 1]}
        # Meaning:    only those tier indices are permitted for that bucket.
        #
        # Resolution when classifier's tier is NOT in the allowed set:
        #   1. Try the highest allowed tier ≤ recommended (stay cheap / prefer lower)
        #   2. If none, fall to the lowest allowed tier > recommended (floor bump)
        #   3. If allowed list is empty → no constraint (shouldn't happen; UI guards)
        #
        # Legacy `bucket_min_tiers` (old min_tier floor) is converted on-the-fly
        # to an equivalent allowed_tiers list so old config keeps working.
        allowed_tiers_cfg: dict = routing_cfg.get("bucket_allowed_tiers", {})
        bucket_allowed: list[int] | None = allowed_tiers_cfg.get(bucket)

        if bucket_allowed is None:
            # Legacy fallback: convert min_tier → [min_t, min_t+1, ..., 2]
            min_tiers: dict = routing_cfg.get("bucket_min_tiers", {})
            min_t = min_tiers.get(bucket, 0)
            if min_t > 0:
                bucket_allowed = [t for t in (0, 1, 2) if t >= min_t]
            # else: no constraint, all tiers permitted

        if bucket_allowed is not None and bucket_allowed and tier not in bucket_allowed:
            lower = [t for t in bucket_allowed if t <= tier]
            adjusted = max(lower) if lower else min(bucket_allowed)
            policy_note = (
                f"[allowed-tiers policy: tier {tier} blocked for '{bucket}' "
                f"(allowed={bucket_allowed}) → adjusted to {adjusted}]"
            )
            log.info(
                "🎯  [allowed-tiers] bucket=%s classifier_tier=%d → %d (allowed=%s)",
                bucket, tier, adjusted, bucket_allowed,
            )
            tier = adjusted

        # ── Confidence-tiered routing cap ─────────────────────────────────
        # Three confidence bands:
        #   LOW    < low_threshold  (default 0.35):  already caught above → local
        #   MEDIUM low..med_thresh  (default 0.60):  cap at tier 1 (free cloud max)
        #                                            — uncertain prompt, avoid paid cost
        #   HIGH   >= med_threshold (default 0.60):  full tier allowed (including paid)
        #
        # The cap is a soft CEILING applied after the allowed-tiers policy.
        # It will not cap below the lowest allowed tier for this bucket.
        # e.g. if user sets allowed=[2] for "critical", confidence is medium →
        # cap would want tier 1, but tier 1 is not allowed → cap is bypassed.
        confidence_medium_threshold: float = routing_cfg.get(
            "classifier_confidence_medium_threshold", 0.60
        )
        confidence_cap_applied = False
        if confidence < confidence_medium_threshold and tier > 1:
            cap_target = 1
            # Don't cap below the lowest allowed tier
            if bucket_allowed is not None and bucket_allowed:
                cap_target = max(1, min(bucket_allowed))
            if tier > cap_target:
                log.info(
                    "🛡️  [confidence-cap] bucket=%s tier=%d → capped at %d "
                    "(confidence %.0%% < medium threshold %.0%%)",
                    bucket, tier, cap_target, confidence * 100, confidence_medium_threshold * 100,
                )
                tier = cap_target
                confidence_cap_applied = True

        # Map tier to provider using fallback_chain from config
        fallback_chain: list[str] = routing_cfg.get("fallback_chain", ["local"])
        tier_idx = min(tier, len(fallback_chain) - 1)
        resolved_provider = fallback_chain[tier_idx]

        backend = Backend.LOCAL if resolved_provider == "local" else Backend.CLOUD
        cloud_provider = "" if backend == Backend.LOCAL else resolved_provider

        reason_parts = [
            f"Centroid classifier → '{bucket}' bucket",
            f"(confidence {confidence:.0%}, tier {tier} → {resolved_provider})",
        ]
        if policy_note:
            reason_parts.append(policy_note)
        if confidence_cap_applied:
            reason_parts.append(
                f"[confidence cap: {confidence:.0%} < {confidence_medium_threshold:.0%} → tier {tier} max]"
            )

        return RoutingDecision(
            backend=backend,
            rule_name=f"centroid:{bucket}",
            reason=" ".join(reason_parts),
            token_count=tokens,
            prompt_preview=preview,
            model_requested=model,
            provider=cloud_provider,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            bucket=bucket,
            bucket_confidence=confidence,
        )
    except Exception:
        pass   # classifier unavailable — fall through to hard default

    # No rule matched, no classifier — use default
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
