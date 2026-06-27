"""
main.py — Local-First AI API Router  (multi-provider edition)

Drop-in OpenAI-compatible proxy that routes requests to any local LLM
(Ollama, LM Studio, MLX, llama.cpp) or cloud API (OpenAI, Anthropic, Groq,
Gemini, Together AI, Mistral) based on configurable rules.

Supported provider types:
  ollama             — /api/chat  (Ollama native)
  openai             — /chat/completions  (OpenAI)
  openai_compatible  — /chat/completions  (Groq, Gemini, LM Studio, MLX, etc.)
  anthropic          — /v1/messages  (auto-converted from/to OpenAI format)

Usage:
    pip install -r requirements.txt
    cp .env.example .env
    python main.py

Switch active provider at runtime:
    curl -X POST http://localhost:8080/v1/providers/cloud/activate/groq
    curl -X POST http://localhost:8080/v1/providers/local/activate/lmstudio
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from router_engine import Backend, RoutingDecision, decide
import centroid_classifier as _cc
from rating_fetcher import load_ratings, refresh_ratings, best_model_for_bucket
import model_discovery as _md

# ── Persistent model preferences ──────────────────────────────────────────────
_PREFS_FILE = os.path.join(os.path.dirname(__file__), "model_preferences.json")

def _load_prefs() -> dict:
    """Load runtime preferences (override config.yaml defaults)."""
    try:
        if os.path.exists(_PREFS_FILE):
            with open(_PREFS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_prefs(prefs: dict) -> None:
    with open(_PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)

def _effective_prefs() -> dict:
    """Merge config.yaml bucket_preferences (defaults) with runtime overrides."""
    defaults = ROUTING_CFG.get("bucket_preferences", {})
    overrides = _load_prefs()
    # runtime overrides win; skip empty string values
    merged = {k: v for k, v in defaults.items() if v}
    merged.update({k: v for k, v in overrides.items() if v})
    return merged

# ─── Bootstrap ────────────────────────────────────────────────────────────────

load_dotenv()
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CFG = load_config()
SERVER_CFG   = CFG.get("server", {})
PROVIDERS_CFG = CFG.get("providers", {})
ROUTING_CFG  = CFG.get("routing", {})
OBS_CFG      = CFG.get("observability", {})

logging.basicConfig(
    level=getattr(logging, SERVER_CFG.get("log_level", "info").upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diksuchi")


# ─── In-memory request log ────────────────────────────────────────────────────

MAX_LOG = OBS_CFG.get("request_log_size", 200)
request_log: deque[dict] = deque(maxlen=MAX_LOG)
stats: dict[str, Any] = {
    "total": 0,
    "local": 0,
    "cloud": 0,
    "errors": 0,
    "fallbacks": 0,
    "rules_hit": {},
    "started_at": datetime.utcnow().isoformat() + "Z",
}


# ─── Model pricing & context-window tables ────────────────────────────────────
# (input_per_1M_tokens, output_per_1M_tokens) in USD.
# Matched by substring against the model id — first hit wins, so put more
# specific strings (e.g. "gpt-4o-mini") before broader ones ("gpt-4o").
_MODEL_PRICING: list[tuple[str, float, float]] = [
    # ── OpenAI ────────────────────────────────────────────────────────────────
    ("gpt-5.3-codex",        30.00, 120.00),
    ("gpt-5.2-pro",          30.00, 120.00),
    ("gpt-5.2",              15.00,  60.00),
    ("gpt-5.1",              10.00,  40.00),
    ("gpt-5-nano",            0.40,   1.60),
    ("gpt-5-mini",            1.10,   4.40),
    ("gpt-5-pro",            25.00, 100.00),
    ("gpt-5",                15.00,  60.00),
    ("o3-pro",              200.00, 800.00),
    ("o3",                   10.00,  40.00),
    ("gpt-4o-mini",           0.15,   0.60),
    ("gpt-4o",                2.50,  10.00),
    ("gpt-4.1-mini",          0.40,   1.60),
    ("gpt-4.1",               2.00,   8.00),
    # ── Anthropic ─────────────────────────────────────────────────────────────
    ("claude-opus-4",        15.00,  75.00),
    ("claude-sonnet-4",       3.00,  15.00),
    ("claude-haiku-4",        0.80,   4.00),
    ("claude-3-5-sonnet",     3.00,  15.00),
    ("claude-3-5-haiku",      0.80,   4.00),
    # ── OpenRouter paid (via openrouter) ─────────────────────────────────────
    ("deepseek/deepseek-r1",  0.55,   2.19),
    ("deepseek/deepseek-v3",  0.28,   1.10),
    ("anthropic/claude-opus", 15.00,  75.00),
    ("anthropic/claude-sonnet", 3.00, 15.00),
    ("anthropic/claude-haiku", 0.80,  4.00),
    ("openai/gpt-5",         15.00,  60.00),
    ("openai/gpt-4",          2.50,  10.00),
    ("google/gemini-2.5-pro", 1.25,   5.00),
    ("google/gemini-2.5-flash", 0.10, 0.40),
    ("x-ai/grok-3-mini",      0.30,   0.50),
    ("x-ai/grok-3",           3.00,   9.00),
    ("meta-llama/llama-4-maverick", 0.17, 0.60),
    ("mistralai/mistral-large", 2.00,  6.00),
    ("mistralai/codestral",   0.30,   0.90),
    # ── OpenRouter FREE (:free suffix) → always $0 ───────────────────────────
    (":free",                 0.00,   0.00),
    # ── Groq ──────────────────────────────────────────────────────────────────
    ("llama-3.3-70b",         0.59,   0.79),
    ("llama-3.1-70b",         0.59,   0.79),
    ("llama-3.1-8b",          0.05,   0.08),
    ("mixtral-8x7b",          0.24,   0.24),
    # ── Local models → always free ────────────────────────────────────────────
    ("gemma",                 0.00,   0.00),
    ("llama",                 0.00,   0.00),
    ("qwen",                  0.00,   0.00),
    ("mistral",               0.00,   0.00),
    ("phi",                   0.00,   0.00),
    ("deepseek",              0.00,   0.00),
    ("codellama",             0.00,   0.00),
    ("tinyllama",             0.00,   0.00),
]

# context window in tokens, keyed by substring of model id
_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    ("gpt-4.1",           1_047_576),
    ("gemini-2.5",        1_048_576),
    ("gemini-1.5-pro",    2_000_000),
    ("llama-4",             524_288),
    ("claude",              200_000),
    ("gpt-5",               200_000),
    ("gpt-4o",              128_000),
    ("deepseek-r1",         128_000),
    ("llama-3.3",           128_000),
    ("llama-3.1",           128_000),
    ("qwen2.5",             128_000),
    ("gemma3",              128_000),
    ("phi4",                128_000),
    ("mistral",              32_768),
    ("llama-3.2",            32_768),
    ("gemma3:1b",            32_768),
]


def _pricing(model: str) -> tuple[float, float]:
    """Return (input_per_1M, output_per_1M) USD for a model id."""
    m = (model or "").lower()
    for key, inp, out in _MODEL_PRICING:
        if key in m:
            return inp, out
    return 0.0, 0.0


def _context_window(model: str) -> int:
    m = (model or "").lower()
    for key, size in _CONTEXT_WINDOWS:
        if key in m:
            return size
    return 128_000   # safe default


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp_rate, out_rate = _pricing(model)
    return round(
        (prompt_tokens * inp_rate + completion_tokens * out_rate) / 1_000_000, 6
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ProviderManager — builds and holds clients for every configured provider
# ─────────────────────────────────────────────────────────────────────────────

class ProviderManager:
    def __init__(self, providers_cfg: dict):
        self._cfg = providers_cfg
        self._local_map: dict[str, dict] = {}
        self._cloud_map: dict[str, dict] = {}
        self._clients: dict[str, httpx.AsyncClient] = {}

        for p in providers_cfg.get("local", {}).get("list", []):
            self._local_map[p["name"]] = p
        for p in providers_cfg.get("cloud", {}).get("list", []):
            self._cloud_map[p["name"]] = p

        # Runtime-mutable active providers (can be switched via API)
        self._active_local: str = providers_cfg.get("local", {}).get("active", "ollama")
        self._active_cloud: str = providers_cfg.get("cloud", {}).get("active", "openai")

        # Live model discovery state
        self._discovery_at: str | None = None       # ISO timestamp of last run
        self._discovery_counts: dict[str, int] = {} # provider → new models found

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def active_local(self) -> str:
        return self._active_local

    @active_local.setter
    def active_local(self, name: str) -> None:
        if name not in self._local_map:
            raise ValueError(f"Unknown local provider: {name!r}")
        self._active_local = name
        log.info("🔄  Active local provider → %s", name)

    @property
    def active_cloud(self) -> str:
        return self._active_cloud

    @active_cloud.setter
    def active_cloud(self, name: str) -> None:
        if name not in self._cloud_map:
            raise ValueError(f"Unknown cloud provider: {name!r}")
        self._active_cloud = name
        log.info("🔄  Active cloud provider → %s", name)

    def local_providers(self) -> list[dict]:
        return list(self._local_map.values())

    def cloud_providers(self) -> list[dict]:
        return list(self._cloud_map.values())

    def get_local(self, name: str) -> dict:
        return self._local_map.get(name, {})

    def get_cloud(self, name: str) -> dict:
        return self._cloud_map.get(name, {})

    def get_models(self, side: str, provider_name: str) -> list[str]:
        """Return the current model list for a given provider."""
        p = (self._local_map if side == "local" else self._cloud_map).get(provider_name, {})
        return [m for m in p.get("models", []) if m]

    async def health_check_local(self, provider_name: str) -> bool:
        """
        Quick liveness check before committing to a streaming response.
        Tries the provider's /api/tags (Ollama) or /v1/models endpoint.
        Returns True if the provider responds within 3 s, False otherwise.
        This prevents sending response headers before knowing the local
        provider is actually up — otherwise a mid-stream failure can't be
        recovered with a clean fallback.
        """
        p = self.get_local(provider_name)
        if not p:
            return False
        client = self._clients.get(f"local:{provider_name}")
        if not client:
            return False
        ptype = p.get("type", "openai_compatible")
        try:
            if ptype == "ollama":
                resp = await client.get("/api/tags", timeout=3.0)
            else:
                resp = await client.get("/v1/models", timeout=3.0)
            return resp.status_code < 500
        except Exception:
            return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        # Build local clients
        for name, p in self._local_map.items():
            # Docker Compose can override Ollama URL via env var
            base = p.get("base_url", "http://localhost:11434")
            if name == "ollama":
                base = os.getenv("AI_ROUTER_OLLAMA_URL", base)
            self._clients[f"local:{name}"] = httpx.AsyncClient(
                base_url=base,
                timeout=p.get("timeout_seconds", 120),
            )
        # Build cloud clients
        for name, p in self._cloud_map.items():
            headers = {"Content-Type": "application/json"}
            api_key = os.getenv(p.get("api_key_env", ""), "")
            ptype = p.get("type", "openai")
            if ptype == "anthropic":
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
            elif ptype == "openrouter":
                # OpenRouter requires these headers to identify the app
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                headers["HTTP-Referer"] = "https://github.com/manikantaDasari/diksuchi"
                headers["X-Title"] = "Diksuchi"
            elif api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            self._clients[f"cloud:{name}"] = httpx.AsyncClient(
                base_url=p.get("base_url", ""),
                headers=headers,
                timeout=p.get("timeout_seconds", 60),
            )
        for side, providers in (("local", self._local_map), ("cloud", self._cloud_map)):
            for name, p in providers.items():
                log.info(
                    "🔧  Provider config — %s/%s type=%s base_url=%s enabled=%s",
                    side,
                    name,
                    p.get("type", ""),
                    p.get("base_url", ""),
                    p.get("enabled", True),
                )
        log.info("✅  ProviderManager ready — %d local, %d cloud providers",
                 len(self._local_map), len(self._cloud_map))

        # ── Live model discovery ───────────────────────────────────────────────
        # Non-blocking: discovery errors never crash startup.
        await self.run_discovery()

    async def run_discovery(self) -> dict:
        """
        Discover live model lists from each supported provider and merge them
        into the in-memory provider config.  Updates self._discovery_at.
        Returns a summary dict with per-provider counts.

        Safe to call at any time (startup, background loop, or manual refresh).
        """
        summary: dict[str, int] = {}

        # ── Ollama ────────────────────────────────────────────────────────────
        # Local providers: discovered list is authoritative (only installed models).
        # If discovery succeeds with ≥1 model, REPLACE the seed list entirely.
        # Seed models are just a startup fallback for when Ollama is unreachable.
        if "ollama" in self._local_map:
            try:
                client = self._clients.get("local:ollama")
                if client:
                    discovered = await _md.discover_ollama(client)
                    if discovered:
                        self._local_map["ollama"]["models"] = discovered
                    # else: keep existing seed list — Ollama returned nothing yet
                    summary["ollama"] = len(discovered)
            except Exception as exc:
                log.warning("⚠️  Ollama model discovery failed: %s", exc)
                summary["ollama"] = 0

        # ── OpenRouter free ───────────────────────────────────────────────────
        if "openrouter_free" in self._cloud_map:
            try:
                p = self._cloud_map["openrouter_free"]
                api_key = os.getenv(p.get("api_key_env", ""), "")
                discovered = await _md.discover_openrouter_free(api_key)
                static = p.get("models", [])
                merged = _md.merge_model_lists(static, discovered)
                p["models"] = merged
                summary["openrouter_free"] = len(discovered)
            except Exception as exc:
                log.warning("⚠️  OpenRouter model discovery failed: %s", exc)
                summary["openrouter_free"] = 0

        # ── OpenRouter paid ───────────────────────────────────────────────────
        if "openrouter_paid" in self._cloud_map:
            try:
                p = self._cloud_map["openrouter_paid"]
                api_key = os.getenv(p.get("api_key_env", ""), "")
                # Fetch paid catalogue even without a key (public endpoint),
                # but skip if provider is disabled and no key set.
                if api_key or p.get("enabled", False):
                    discovered = await _md.discover_openrouter_paid(api_key)
                    static = p.get("models", [])
                    merged = _md.merge_model_lists(static, discovered)
                    p["models"] = merged
                    summary["openrouter_paid"] = len(discovered)
            except Exception as exc:
                log.warning("⚠️  OpenRouter paid model discovery failed: %s", exc)
                summary["openrouter_paid"] = 0

        # ── OpenAI (only if key present) ──────────────────────────────────────
        if "openai" in self._cloud_map:
            try:
                p = self._cloud_map["openai"]
                api_key = os.getenv(p.get("api_key_env", ""), "")
                if api_key:
                    client = self._clients.get("cloud:openai")
                    if client:
                        discovered = await _md.discover_openai(client)
                        static = p.get("models", [])
                        merged = _md.merge_model_lists(static, discovered)
                        p["models"] = merged
                        summary["openai"] = len(discovered)
            except Exception as exc:
                log.warning("⚠️  OpenAI model discovery failed: %s", exc)

        # ── Anthropic (only if key present) ──────────────────────────────────
        if "anthropic" in self._cloud_map:
            try:
                p = self._cloud_map["anthropic"]
                api_key = os.getenv(p.get("api_key_env", ""), "")
                if api_key:
                    client = self._clients.get("cloud:anthropic")
                    if client:
                        discovered = await _md.discover_anthropic(client)
                        static = p.get("models", [])
                        merged = _md.merge_model_lists(static, discovered)
                        p["models"] = merged
                        summary["anthropic"] = len(discovered)
            except Exception as exc:
                log.warning("⚠️  Anthropic model discovery failed: %s", exc)

        # ── Groq (only if key present + enabled) ──────────────────────────────
        if "groq" in self._cloud_map:
            try:
                p = self._cloud_map["groq"]
                api_key = os.getenv(p.get("api_key_env", ""), "")
                if api_key and p.get("enabled", False):
                    client = self._clients.get("cloud:groq")
                    if client:
                        discovered = await _md.discover_groq(client)
                        static = p.get("models", [])
                        merged = _md.merge_model_lists(static, discovered)
                        p["models"] = merged
                        summary["groq"] = len(discovered)
            except Exception as exc:
                log.warning("⚠️  Groq model discovery failed: %s", exc)

        self._discovery_at = datetime.utcnow().isoformat() + "Z"
        self._discovery_counts = summary
        total = sum(summary.values())
        log.info("✅  Model discovery complete — %d models across %d providers %s",
                 total, len(summary), summary)
        return summary

    async def shutdown(self) -> None:
        for client in self._clients.values():
            await client.aclose()

    def _client(self, side: str, name: str) -> httpx.AsyncClient:
        key = f"{side}:{name}"
        if key not in self._clients:
            raise HTTPException(status_code=503, detail=f"Provider {name!r} client not initialised")
        return self._clients[key]

    # ── Local dispatch ────────────────────────────────────────────────────────

    async def call_local(self, provider_name: str, body: dict, req_id: str) -> dict:
        p = self.get_local(provider_name)
        if not p:
            raise HTTPException(status_code=503, detail=f"Unknown local provider: {provider_name!r}")
        client = self._client("local", provider_name)
        ptype  = p.get("type", "ollama")
        model  = _resolve_local_model(body.get("model", ""), p)

        if ptype == "ollama":
            payload = _ollama_payload(body, model_override=model)
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            return _ollama_to_openai(resp.json(), model, req_id)
        else:
            # openai_compatible (LM Studio, MLX, llama.cpp)
            resp = await client.post("/chat/completions", json={**body, "model": model})
            resp.raise_for_status()
            return resp.json()

    async def stream_local(self, provider_name: str, body: dict,
                           req_id: str) -> AsyncIterator[str]:
        p = self.get_local(provider_name)
        if not p:
            return
        client = self._client("local", provider_name)
        ptype  = p.get("type", "ollama")
        model  = _resolve_local_model(body.get("model", ""), p)

        if ptype == "ollama":
            payload = _ollama_payload(body, model_override=model, stream=True)
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    done    = chunk.get("done", False)
                    sse = {
                        "id": req_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content} if content else {},
                            "finish_reason": "stop" if done else None,
                        }],
                    }
                    yield f"data: {json.dumps(sse)}\n\n"
                    if done:
                        break
        else:
            # openai_compatible local — stream directly
            stream_body = {**body, "model": model, "stream": True}
            async with client.stream("POST", "/chat/completions", json=stream_body) as resp:
                async for raw in resp.aiter_bytes():
                    yield raw.decode("utf-8", errors="replace")
        yield "data: [DONE]\n\n"

    # ── Cloud dispatch ────────────────────────────────────────────────────────

    async def call_cloud(self, provider_name: str, body: dict, req_id: str) -> dict:
        p = self.get_cloud(provider_name)
        if not p:
            raise HTTPException(status_code=503, detail=f"Unknown cloud provider: {provider_name!r}")
        client = self._client("cloud", provider_name)
        ptype  = p.get("type", "openai")
        _req_model = body.get("model") or ""
        model  = _req_model if (_req_model and _req_model.lower() != "auto") else p.get("default_model", "gpt-4o-mini")

        if ptype == "anthropic":
            return await _call_anthropic(client, body, model, req_id)
        else:
            resp = await client.post("/chat/completions", json={**body, "model": model})
            resp.raise_for_status()
            return resp.json()

    async def stream_cloud(self, provider_name: str, body: dict,
                           req_id: str) -> AsyncIterator[str]:
        p = self.get_cloud(provider_name)
        if not p:
            return
        client = self._client("cloud", provider_name)
        ptype  = p.get("type", "openai")
        _req_model = body.get("model") or ""
        model  = _req_model if (_req_model and _req_model.lower() != "auto") else p.get("default_model", "gpt-4o-mini")

        if ptype == "anthropic":
            async for chunk in _stream_anthropic(client, body, model, req_id):
                yield chunk
        else:
            stream_body = {**body, "model": model, "stream": True}
            async with client.stream("POST", "/chat/completions", json=stream_body) as resp:
                async for raw in resp.aiter_bytes():
                    yield raw.decode("utf-8", errors="replace")
        yield "data: [DONE]\n\n"


# ─── Provider-specific helpers ────────────────────────────────────────────────

def find_provider_for_model(model_id: str) -> tuple[str, str] | None:
    """
    Scan all configured providers to find which one owns this model.
    Returns (side, provider_name) or None if not found.
    Tries exact match, then fuzzy substring match.

    Examples:
      "claude-opus-4-6"                    → ("cloud", "anthropic")
      "gpt-4o"                             → ("cloud", "openai")
      "deepseek/deepseek-r1:free"          → ("cloud", "openrouter_free")
      "gemma3:1b"                          → ("local",  "ollama")
      "grok-3"                             → ("cloud", "xai")
    """
    m = model_id.lower().strip()

    def _match(candidate: str) -> bool:
        c = candidate.lower()
        return c == m or m in c or c in m

    # Cloud first (more specific pinning usually means cloud)
    for p in PROVIDERS_CFG.get("cloud", {}).get("list", []):
        for model in p.get("models", []):
            if _match(model):
                return ("cloud", p["name"])
        # also match on provider name itself as shorthand
        if _match(p["name"]):
            return ("cloud", p["name"])

    # Then local
    for p in PROVIDERS_CFG.get("local", {}).get("list", []):
        for model in p.get("models", []):
            if _match(model):
                return ("local", p["name"])

    return None


def _resolve_local_model(requested: str, provider_cfg: dict) -> str:
    """Use requested model if it's a real local model name, else fall back to provider default.

    Sentinels that should resolve to the provider default:
      - empty / None      — caller didn't specify a model
      - "auto"            — router's wildcard sentinel, not a real model name
      - cloud-prefixed    — gpt-*, claude-*, gemini*, o1-*, o3-* were meant for cloud
    """
    cloud_prefixes = ["gpt-", "claude-", "gemini", "o1-", "o3-"]
    if (requested
            and requested.lower() != "auto"
            and not any(requested.lower().startswith(p) for p in cloud_prefixes)):
        return requested
    return provider_cfg.get("default_model", "llama3.2")


def _ollama_payload(body: dict, model_override: str, stream: bool = False) -> dict:
    return {
        "model": model_override,
        "messages": body.get("messages", []),
        "stream": stream,
        "options": {
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 1.0),
            **({"num_predict": body["max_tokens"]} if "max_tokens" in body else {}),
        },
    }


def _ollama_to_openai(data: dict, model: str, req_id: str) -> dict:
    content = data.get("message", {}).get("content", "")
    pt = data.get("prompt_eval_count", 0)
    ct = data.get("eval_count", 0)
    return {
        "id": req_id, "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


async def _call_anthropic(client: httpx.AsyncClient, body: dict,
                          model: str, req_id: str) -> dict:
    """Convert OpenAI-format body → Anthropic /v1/messages, return OpenAI-format response."""
    messages = body.get("messages", [])
    system_parts, filtered = [], []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            filtered.append(m)

    ant_body: dict = {
        "model": model,
        "max_tokens": body.get("max_tokens", 4096),
        "messages": filtered,
    }
    if system_parts:
        ant_body["system"] = "\n".join(system_parts)
    if "temperature" in body:
        ant_body["temperature"] = body["temperature"]

    resp = await client.post("/v1/messages", json=ant_body)
    resp.raise_for_status()
    data = resp.json()

    content = "".join(
        blk.get("text", "") for blk in data.get("content", [])
        if blk.get("type") == "text"
    )
    usage = data.get("usage", {})
    pt = usage.get("input_tokens", 0)
    ct = usage.get("output_tokens", 0)
    return {
        "id": req_id, "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": data.get("stop_reason", "stop")}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


async def _stream_anthropic(client: httpx.AsyncClient, body: dict,
                             model: str, req_id: str) -> AsyncIterator[str]:
    """Convert Anthropic SSE stream → OpenAI delta SSE format."""
    messages = body.get("messages", [])
    system_parts, filtered = [], []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            filtered.append(m)

    ant_body: dict = {
        "model": model,
        "max_tokens": body.get("max_tokens", 4096),
        "messages": filtered,
        "stream": True,
    }
    if system_parts:
        ant_body["system"] = "\n".join(system_parts)

    async with client.stream("POST", "/v1/messages", json=ant_body) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw in ("[DONE]", ""):
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            etype = event.get("type", "")
            if etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "delta": {"content": text},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            elif etype == "message_stop":
                chunk = {
                    "id": req_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                break


# ─── Logging helper ───────────────────────────────────────────────────────────

def _log_request(
    req_id: str,
    decision: RoutingDecision,
    provider: str,
    duration_ms: float,
    status: int,
    error: str | None = None,
    fallback: bool = False,
    model_used: str | None = None,
    usage: dict | None = None,          # response.usage from the upstream provider
    fallback_reason: str | None = None, # why the previous tier failed
) -> None:

    _model = model_used or decision.model_requested or ""

    # ── Token counts (from upstream response) ─────────────────────────────────
    prompt_tokens     = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    total_tokens      = (usage or {}).get("total_tokens")

    # ── Cost estimate ─────────────────────────────────────────────────────────
    cost_usd: float | None = None
    if usage and _model:
        cost_usd = _estimate_cost(_model, prompt_tokens or 0, completion_tokens or 0)

    # ── Speed (output tokens / second) ───────────────────────────────────────
    speed_tps: float | None = None
    if completion_tokens and duration_ms > 0:
        speed_tps = round(completion_tokens / (duration_ms / 1000), 1)

    # ── Context window utilisation ────────────────────────────────────────────
    ctx_window = _context_window(_model) if _model else None
    ctx_pct: float | None = (
        round((prompt_tokens / ctx_window) * 100, 2)
        if prompt_tokens and ctx_window
        else None
    )

    entry = {
        "id":               req_id,
        "ts":               datetime.utcnow().isoformat() + "Z",
        # ── Routing ───────────────────────────────────────────────────────────
        "backend":          decision.backend.value,
        "provider":         provider,
        "rule":             decision.rule_name,
        "reason":           decision.reason,
        "bucket":           decision.bucket or None,
        "bucket_confidence": round(decision.bucket_confidence, 2) if decision.bucket_confidence else None,
        # ── Models ────────────────────────────────────────────────────────────
        "model_requested":  decision.model_requested,
        "model_used":       _model,
        "model":            _model,          # dashboard compat alias
        # ── Prompt ───────────────────────────────────────────────────────────
        "preview":          decision.prompt_preview,
        "prompt_tokens":    prompt_tokens,   # from upstream response (accurate)
        "input_tokens":     decision.token_count,  # router's pre-send estimate
        # ── Response stats ────────────────────────────────────────────────────
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
        "cost_usd":          cost_usd,
        "speed_tps":         speed_tps,
        # ── Context window ────────────────────────────────────────────────────
        "context_window":   ctx_window,
        "context_pct":      ctx_pct,
        # ── Timing ───────────────────────────────────────────────────────────
        "duration_ms":      round(duration_ms, 1),
        # ── Status ───────────────────────────────────────────────────────────
        "status":           status,
        "fallback":         fallback,
        "fallback_reason":  fallback_reason,
        "error":            error,
    }
    request_log.appendleft(entry)

    stats["total"] += 1
    stats[decision.backend.value] += 1
    if error:
        stats["errors"] += 1
    if fallback:
        stats["fallbacks"] += 1
    stats["rules_hit"][decision.rule_name] = (
        stats["rules_hit"].get(decision.rule_name, 0) + 1
    )

    if OBS_CFG.get("log_routing_decisions", True):
        icon    = "🏠" if decision.backend == Backend.LOCAL else "☁️"
        fb_tag  = f"  [FALLBACK from {fallback_reason[:60]}]" if fallback else ""
        tok_tag = (
            f"  in={prompt_tokens} out={completion_tokens} total={total_tokens}"
            if usage else f"  ~{decision.token_count}tok(est)"
        )
        cost_tag  = f"  ${cost_usd:.5f}" if cost_usd is not None else ""
        speed_tag = f"  {speed_tps:.0f}t/s" if speed_tps else ""
        ctx_tag   = f"  ctx={ctx_pct:.1f}%" if ctx_pct else ""
        log.info(
            "%s  [%s] %s/%s  model=%s  rule=%s  %.0fms%s%s%s%s%s",
            icon, req_id[:8], decision.backend.value, provider, _model or "?",
            decision.rule_name, duration_ms, tok_tag, cost_tag, speed_tag, ctx_tag, fb_tag,
        )


# ─── FastAPI app ──────────────────────────────────────────────────────────────

pm: ProviderManager  # set in lifespan

_discovery_task: asyncio.Task | None = None   # background refresh handle

DISCOVERY_INTERVAL_H: int = 6   # re-discover every N hours


async def _discovery_background_loop() -> None:
    """Re-runs model discovery every DISCOVERY_INTERVAL_H hours."""
    while True:
        await asyncio.sleep(DISCOVERY_INTERVAL_H * 3600)
        log.info("⏰  Background model discovery refresh (every %dh)…", DISCOVERY_INTERVAL_H)
        try:
            await pm.run_discovery()
        except Exception as exc:
            log.warning("⚠️  Background discovery error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pm, _discovery_task
    log.info("🚀  Diksuchi starting — port %s", SERVER_CFG.get("port", 8081))
    pm = ProviderManager(PROVIDERS_CFG)
    await pm.startup()   # ← includes initial model discovery

    # ── Init centroid classifier ────────────────────────────────────────────
    buckets = ROUTING_CFG.get("centroid_buckets", {})
    if buckets:
        _cc.init_classifier(buckets)
    else:
        log.warning("⚠️  No centroid_buckets defined in config.yaml — classifier disabled")

    # ── Background model discovery refresh every 6 h ────────────────────────
    _discovery_task = asyncio.create_task(_discovery_background_loop())

    yield

    _discovery_task.cancel()
    try:
        await _discovery_task
    except asyncio.CancelledError:
        pass
    await pm.shutdown()
    log.info("👋  Diksuchi shut down")


app = FastAPI(
    title="Diksuchi — Local-First AI Router",
    version="3.0.0",
    description="Diksuchi (दिक्सूची): Multi-provider AI router with centroid classification. Supports Ollama, LM Studio, MLX, OpenAI, Anthropic, Groq, Gemini, Together AI, xAI.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER_CFG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve dashboard.html and other static assets ──────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/dashboard", include_in_schema=False)
@app.get("/dashboard.html", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(os.path.join(_BASE_DIR, "dashboard.html"), media_type="text/html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico = os.path.join(_BASE_DIR, "favicon.ico")
    if os.path.exists(ico):
        return FileResponse(ico)
    return JSONResponse({}, status_code=204)


# ─── Core proxy endpoint ──────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_id = "chatcmpl-" + uuid.uuid4().hex[:16]
    t0 = time.perf_counter()

    try:
        body: dict = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    messages = body.get("messages", [])
    model    = body.get("model", "") or ""
    stream   = body.get("stream", False)

    req_headers = {k.lower(): v for k, v in request.headers.items()}

    # Inject runtime per-bucket routing policy into routing config
    _rt_prefs = _load_prefs()
    _rt_cfg   = {
        **ROUTING_CFG,
        "bucket_min_tiers":     _rt_prefs.get("min_tiers", {}),      # legacy floor
        "bucket_allowed_tiers": _rt_prefs.get("allowed_tiers", {}),  # multi-select policy
    }
    decision = decide(messages, model, req_headers, _rt_cfg)

    # ── Model selection (priority order) ─────────────────────────────────────
    # 1. User-pinned model preference (highest priority)
    # 2. Ratings-guided auto-select (best model for bucket from active provider)
    # 3. Requested model from body (pass-through)
    # 4. Provider default_model (lowest priority, fallback inside pm.call_*)
    #
    # Applies when centroid classifier fired OR when a policy-override rule
    # rerouted to a different tier (policy: rules carry a valid bucket).
    model_used: str | None = None   # tracks actual model sent, for logging

    if decision.bucket and (
        decision.rule_name.startswith("centroid:")
        or decision.rule_name.startswith("policy:")
    ):
        prefs = _effective_prefs()
        preferred_model = prefs.get(decision.bucket)

        if preferred_model:
            # ── Path 1: user-pinned model ─────────────────────────────────
            found = find_provider_for_model(preferred_model)
            if found:
                pref_side, pref_provider = found
                decision = RoutingDecision(
                    backend=Backend.LOCAL if pref_side == "local" else Backend.CLOUD,
                    rule_name=f"pref:{decision.bucket}",
                    reason=(
                        f"User preference: {decision.bucket} → {preferred_model} "
                        f"(via {pref_provider})"
                    ),
                    token_count=decision.token_count,
                    prompt_preview=decision.prompt_preview,
                    model_requested=model,
                    provider=pref_provider if pref_side == "cloud" else "",
                    bucket=decision.bucket,
                    bucket_confidence=decision.bucket_confidence,
                )
                body = {**body, "model": preferred_model}
                model_used = preferred_model
                log.info("📌  [pref] bucket=%s → model=%s provider=%s",
                         decision.bucket, preferred_model, pref_provider)
            else:
                log.warning("⚠️  Preferred model %r not found — falling to ratings auto-select",
                            preferred_model)

        if not model_used:
            # ── Path 2: ratings-guided auto-select ────────────────────────
            # Picks the highest-rated model for this bucket from the models
            # actually available in the active provider.
            # Only fires when ratings file exists (user has run a refresh).
            _active_provider = (
                pm.active_local if decision.backend == Backend.LOCAL else pm.active_cloud
            )
            _side = "local" if decision.backend == Backend.LOCAL else "cloud"
            _available = pm.get_models(_side, _active_provider)
            if _available:
                _best = best_model_for_bucket(decision.bucket, _available)
                if _best:
                    body = {**body, "model": _best}
                    model_used = _best
                    log.info("🎯  [ratings-auto] bucket=%s → %s (replacing %r, provider=%s)",
                             decision.bucket, _best, model, _active_provider)

    # If no override was found, use whatever was in the original request.
    # Never keep "auto" as model_used — it's a sentinel, not a real model name.
    # Resolve it to the provider's default_model so logs are meaningful.
    if not model_used:
        _raw = body.get("model") or model or ""
        if _raw and _raw.lower() == "auto":
            # Pick the active provider's default_model as a best-effort label
            if decision.backend == Backend.LOCAL:
                _prov_cfg = pm.get_local(decision.provider or pm.active_local)
            else:
                _prov_cfg = pm.get_cloud(decision.provider or pm.active_cloud)
            model_used = _prov_cfg.get("default_model") or _raw
        else:
            model_used = _raw or None

    # Resolve actual provider name (rule pin > active default)
    if decision.backend == Backend.LOCAL:
        provider_name = decision.provider or pm.active_local
    else:
        provider_name = decision.provider or pm.active_cloud

    # ── 3-TIER FALLBACK CHAIN ─────────────────────────────────────────────────
    # Tier 0 = local, Tier 1 = openrouter_free, Tier 2 = paid cloud
    # On failure, cascade forward through the chain automatically.
    fallback_chain: list[str] = ROUTING_CFG.get("fallback_chain", ["local"])

    # Determine start position in the chain based on routing decision
    if decision.backend == Backend.LOCAL:
        start_tier = 0
    else:
        # Find pinned provider in the chain, default to 1
        try:
            start_tier = fallback_chain.index(provider_name)
        except ValueError:
            start_tier = 1

    # ── Per-bucket fallback guard ─────────────────────────────────────────────
    # If the routing decision carries a bucket (centroid or policy-override),
    # look up the user's allowed_tiers for that bucket and build a set of
    # permitted tier indices.  The fallback loop below will skip any tier not
    # in this set, preventing the chain from cascading into a paid tier the
    # user explicitly excluded (e.g. allowed=[1] must never fall back to tier 2).
    #
    # None  → no restriction; all tiers in the chain are permitted.
    # set() → (shouldn't happen — UI guards prevent empty lists) — treated as None.
    _rt_allowed = _load_prefs().get("allowed_tiers", {})
    _bucket_permitted_tiers: set[int] | None = None
    if decision.bucket and decision.bucket in _rt_allowed:
        _tiers = _rt_allowed[decision.bucket]
        if _tiers:   # non-empty list → enforce
            _bucket_permitted_tiers = set(_tiers)
            log.debug(
                "🔒  [policy] bucket=%s fallback restricted to tiers %s",
                decision.bucket, sorted(_bucket_permitted_tiers),
            )

    # ── LOCAL (Tier 0) ────────────────────────────────────────────────────────
    if decision.backend == Backend.LOCAL:
        if stream:
            # ── Streaming preflight ───────────────────────────────────────
            # Verify local is reachable BEFORE sending HTTP 200 + headers.
            # Once we start streaming, we can't issue a clean fallback if
            # the provider fails mid-response — the client already got the
            # "200 OK" response headers. This quick ping (timeout=3s) lets
            # us cascade to the cloud stream before committing.
            _local_alive = await pm.health_check_local(provider_name)
            if not _local_alive:
                log.warning(
                    "⚡  [%s] Local stream preflight failed (%s down) "
                    "— cascading to cloud stream",
                    req_id[:8], provider_name,
                )
                start_tier = 1   # skip local, stream from cloud instead
            else:
                async def local_stream_gen():
                    async for chunk in pm.stream_local(provider_name, body, req_id):
                        yield chunk
                    _log_request(req_id, decision, provider_name,
                                 (time.perf_counter() - t0) * 1000, 200,
                                 model_used=model_used)
                return StreamingResponse(local_stream_gen(), media_type="text/event-stream")

        if start_tier == 0:   # only try local if we haven't skipped it
            local_error: str | None = None
            try:
                result = await pm.call_local(provider_name, body, req_id)
                _usage   = result.get("usage") or {}
                _dur_ms  = (time.perf_counter() - t0) * 1000
                _mdl     = model_used or ""
                _cost    = _estimate_cost(_mdl, _usage.get("prompt_tokens",0), _usage.get("completion_tokens",0))
                _speed   = round(_usage["completion_tokens"] / (_dur_ms/1000), 1) if _usage.get("completion_tokens") and _dur_ms > 0 else None
                _ctx_w   = _context_window(_mdl)
                _ctx_pct = round(_usage.get("prompt_tokens",0) / _ctx_w * 100, 2) if _usage.get("prompt_tokens") else None
                result["x_router"] = {
                    "backend": "local", "provider": provider_name,
                    "rule": decision.rule_name, "reason": decision.reason,
                    "tokens": decision.token_count,
                    "model_used": model_used,
                    "bucket": decision.bucket, "bucket_confidence": decision.bucket_confidence,
                    "fallback": False,
                    # ── Rich stats ─────────────────────────────────────────────
                    "prompt_tokens":     _usage.get("prompt_tokens"),
                    "completion_tokens": _usage.get("completion_tokens"),
                    "total_tokens":      _usage.get("total_tokens"),
                    "cost_usd":          _cost,
                    "speed_tps":         _speed,
                    "context_window":    _ctx_w,
                    "context_pct":       _ctx_pct,
                    "duration_ms":       round(_dur_ms, 1),
                }
                _log_request(req_id, decision, provider_name, _dur_ms, 200,
                             model_used=model_used, usage=_usage)
                return JSONResponse(result)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                local_error = f"{type(exc).__name__} on local/{provider_name}"
            except httpx.HTTPStatusError as exc:
                local_error = f"HTTP {exc.response.status_code} from local/{provider_name}"

            # Cascade: try each remaining tier in the chain
            log.warning("⚡  [%s] Local failed: %s — cascading fallback chain", req_id[:8], local_error)
            start_tier = 1   # skip local, start from next tier

    # ── CLOUD TIERS (cascade through remaining fallback_chain slots) ──────────
    last_error = ""
    for tier_idx in range(start_tier, len(fallback_chain)):
        tier_provider = fallback_chain[tier_idx]
        if tier_provider == "local":
            continue   # already tried / skipped

        # ── Bucket policy gate ────────────────────────────────────────────────
        # If the user restricted this bucket to specific tiers, skip any tier
        # outside that set — even during fallback.  This prevents e.g. a free-
        # cloud-only policy from silently falling through to a paid provider
        # when the free provider returns a transient 404.
        if _bucket_permitted_tiers is not None and tier_idx not in _bucket_permitted_tiers:
            last_error = (
                f"Tier {tier_idx} ({tier_provider}) skipped — "
                f"not in allowed_tiers {sorted(_bucket_permitted_tiers)} "
                f"for '{decision.bucket}' bucket"
            )
            log.info(
                "🚫  [%s] %s", req_id[:8], last_error,
            )
            continue

        is_fallback = tier_idx > start_tier
        if is_fallback:
            log.warning("⚡  [%s] Tier %d (%s) failed — trying tier %d (%s)",
                        req_id[:8], tier_idx - 1, fallback_chain[tier_idx - 1],
                        tier_idx, tier_provider)

        if stream:
            async def cloud_stream_gen(pname=tier_provider):
                async for chunk in pm.stream_cloud(pname, body, req_id):
                    yield chunk
                _log_request(req_id, decision, pname,
                             (time.perf_counter() - t0) * 1000, 200,
                             model_used=model_used)
            return StreamingResponse(cloud_stream_gen(), media_type="text/event-stream")

        try:
            result = await pm.call_cloud(tier_provider, body, req_id)
            _usage   = result.get("usage") or {}
            _dur_ms  = (time.perf_counter() - t0) * 1000
            _mdl     = model_used or result.get("model") or ""
            _cost    = _estimate_cost(_mdl, _usage.get("prompt_tokens",0), _usage.get("completion_tokens",0))
            _speed   = round(_usage["completion_tokens"] / (_dur_ms/1000), 1) if _usage.get("completion_tokens") and _dur_ms > 0 else None
            _ctx_w   = _context_window(_mdl)
            _ctx_pct = round(_usage.get("prompt_tokens",0) / _ctx_w * 100, 2) if _usage.get("prompt_tokens") else None
            result["x_router"] = {
                "backend": "cloud", "provider": tier_provider,
                "rule": decision.rule_name, "reason": decision.reason,
                "tokens": decision.token_count,
                "model_used": _mdl or model_used,
                "bucket": decision.bucket, "bucket_confidence": decision.bucket_confidence,
                "fallback": is_fallback,
                **({"fallback_reason": last_error} if is_fallback else {}),
                # ── Rich stats ─────────────────────────────────────────────────
                "prompt_tokens":     _usage.get("prompt_tokens"),
                "completion_tokens": _usage.get("completion_tokens"),
                "total_tokens":      _usage.get("total_tokens"),
                "cost_usd":          _cost,
                "speed_tps":         _speed,
                "context_window":    _ctx_w,
                "context_pct":       _ctx_pct,
                "duration_ms":       round(_dur_ms, 1),
            }
            _log_request(req_id, decision, tier_provider, _dur_ms, 200,
                         fallback=is_fallback, model_used=_mdl or model_used,
                         usage=_usage, fallback_reason=last_error if is_fallback else None)
            return JSONResponse(result)
        except httpx.ConnectError as exc:
            last_error = f"Cloud provider {tier_provider!r} unreachable: {exc}"
        except httpx.HTTPStatusError as exc:
            last_error = f"HTTP {exc.response.status_code} from {tier_provider!r}: {exc.response.text[:200]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__} from {tier_provider!r}: {exc}"

    # All tiers exhausted
    _log_request(req_id, decision, provider_name,
                 (time.perf_counter() - t0) * 1000, 503, last_error,
                 model_used=model_used)
    raise HTTPException(status_code=503, detail=f"All providers in fallback chain failed. Last error: {last_error}")


# ─── Route preview (dry-run) ──────────────────────────────────────────────────

@app.post("/v1/route/preview")
async def route_preview(request: Request):
    try:
        body: dict = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    messages = body.get("messages", [{"role": "user", "content": body.get("prompt", "")}])
    model = body.get("model", "")
    req_headers = {k.lower(): v for k, v in request.headers.items()}
    _pv_prefs = _load_prefs()
    _pv_cfg   = {
        **ROUTING_CFG,
        "bucket_min_tiers":     _pv_prefs.get("min_tiers", {}),
        "bucket_allowed_tiers": _pv_prefs.get("allowed_tiers", {}),
    }
    decision = decide(messages, model, req_headers, _pv_cfg)
    provider_name = (decision.provider or
                     (pm.active_local if decision.backend == Backend.LOCAL else pm.active_cloud))
    return {
        "backend": decision.backend.value,
        "provider": provider_name,
        "rule_name": decision.rule_name,
        "reason": decision.reason,
        "token_count": decision.token_count,
        "model_requested": decision.model_requested,
        "prompt_preview": decision.prompt_preview,
        "bucket": decision.bucket or None,
        "bucket_confidence": decision.bucket_confidence or None,
    }


# ─── Provider management endpoints ───────────────────────────────────────────

@app.get("/v1/providers")
async def list_providers():
    """List all configured providers with their status and active flags."""
    def enrich(plist: list[dict], side: str, active: str) -> list[dict]:
        enriched = []
        for p in plist:
            api_key_env = p.get("api_key_env", "")
            has_key = bool(api_key_env and os.getenv(api_key_env, ""))
            # Local providers never need an API key
            if side == "local":
                has_key = True
                api_key_env = ""
            enriched.append({
                "name": p["name"],
                "type": p.get("type", ""),
                "base_url": p.get("base_url", ""),
                "default_model": p.get("default_model", ""),
                "models": p.get("models", []),
                "model_count": len(p.get("models", [])),
                "enabled": p.get("enabled", True),
                "active": p["name"] == active,
                "side": side,
                "has_key": has_key,       # True = API key present in env
                "api_key_env": api_key_env,  # env var name to set (empty for local)
            })
        return enriched
    return {
        "local": {
            "active": pm.active_local,
            "providers": enrich(pm.local_providers(), "local", pm.active_local),
        },
        "cloud": {
            "active": pm.active_cloud,
            "providers": enrich(pm.cloud_providers(), "cloud", pm.active_cloud),
        },
        "discovery": {
            "last_run": pm._discovery_at,
            "next_run_in_h": DISCOVERY_INTERVAL_H,
            "counts": pm._discovery_counts,
        },
    }


@app.post("/v1/providers/refresh")
async def refresh_provider_models():
    """
    Manually trigger live model discovery for all supported providers.
    Useful after installing new Ollama models or enabling new cloud providers.
    Normally runs automatically at startup and every 6 h.
    """
    summary = await pm.run_discovery()
    return {
        "ok": True,
        "discovered_at": pm._discovery_at,
        "counts": summary,
        "tip": "Model lists updated in memory. No restart needed.",
    }


# ─── OpenRouter Mode ─────────────────────────────────────────────────────────
# When enabled: routes all cloud traffic through openrouter_paid (one key for
# Claude, GPT-5, Gemini, Grok, etc.) and updates the in-memory fallback chain.
# When disabled: restores openai as active cloud + original fallback chain.

_or_mode: bool = False
_or_mode_prev_cloud: str = "openai"        # restored on disable
_original_fallback: list[str] = []         # copy of config.yaml fallback_chain

@app.get("/v1/providers/openrouter-mode")
async def get_openrouter_mode():
    return {
        "enabled": _or_mode,
        "active_cloud": pm.active_cloud,
        "fallback_chain": ROUTING_CFG.get("fallback_chain", []),
    }

@app.post("/v1/providers/openrouter-mode")
async def set_openrouter_mode(request: Request):
    global _or_mode, _or_mode_prev_cloud, _original_fallback

    body = await request.json()
    enable = bool(body.get("enabled", False))

    if enable:
        if "openrouter_paid" not in pm._cloud_map:
            raise HTTPException(
                status_code=400,
                detail="openrouter_paid provider not configured in config.yaml"
            )
        api_key = os.getenv(
            pm._cloud_map["openrouter_paid"].get("api_key_env", ""), ""
        )
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="OPENROUTER_API_KEY not set in .env — add it to use OpenRouter Mode"
            )
        # Snapshot current state so we can restore it
        if not _or_mode:
            _or_mode_prev_cloud = pm.active_cloud
            _original_fallback  = list(ROUTING_CFG.get("fallback_chain", []))

        # Enable the paid provider and make it active
        pm._cloud_map["openrouter_paid"]["enabled"] = True
        pm.active_cloud = "openrouter_paid"

        # Swap the fallback chain: replace any direct-provider paid entries
        # with openrouter_paid so that cloud fallback also routes through OR.
        new_chain = []
        _or_paid_added = False
        for entry in (_original_fallback or ["local", "openrouter_free", "openai"]):
            if entry in ("openai", "anthropic", "groq", "gemini", "xai", "mistral"):
                if not _or_paid_added:
                    new_chain.append("openrouter_paid")
                    _or_paid_added = True
                # drop the original paid entry — OR covers it
            else:
                new_chain.append(entry)
        if not _or_paid_added:
            new_chain.append("openrouter_paid")
        ROUTING_CFG["fallback_chain"] = new_chain
        _or_mode = True

        log.info("🌐  OpenRouter Mode ON — active cloud: openrouter_paid, chain: %s", new_chain)
        return {
            "ok": True, "enabled": True,
            "active_cloud": pm.active_cloud,
            "fallback_chain": new_chain,
            "message": "All cloud traffic now routes through OpenRouter.",
        }

    else:
        # Restore previous state
        if _or_mode_prev_cloud and _or_mode_prev_cloud in pm._cloud_map:
            pm.active_cloud = _or_mode_prev_cloud
        if _original_fallback:
            ROUTING_CFG["fallback_chain"] = list(_original_fallback)
        _or_mode = False

        log.info("🔌  OpenRouter Mode OFF — restored cloud: %s, chain: %s",
                 pm.active_cloud, ROUTING_CFG.get("fallback_chain"))
        return {
            "ok": True, "enabled": False,
            "active_cloud": pm.active_cloud,
            "fallback_chain": ROUTING_CFG.get("fallback_chain", []),
            "message": "Restored original provider configuration.",
        }


@app.post("/v1/providers/local/activate/{name}")
async def activate_local_provider(name: str):
    try:
        pm.active_local = name
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True, "active_local": pm.active_local}


@app.post("/v1/providers/cloud/activate/{name}")
async def activate_cloud_provider(name: str):
    try:
        pm.active_cloud = name
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True, "active_cloud": pm.active_cloud}


# ─── Stats, log, config ───────────────────────────────────────────────────────

@app.get("/v1/stats")
async def get_stats():
    total = max(stats["total"], 1)
    return {
        **stats,
        "local_pct":    round(stats["local"]     / total * 100, 1),
        "cloud_pct":    round(stats["cloud"]     / total * 100, 1),
        "fallback_pct": round(stats["fallbacks"] / total * 100, 1),
        "active_local":  pm.active_local,
        "active_cloud":  pm.active_cloud,
    }


@app.get("/v1/log")
async def get_log(limit: int = 50):
    return {"entries": list(request_log)[:limit]}


@app.get("/v1/config")
async def get_config():
    safe = json.loads(json.dumps(CFG))
    for p in safe.get("providers", {}).get("cloud", {}).get("list", []):
        p.pop("api_key_env", None)
    return safe


# ─── Ratings endpoints ────────────────────────────────────────────────────────

@app.get("/v1/ratings")
async def get_ratings():
    """Return currently cached model ratings."""
    ratings = load_ratings()
    if not ratings:
        return {
            "status": "not_fetched",
            "message": "No ratings yet. Trigger a fetch via POST /v1/ratings/refresh, "
                       "the dashboard button, or: make ratings",
            "models": {},
        }
    return {
        "status": "ok",
        "model_count": len(ratings),
        "models": ratings,
    }


@app.post("/v1/ratings/refresh")
async def trigger_ratings_refresh():
    """
    Fetch fresh model ratings from all sources and merge.
    Sources: HF Leaderboard v2, OpenRouter live catalog, bundled baseline.
    User-triggered only — never runs automatically.
    """
    rating_cfg = ROUTING_CFG.get("rating", {})
    weights    = rating_cfg.get("source_weights")
    # Pass the OR key so the catalog fetch can optionally auth for higher rate limits.
    # The public endpoint works without a key too; this is best-effort.
    or_key = os.getenv("OPENROUTER_API_KEY") or None
    try:
        result = await refresh_ratings(weights=weights, or_api_key=or_key)
        return {
            "status": "ok",
            "model_count": result["model_count"],
            "updated_at":  result["updated_at"],
            "sources":     result["sources"],
            "errors":      result["errors"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ratings refresh failed: {exc}")


@app.get("/v1/ratings/best/{bucket}")
async def best_model_for_bucket_endpoint(bucket: str, provider: str = "openrouter_free"):
    """
    Given a capability bucket and provider, return the highest-rated available model.
    Example: GET /v1/ratings/best/coding?provider=openrouter_free
    """
    valid_buckets = {"simple", "coding", "reasoning", "creative", "critical"}
    if bucket not in valid_buckets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bucket {bucket!r}. Valid: {sorted(valid_buckets)}",
        )

    provider_cfg = {}
    for p in PROVIDERS_CFG.get("cloud", {}).get("list", []):
        if p["name"] == provider:
            provider_cfg = p
            break

    available_models = provider_cfg.get("models", [])
    if not available_models:
        raise HTTPException(status_code=404, detail=f"Provider {provider!r} not found or has no models")

    best = best_model_for_bucket(bucket, available_models)
    return {
        "bucket":    bucket,
        "provider":  provider,
        "best_model": best or provider_cfg.get("default_model"),
        "available":  available_models,
    }


# ─── Model preferences endpoints ─────────────────────────────────────────────

VALID_BUCKETS = {"simple", "coding", "reasoning", "creative", "critical"}

@app.get("/v1/preferences")
async def get_preferences():
    """
    Return current per-bucket model preferences.
    Shows config.yaml defaults merged with runtime overrides.
    Model lists come from live ProviderManager maps (post-discovery),
    not from the frozen PROVIDERS_CFG YAML dict.
    """
    prefs = _effective_prefs()
    # Use pm's live maps — these are mutated by run_discovery() so they
    # reflect the actual models returned by each provider's API.
    all_models: list[dict] = []
    for p in pm.cloud_providers():
        api_key_env = p.get("api_key_env", "")
        has_key = bool(api_key_env and os.getenv(api_key_env, ""))
        for m in p.get("models", []):
            all_models.append({
                "model": m,
                "provider": p["name"],
                "side": "cloud",
                "type": p.get("type", ""),
                "has_key": has_key,
            })
    for p in pm.local_providers():
        for m in p.get("models", []):
            all_models.append({
                "model": m,
                "provider": p["name"],
                "side": "local",
                "type": p.get("type", ""),
                "has_key": True,   # local providers never need an API key
            })

    # Per-bucket routing policy overrides (runtime, separate from model pins)
    rt_prefs      = _load_prefs()
    min_tiers     = rt_prefs.get("min_tiers", {})
    allowed_tiers = rt_prefs.get("allowed_tiers", {})
    # Config-defined default tiers (from centroid_buckets)
    cfg_tiers  = {
        b: ROUTING_CFG.get("centroid_buckets", {}).get(b, {}).get("tier", 0)
        for b in VALID_BUCKETS
    }

    def _effective_allowed(bucket: str) -> list[int]:
        """Return the effective allowed-tiers list for a bucket.
        Priority: explicit allowed_tiers override > legacy min_tier > default (all)."""
        if bucket in allowed_tiers:
            return allowed_tiers[bucket]
        # Convert legacy min_tier to allowed list
        min_t = min_tiers.get(bucket, cfg_tiers.get(bucket, 0))
        return [t for t in (0, 1, 2) if t >= min_t]

    return {
        "preferences": {
            b: {"model": prefs.get(b, ""), "auto": not bool(prefs.get(b))}
            for b in VALID_BUCKETS
        },
        "min_tiers": {
            b: min_tiers.get(b, cfg_tiers.get(b, 0))   # legacy — kept for compat
            for b in VALID_BUCKETS
        },
        "allowed_tiers": {
            b: _effective_allowed(b)    # [0,1,2] = all allowed, [1,2] = skip local, etc.
            for b in VALID_BUCKETS
        },
        "config_tiers": cfg_tiers,    # read-only, shows what config.yaml defines
        "available_models": all_models,
        "tip": "POST /v1/preferences with {bucket: model_id} to pin a model. "
               "Use {allowed_tiers: {bucket: [0,1,2]}} to control which tiers are permitted.",
    }


@app.post("/v1/preferences")
async def update_preferences(request: Request):
    """
    Pin one or more buckets to a specific model.

    Body examples:
      {"reasoning": "claude-opus-4-6"}
      {"coding": "codellama", "simple": "gemma3:1b"}
      {"critical": "gpt-4o", "reasoning": "deepseek/deepseek-r1:free"}
      {"reasoning": ""}        ← empty string = reset to auto
    """
    try:
        body: dict = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    # ── Handle min_tiers (legacy floor, kept for backward compat) ──────────────
    min_tiers_update: dict = body.pop("min_tiers", None) or {}
    if min_tiers_update:
        invalid_mt = [k for k in min_tiers_update if k not in VALID_BUCKETS]
        if invalid_mt:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown bucket(s) in min_tiers: {invalid_mt}. Valid: {sorted(VALID_BUCKETS)}",
            )
        invalid_tv = [v for v in min_tiers_update.values() if v not in (0, 1, 2)]
        if invalid_tv:
            raise HTTPException(status_code=400, detail=f"min_tier values must be 0, 1, or 2. Got: {invalid_tv}")

    # ── Handle allowed_tiers (multi-select policy, supersedes min_tiers) ───────
    # Body may include {"allowed_tiers": {"reasoning": [1, 2], "simple": [0, 1]}}
    # Each value is a list of tier indices (0=local, 1=free cloud, 2=paid cloud).
    # An empty list [] means "no restriction" (reset to default = all tiers allowed).
    allowed_tiers_update: dict = body.pop("allowed_tiers", None) or {}
    if allowed_tiers_update:
        invalid_ab = [k for k in allowed_tiers_update if k not in VALID_BUCKETS]
        if invalid_ab:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown bucket(s) in allowed_tiers: {invalid_ab}. Valid: {sorted(VALID_BUCKETS)}",
            )
        for bucket, tiers in allowed_tiers_update.items():
            if not isinstance(tiers, list):
                raise HTTPException(status_code=400, detail=f"allowed_tiers values must be lists. Got {type(tiers)} for '{bucket}'")
            invalid_tv = [v for v in tiers if v not in (0, 1, 2)]
            if invalid_tv:
                raise HTTPException(status_code=400, detail=f"Tier values must be 0, 1, or 2. Got: {invalid_tv} for '{bucket}'")
            if tiers and len(tiers) == 0:
                raise HTTPException(status_code=400, detail=f"allowed_tiers for '{bucket}' cannot be empty — at least one tier required.")

    # ── Handle model pins (remaining body keys) ────────────────────────────────
    invalid = [k for k in body if k not in VALID_BUCKETS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown bucket(s): {invalid}. Valid: {sorted(VALID_BUCKETS)}",
        )

    warnings = []
    for bucket, model_id in body.items():
        if model_id and not find_provider_for_model(model_id):
            warnings.append(
                f"Model {model_id!r} not found in any configured provider — "
                f"preference saved but may not route correctly until provider is enabled."
            )

    prefs = _load_prefs()

    # Update model pins
    prefs.update(body)
    prefs = {k: v for k, v in prefs.items() if v or k == "min_tiers"}

    # Update min_tiers sub-dict (legacy)
    if min_tiers_update:
        existing_mt = prefs.get("min_tiers", {})
        existing_mt.update(min_tiers_update)
        # Normalise: remove entries that match config default (clean file)
        cfg_tiers = {
            b: ROUTING_CFG.get("centroid_buckets", {}).get(b, {}).get("tier", 0)
            for b in VALID_BUCKETS
        }
        existing_mt = {b: v for b, v in existing_mt.items() if v != cfg_tiers.get(b, 0)}
        if existing_mt:
            prefs["min_tiers"] = existing_mt
        else:
            prefs.pop("min_tiers", None)

    # Update allowed_tiers sub-dict
    if allowed_tiers_update:
        existing_at = prefs.get("allowed_tiers", {})
        for bucket, tiers in allowed_tiers_update.items():
            sorted_tiers = sorted(set(tiers))
            if sorted_tiers == [0, 1, 2]:
                # All tiers allowed = default, remove override to keep prefs clean
                existing_at.pop(bucket, None)
            else:
                existing_at[bucket] = sorted_tiers
        if existing_at:
            prefs["allowed_tiers"] = existing_at
        else:
            prefs.pop("allowed_tiers", None)

    _save_prefs(prefs)

    return {
        "ok": True,
        "preferences":   {b: prefs.get(b, "") for b in VALID_BUCKETS},
        "min_tiers":     prefs.get("min_tiers", {}),
        "allowed_tiers": prefs.get("allowed_tiers", {}),
        "warnings": warnings,
    }


@app.delete("/v1/preferences/{bucket}")
async def reset_preference(bucket: str):
    """Reset a single bucket back to auto-select (remove the pin)."""
    if bucket not in VALID_BUCKETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown bucket {bucket!r}. Valid: {sorted(VALID_BUCKETS)}",
        )
    prefs = _load_prefs()
    removed = bucket in prefs
    prefs.pop(bucket, None)
    _save_prefs(prefs)
    return {
        "ok": True,
        "bucket": bucket,
        "reset": removed,
        "preferences": {b: prefs.get(b, "") for b in VALID_BUCKETS},
    }


@app.delete("/v1/preferences")
async def reset_all_preferences():
    """Reset all buckets to auto-select."""
    _save_prefs({})
    return {"ok": True, "message": "All preferences reset to auto-select"}


@app.get("/v1/preferences/resolve/{bucket}")
async def resolve_model_for_bucket(bucket: str):
    """
    Preview which model would be used for a given bucket right now,
    considering preferences + ratings.
    """
    if bucket not in VALID_BUCKETS:
        raise HTTPException(status_code=400, detail=f"Unknown bucket {bucket!r}")

    prefs = _effective_prefs()
    preferred = prefs.get(bucket)

    if preferred:
        found = find_provider_for_model(preferred)
        return {
            "bucket":   bucket,
            "model":    preferred,
            "provider": found[1] if found else "unknown",
            "source":   "user_preference",
        }

    # Fall back to rating-based selection per tier
    from centroid_classifier import get_classifier_tier
    tier = get_classifier_tier(bucket)
    fallback_chain = ROUTING_CFG.get("fallback_chain", ["local"])
    provider_name = fallback_chain[min(tier, len(fallback_chain) - 1)]

    provider_cfg = {}
    side = "local" if provider_name == "local" else "cloud"
    for p in PROVIDERS_CFG.get(side, {}).get("list", []):
        if p["name"] == provider_name:
            provider_cfg = p
            break

    available = provider_cfg.get("models", [])
    best = best_model_for_bucket(bucket, available) or provider_cfg.get("default_model", "")

    return {
        "bucket":   bucket,
        "model":    best,
        "provider": provider_name,
        "source":   "auto_rating",
        "tier":     tier,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok", "version": "2.0.0",
        "active_local": pm.active_local,
        "active_cloud": pm.active_cloud,
    }


@app.get("/")
async def root():
    return {
        "name": "Diksuchi — Local-First AI API Router",
        "version": "3.0.0",
        "endpoints": {
            "chat_completions":          "POST /v1/chat/completions",
            "route_preview":             "POST /v1/route/preview",
            "list_providers":            "GET  /v1/providers",
            "refresh_models":            "POST /v1/providers/refresh",
            "activate_local":            "POST /v1/providers/local/activate/{name}",
            "activate_cloud":            "POST /v1/providers/cloud/activate/{name}",
            "ratings":                   "GET  /v1/ratings",
            "ratings_refresh":           "POST /v1/ratings/refresh",
            "ratings_best_for_bucket":   "GET  /v1/ratings/best/{bucket}",
            "preferences":               "GET  /v1/preferences",
            "set_preferences":           "POST /v1/preferences",
            "reset_preference":          "DELETE /v1/preferences/{bucket}",
            "reset_all_preferences":     "DELETE /v1/preferences",
            "resolve_model":             "GET  /v1/preferences/resolve/{bucket}",
            "stats":                     "GET  /v1/stats",
            "log":                       "GET  /v1/log",
            "config":                    "GET  /v1/config",
            "health":                    "GET  /health",
        },
        "routing": {
            "tiers": ["local (Tier 0, $0)", "openrouter_free (Tier 1, $0)", "paid cloud (Tier 2)"],
            "classifier": "centroid (all-MiniLM-L6-v2, ~8ms)",
            "buckets": ["simple", "coding", "reasoning", "creative", "critical"],
        },
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=SERVER_CFG.get("host", "0.0.0.0"),
        port=SERVER_CFG.get("port", 8080),
        reload=False,
        log_level=SERVER_CFG.get("log_level", "info"),
    )
