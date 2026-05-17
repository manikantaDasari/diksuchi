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
from fastapi.responses import JSONResponse, StreamingResponse

from router_engine import Backend, RoutingDecision, decide

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
log = logging.getLogger("ai-router")


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
        model  = body.get("model") or p.get("default_model", "gpt-4o-mini")

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
        model  = body.get("model") or p.get("default_model", "gpt-4o-mini")

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

def _resolve_local_model(requested: str, provider_cfg: dict) -> str:
    """Use requested model if it's not a cloud-only name, else fall back to provider default."""
    cloud_prefixes = ["gpt-", "claude-", "gemini", "o1-", "o3-"]
    if requested and not any(requested.lower().startswith(p) for p in cloud_prefixes):
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

def _log_request(req_id: str, decision: RoutingDecision, provider: str,
                 duration_ms: float, status: int,
                 error: str | None = None, fallback: bool = False) -> None:
    entry = {
        "id": req_id,
        "ts": datetime.utcnow().isoformat() + "Z",
        "backend": decision.backend.value,
        "provider": provider,
        "rule": decision.rule_name,
        "reason": decision.reason,
        "tokens": decision.token_count,
        "model": decision.model_requested,
        "preview": decision.prompt_preview,
        "duration_ms": round(duration_ms, 1),
        "status": status,
        "error": error,
        "fallback": fallback,
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
        icon = "🏠" if decision.backend == Backend.LOCAL else "☁️"
        fb_tag = "  [FALLBACK→cloud]" if fallback else ""
        log.info("%s  [%s] %s/%s  rule=%s  tokens=%d  %.0fms%s",
                 icon, req_id[:8], decision.backend.value, provider,
                 decision.rule_name, decision.token_count, duration_ms, fb_tag)


# ─── FastAPI app ──────────────────────────────────────────────────────────────

pm: ProviderManager  # set in lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pm
    log.info("🚀  AI Router starting — port %s", SERVER_CFG.get("port", 8080))
    pm = ProviderManager(PROVIDERS_CFG)
    await pm.startup()
    yield
    await pm.shutdown()
    log.info("👋  AI Router shut down")


app = FastAPI(
    title="Local-First AI API Router",
    version="2.0.0",
    description="Multi-provider AI router: Ollama, LM Studio, MLX, OpenAI, Anthropic, Groq, Gemini, Together AI",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER_CFG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    decision = decide(messages, model, req_headers, ROUTING_CFG)

    # Resolve actual provider name (rule pin > active default)
    if decision.backend == Backend.LOCAL:
        provider_name = decision.provider or pm.active_local
    else:
        provider_name = decision.provider or pm.active_cloud

    # ── LOCAL ─────────────────────────────────────────────────────────────────
    if decision.backend == Backend.LOCAL:
        if stream:
            async def local_stream_gen():
                async for chunk in pm.stream_local(provider_name, body, req_id):
                    yield chunk
                _log_request(req_id, decision, provider_name,
                             (time.perf_counter() - t0) * 1000, 200)
            return StreamingResponse(local_stream_gen(), media_type="text/event-stream")

        local_error: str | None = None
        try:
            result = await pm.call_local(provider_name, body, req_id)
            result["x_router"] = {
                "backend": "local", "provider": provider_name,
                "rule": decision.rule_name, "reason": decision.reason,
                "tokens": decision.token_count, "fallback": False,
            }
            _log_request(req_id, decision, provider_name,
                         (time.perf_counter() - t0) * 1000, 200)
            return JSONResponse(result)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            local_error = f"{type(exc).__name__} on local/{provider_name} — falling back to cloud"
        except httpx.HTTPStatusError as exc:
            local_error = f"HTTP {exc.response.status_code} from local/{provider_name} — falling back to cloud"

        # Automatic fallback → cloud
        log.warning("⚡  [%s] %s", req_id[:8], local_error)
        fallback_provider = pm.active_cloud
        try:
            result = await pm.call_cloud(fallback_provider, body, req_id)
            result["x_router"] = {
                "backend": "cloud", "provider": fallback_provider,
                "rule": decision.rule_name, "reason": decision.reason,
                "tokens": decision.token_count, "fallback": True,
                "fallback_reason": local_error,
            }
            _log_request(req_id, decision, fallback_provider,
                         (time.perf_counter() - t0) * 1000, 200, fallback=True)
            return JSONResponse(result)
        except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
            err = f"Both local and cloud failed: {exc}"
            _log_request(req_id, decision, fallback_provider,
                         (time.perf_counter() - t0) * 1000, 503, err)
            raise HTTPException(status_code=503, detail=err)

    # ── CLOUD ─────────────────────────────────────────────────────────────────
    if stream:
        async def cloud_stream_gen():
            async for chunk in pm.stream_cloud(provider_name, body, req_id):
                yield chunk
            _log_request(req_id, decision, provider_name,
                         (time.perf_counter() - t0) * 1000, 200)
        return StreamingResponse(cloud_stream_gen(), media_type="text/event-stream")

    try:
        result = await pm.call_cloud(provider_name, body, req_id)
        result["x_router"] = {
            "backend": "cloud", "provider": provider_name,
            "rule": decision.rule_name, "reason": decision.reason,
            "tokens": decision.token_count, "fallback": False,
        }
        _log_request(req_id, decision, provider_name,
                     (time.perf_counter() - t0) * 1000, 200)
        return JSONResponse(result)
    except httpx.ConnectError:
        err = f"Cloud provider {provider_name!r} unreachable"
        _log_request(req_id, decision, provider_name,
                     (time.perf_counter() - t0) * 1000, 503, err)
        raise HTTPException(status_code=503, detail=err)
    except httpx.HTTPStatusError as exc:
        body_text = exc.response.text
        _log_request(req_id, decision, provider_name,
                     (time.perf_counter() - t0) * 1000, exc.response.status_code, body_text[:200])
        raise HTTPException(status_code=exc.response.status_code, detail=body_text)


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
    decision = decide(messages, model, req_headers, ROUTING_CFG)
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
    }


# ─── Provider management endpoints ───────────────────────────────────────────

@app.get("/v1/providers")
async def list_providers():
    """List all configured providers with their status and active flags."""
    def enrich(plist: list[dict], side: str, active: str) -> list[dict]:
        return [
            {
                "name": p["name"],
                "type": p.get("type", ""),
                "base_url": p.get("base_url", ""),
                "default_model": p.get("default_model", ""),
                "models": p.get("models", []),
                "enabled": p.get("enabled", True),
                "active": p["name"] == active,
                "side": side,
            }
            for p in plist
        ]
    return {
        "local": {
            "active": pm.active_local,
            "providers": enrich(pm.local_providers(), "local", pm.active_local),
        },
        "cloud": {
            "active": pm.active_cloud,
            "providers": enrich(pm.cloud_providers(), "cloud", pm.active_cloud),
        },
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
        "name": "Local-First AI API Router",
        "version": "2.0.0",
        "endpoints": {
            "chat_completions":      "POST /v1/chat/completions",
            "route_preview":         "POST /v1/route/preview",
            "list_providers":        "GET  /v1/providers",
            "activate_local":        "POST /v1/providers/local/activate/{name}",
            "activate_cloud":        "POST /v1/providers/cloud/activate/{name}",
            "stats":                 "GET  /v1/stats",
            "log":                   "GET  /v1/log",
            "config":                "GET  /v1/config",
            "health":                "GET  /health",
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
