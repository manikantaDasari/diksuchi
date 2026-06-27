"""
model_discovery.py — Live model-list discovery for Diksuchi.

Runs at startup (and every 6 h in the background) to keep each provider's
model list current without touching config.yaml.

Supported providers and their discovery endpoints:
  ollama          → GET /api/tags              (local, no auth)
  openrouter_free → GET /api/v1/models         (public, no auth; filters to :free / price==0)
  openai          → GET /v1/models             (requires OPENAI_API_KEY)
  groq            → GET /openai/v1/models      (requires GROQ_API_KEY)
  anthropic       → no models endpoint         (static list only)
  gemini          → no OpenAI-compat endpoint  (static list only)
  xai             → no public models endpoint  (static list only)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass

log = logging.getLogger("diksuchi")

# ── OpenRouter: chat-capable model ID prefixes we care about ─────────────────
# (excludes image-gen, audio, embedding models)
_OR_SKIP_KEYWORDS = (
    "stable-diffusion", "dall-e", "whisper", "tts-", "embedding",
    "clip", "blip", "midjourney", "flux", "imagen",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-provider discovery functions
# ─────────────────────────────────────────────────────────────────────────────

async def discover_ollama(client: httpx.AsyncClient) -> list[str]:
    """
    Fetch locally installed Ollama models via GET /api/tags.
    Returns a sorted list of model name strings (e.g. ["gemma3:1b", "llama3.2"]).
    """
    resp = await client.get("/api/tags", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    models = sorted(m["name"] for m in data.get("models", []))
    log.info("🔍  Ollama: discovered %d installed models", len(models))
    return models


async def discover_openrouter_free(api_key: str = "") -> list[str]:
    """
    Fetch all models from OpenRouter's public catalogue and return only the
    free ones (pricing.prompt == "0" and pricing.completion == "0", or ID ends
    in :free).

    No authentication required — the models endpoint is public.
    """
    headers = {
        "HTTP-Referer": "https://github.com/manikantaDasari/diksuchi",
        "X-Title": "Diksuchi",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    free_models: list[str] = []
    for m in data.get("data", []):
        mid: str = m.get("id", "")
        if not mid:
            continue

        # Skip non-chat model types
        if any(kw in mid.lower() for kw in _OR_SKIP_KEYWORDS):
            continue

        pricing = m.get("pricing", {})
        prompt_price     = str(pricing.get("prompt",     "1") or "1")
        completion_price = str(pricing.get("completion", "1") or "1")
        is_truly_free = (prompt_price == "0" and completion_price == "0")
        has_free_suffix = mid.endswith(":free")

        if is_truly_free or has_free_suffix:
            # Normalise: prefer the :free suffix variant for routing consistency
            if not has_free_suffix and is_truly_free:
                mid = mid + ":free"
            free_models.append(mid)

    # Deduplicate (API sometimes lists both variants)
    seen: set[str] = set()
    deduped = [m for m in free_models if not (m in seen or seen.add(m))]
    deduped.sort()

    log.info("🔍  OpenRouter: discovered %d free models (from %d total)",
             len(deduped), len(data.get("data", [])))
    return deduped


async def discover_openrouter_paid(api_key: str = "") -> list[str]:
    """
    Fetch all PAID models from OpenRouter's catalogue.

    Complements discover_openrouter_free() — returns everything that is NOT
    free, i.e. models where pricing > 0 and no ':free' suffix.  This lets
    users access OpenAI, Anthropic, Google, xAI, Mistral, etc. through a
    single OPENROUTER_API_KEY instead of juggling individual provider keys.

    Results are sorted by provider prefix first (anthropic/ → google/ →
    meta-llama/ → mistralai/ → openai/ → x-ai/ → everything else),
    then by model name within each provider — so the dashboard optgroup
    reads like a provider catalogue.

    No authentication required for the models endpoint, but api_key is
    forwarded if present (gives access to rate-limit-only models).
    """
    headers = {
        "HTTP-Referer": "https://github.com/manikantaDasari/diksuchi",
        "X-Title": "Diksuchi",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    # Provider sort order — well-known providers first
    _PROVIDER_ORDER = [
        "anthropic/", "openai/", "google/", "x-ai/",
        "meta-llama/", "mistralai/", "deepseek/", "qwen/",
        "cohere/", "perplexity/",
    ]

    def _provider_rank(mid: str) -> tuple[int, str]:
        for i, prefix in enumerate(_PROVIDER_ORDER):
            if mid.startswith(prefix):
                return (i, mid)
        return (len(_PROVIDER_ORDER), mid)

    # Collect (context_length, sort_rank, mid) tuples for clean sorting
    paid_candidates: list[tuple[int, tuple[int, str], str]] = []
    for m in data.get("data", []):
        mid: str = m.get("id", "")
        if not mid:
            continue

        # Skip non-chat model types (image-gen, audio, embeddings…)
        if any(kw in mid.lower() for kw in _OR_SKIP_KEYWORDS):
            continue

        # Skip :free models — those belong to discover_openrouter_free()
        if mid.endswith(":free"):
            continue

        pricing = m.get("pricing", {})
        prompt_price     = str(pricing.get("prompt",     "0") or "0")
        completion_price = str(pricing.get("completion", "0") or "0")
        is_truly_free = (prompt_price == "0" and completion_price == "0")
        if is_truly_free:
            continue   # already in the free list; avoid duplicates

        ctx = int(m.get("context_length", 0) or 0)
        paid_candidates.append((ctx, _provider_rank(mid), mid))

    # Deduplicate by model id
    seen: set[str] = set()
    unique: list[tuple[int, tuple[int, str], str]] = []
    for entry in paid_candidates:
        if entry[2] not in seen:
            seen.add(entry[2])
            unique.append(entry)

    # Sort: largest context first (best models), then by provider order within ties
    unique.sort(key=lambda t: (-t[0], t[1]))

    # Cap at top 50 — keeps the dropdown manageable
    TOP_N = 50
    top = unique[:TOP_N]

    # Re-sort the final list by provider order for a readable optgroup experience
    top.sort(key=lambda t: t[1])
    result = [t[2] for t in top]

    log.info("🔍  OpenRouter paid: top %d models by context (from %d total)",
             len(result), len(data.get("data", [])))
    return result


async def discover_openai(client: httpx.AsyncClient) -> list[str]:
    """
    Fetch OpenAI's model catalogue and filter to current chat-capable models.
    Requires OPENAI_API_KEY to be set.
    """
    resp = await client.get("/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    CHAT_PREFIXES = ("gpt-5", "gpt-4", "o1", "o3", "o4", "chatgpt")
    SKIP_SUFFIXES = ("-instruct", "-base", "-preview-turbo")
    SKIP_KEYWORDS = ("embedding", "tts", "whisper", "dall-e", "moderation",
                     "babbage", "davinci", "curie", "ada", "ft-")

    models: list[str] = []
    for m in data.get("data", []):
        mid: str = m.get("id", "")
        if not any(mid.startswith(p) for p in CHAT_PREFIXES):
            continue
        if any(kw in mid for kw in SKIP_KEYWORDS):
            continue
        if any(mid.endswith(s) for s in SKIP_SUFFIXES):
            continue
        models.append(mid)

    models.sort(reverse=True)   # newest first (lexicographic — gpt-4.1 > gpt-4o)
    log.info("🔍  OpenAI: discovered %d chat models", len(models))
    return models


async def discover_anthropic(client: httpx.AsyncClient) -> list[str]:
    """
    Fetch Anthropic's available models via GET /v1/models.
    Requires ANTHROPIC_API_KEY (set via x-api-key header on the client).

    Response shape:
      {"data": [{"id": "claude-opus-4-6", "display_name": "...", "type": "model", ...}]}

    Only returns "model" type entries (excludes any non-model objects).
    Sorted newest-first (claude-sonnet-4-6 > claude-3-5-haiku-...).
    """
    resp = await client.get("/v1/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    models: list[str] = [
        m["id"]
        for m in data.get("data", [])
        if m.get("type", "model") == "model" and m.get("id", "")
    ]
    models.sort(reverse=True)   # newest first lexicographically
    log.info("🔍  Anthropic: discovered %d models", len(models))
    return models


async def discover_groq(client: httpx.AsyncClient) -> list[str]:
    """
    Fetch Groq's available models. Requires GROQ_API_KEY.
    Returns all model IDs (Groq's catalogue is small and all chat-capable).
    """
    resp = await client.get("/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    models = sorted(m["id"] for m in data.get("data", []))
    log.info("🔍  Groq: discovered %d models", len(models))
    return models


# ─────────────────────────────────────────────────────────────────────────────
#  Merge helper
# ─────────────────────────────────────────────────────────────────────────────

def merge_model_lists(static: list[str], discovered: list[str]) -> list[str]:
    """
    Merge static config models with discovered models.
    Discovered models are placed first (live data takes priority),
    static models not already in the list are appended as fallback.
    Result is deduplicated, preserving order.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for m in (*discovered, *static):
        if m not in seen:
            seen.add(m)
            merged.append(m)
    return merged
