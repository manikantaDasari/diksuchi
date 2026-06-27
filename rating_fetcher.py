"""
rating_fetcher.py — On-demand model rating fetcher for Diksuchi.

Four data sources, merged with weighted average:

  Source 1 — HF Open LLM Leaderboard v2  (open-weight models, benchmark-grounded)
             datasets-server.huggingface.co  → open-llm-leaderboard-2/contents
             Falls back to v1 if v2 unavailable.
             Weight: 0.45  |  Coverage: Llama, Qwen, Mistral, DeepSeek, Gemma, etc.

  Source 2 — OpenRouter Live Catalog      (ALL models including newest GPT/Claude/Gemini)
             openrouter.ai/api/v1/models  — no auth required for the public catalog
             Uses pricing + context_length + model family → heuristic capability scores
             Weight: 0.10  |  Coverage: 300+ models, always up-to-date

  Source 3 — Bundled baseline             (ships with Diksuchi, works offline)
             model_ratings_baseline.json  — curated scores for ~70 cloud + local models
             Weight: 0.35  |  Coverage: GPT-5, Claude 4, Gemini 2.5, Grok 3, popular open

  Source 4 — Self-benchmark               (self_benchmark.py, fills gaps locally)
             Weight: 0.10  |  Coverage: locally tested models only

Key property: sources are additive. A model appearing only in Source 2 still gets
a score (via normalized weight). A model in both HF + baseline gets a blended score
where the benchmark-grounded HF data dominates (0.45 vs 0.35).

User-triggered ONLY. Never runs on its own.
Trigger via:  dashboard button  |  make ratings  |  POST /v1/ratings/refresh
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("diksuchi")

RATINGS_FILE  = Path(__file__).parent / "model_ratings.json"
BASELINE_FILE = Path(__file__).parent / "model_ratings_baseline.json"

# ─── HF Leaderboard endpoints ────────────────────────────────────────────────
HF_ENDPOINT = "https://datasets-server.huggingface.co/rows"

HF_PARAMS_V2 = {
    "dataset": "open-llm-leaderboard-2/contents",
    "config":  "default",
    "split":   "train",
    "offset":  0,
    "limit":   200,   # v2 has more models
}

HF_PARAMS_V1 = {   # fallback
    "dataset": "open-llm-leaderboard/contents",
    "config":  "default",
    "split":   "train",
    "offset":  0,
    "limit":   150,
}

# ─── OpenRouter catalog ───────────────────────────────────────────────────────
OR_MODELS_URL = "https://openrouter.ai/api/v1/models"

# ─── Bucket score weights per benchmark field ─────────────────────────────────
# Maps HF benchmark → which capability it signals most
_BUCKET_WEIGHTS = {
    #            IFEval  BBH   MATH  GPQA  MUSR  MMLU  avg
    "simple":   [0.40,  0.10, 0.05, 0.05, 0.05, 0.25, 0.10],
    "coding":   [0.35,  0.35, 0.10, 0.05, 0.05, 0.05, 0.05],
    "reasoning":[0.05,  0.25, 0.25, 0.25, 0.15, 0.00, 0.05],
    "creative": [0.50,  0.05, 0.00, 0.00, 0.00, 0.20, 0.25],
    "critical": [0.05,  0.15, 0.25, 0.35, 0.10, 0.00, 0.10],
}

DEFAULT_WEIGHTS = {
    "hf_leaderboard":  0.55,   # benchmark-grounded, open models
    "bundled_baseline": 0.35,  # curated, authoritative for closed models
    "self_benchmark":   0.10,  # locally tested
}
# OR catalog is NOT a weighted source — it seeds models missing from all other sources.
# This avoids contaminating benchmark/baseline scores with heuristic estimates.


# ─── HF row → scores ─────────────────────────────────────────────────────────

def _hf_row_to_scores(row: dict) -> dict:
    """
    Map HF Leaderboard fields → capability bucket scores (0–1).
    Works for both v1 (Average ⬆️) and v2 (Average) field names.
    """
    raw = {
        "ifeval": float(row.get("IFEval",      0) or 0),
        "bbh":    float(row.get("BBH",         0) or 0),
        "math":   float(row.get("MATH Lvl 5",  0) or 0),
        "gpqa":   float(row.get("GPQA",        0) or 0),
        "musr":   float(row.get("MUSR",        0) or 0),
        "mmlu":   float(row.get("MMLU-PRO",    0) or 0),
        # v2 uses "Average", v1 uses "Average ⬆️"
        "avg":    float(row.get("Average", row.get("Average ⬆️", 0)) or 0),
    }

    # benchmarks reported as 0–100 → normalize to 0–1
    n = {k: min(v / 100.0, 1.0) for k, v in raw.items()}
    fields = [n["ifeval"], n["bbh"], n["math"], n["gpqa"], n["musr"], n["mmlu"], n["avg"]]

    scores: dict[str, Any] = {}
    for bucket, weights in _BUCKET_WEIGHTS.items():
        scores[bucket] = round(sum(f * w for f, w in zip(fields, weights)), 3)

    scores["overall"]  = round(n["avg"], 3)
    scores["params_b"] = float(row.get("#Params (B)", 0) or 0)
    return scores


# ─── Source 1: HF Leaderboard (v2 with v1 fallback) ─────────────────────────

async def fetch_hf_leaderboard(limit: int | None = None) -> list[dict]:
    """
    Fetch top models from HF Open LLM Leaderboard.
    Tries v2 first (open-llm-leaderboard-2), falls back to v1 automatically.
    v2 covers newer models (Llama 4, Qwen 3, Gemma 3, etc.) with same benchmark fields.
    No auth required.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        # ── Try v2 first ─────────────────────────────────────────────────────
        params_v2 = dict(HF_PARAMS_V2)
        if limit is not None:
            params_v2["limit"] = limit
        try:
            resp = await client.get(HF_ENDPOINT, params=params_v2)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("rows", [])
            if rows:
                log.info("   HF Leaderboard v2: %d rows", data.get("num_rows_total", len(rows)))
                return _parse_hf_rows(rows, "hf_leaderboard_v2")
        except Exception as exc:
            log.warning("   HF v2 unavailable (%s) — trying v1 fallback", exc)

        # ── Fallback to v1 ───────────────────────────────────────────────────
        params_v1 = dict(HF_PARAMS_V1)
        if limit is not None:
            params_v1["limit"] = limit
        resp = await client.get(HF_ENDPOINT, params=params_v1)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("rows", [])
        log.info("   HF Leaderboard v1 fallback: %d rows", data.get("num_rows_total", len(rows)))
        return _parse_hf_rows(rows, "hf_leaderboard_v1")


def _parse_hf_rows(rows: list[dict], source_tag: str) -> list[dict]:
    """Convert raw HF API rows to scored model entries."""
    results = []
    for item in rows:
        row = item.get("row", {})
        # v2 uses 'fullname', v1 uses 'Model'
        model_name = (row.get("fullname") or row.get("Model") or "").strip()
        if not model_name:
            continue
        entry = {"model": model_name, "source": source_tag, "side": "local"}
        entry.update(_hf_row_to_scores(row))
        results.append(entry)
    return results


# ─── Source 2: OpenRouter Live Catalog ───────────────────────────────────────

def _price_to_quality(pricing: dict) -> float:
    """
    Convert OpenRouter pricing dict → 0-1 quality estimate.
    Higher price per token generally indicates a more capable model.

    Calibration (approximate OR prices as of 2026):
      free          →  0.20  (exists but highly variable quality)
      $0.15/Mtok    →  0.42  (budget mini: gpt-4o-mini tier)
      $0.80/Mtok    →  0.47  (haiku, flash tier)
      $3.0/Mtok     →  0.55  (mid: claude-sonnet, codestral)
      $5.0/Mtok     →  0.59  (strong: gpt-4o)
      $15/Mtok      →  0.70  (flagship: claude-opus, gpt-5)
      $50/Mtok      →  0.82  (ultra-premium)
      ≥$100/Mtok    →  0.90  (capped — baseline/HF will cover these anyway)

    This is intentionally conservative. OR heuristic scores are only used for
    models not covered by HF benchmarks or the curated baseline.
    """
    try:
        p = float(pricing.get("prompt", "0") or 0)
    except (ValueError, TypeError):
        return 0.35

    if p <= 0:
        return 0.20   # free tier

    # p_per_m = price in $/Million tokens
    p_per_m = p * 1_000_000

    # Formula: 0.40 + 0.50 * log1p(p_per_m) / log1p(100)
    # anchors: log1p(0.15)=0.14, log1p(100)=4.62
    # $0.15→0.415, $3→0.55, $15→0.70, $100→0.90
    raw = 0.40 + 0.50 * math.log1p(p_per_m) / math.log1p(100)
    return round(min(raw, 0.90), 3)


def _family_bias(model_id: str) -> dict[str, float]:
    """
    Per-family bucket bias adjustments (additive to base score).
    These encode well-known model strengths so heuristic scores are
    more directionally useful even without real benchmarks.
    NOTE: order matters — more-specific checks come first.
    """
    m = model_id.lower()

    # DeepSeek reasoning (check before generic -r1 pattern)
    if "deepseek" in m and ("r1" in m or "r2" in m):
        return {"reasoning": +0.09, "critical": +0.07, "coding": +0.05,
                "creative": -0.05, "simple": -0.03}

    # Code specialists (check before generic model patterns)
    if any(x in m for x in ["coder", "codestral", "code-", "starcoder", "deepseek-coder"]):
        return {"coding": +0.10, "critical": +0.04, "reasoning": +0.02,
                "creative": -0.06, "simple": -0.02}

    # Extended reasoning / chain-of-thought specialists (OpenAI o-series, etc.)
    if any(x in m for x in ["o1", "o3", "o4", "-r1", "-r2", "thinking", "reasoner"]):
        return {"reasoning": +0.08, "critical": +0.07, "coding": +0.03,
                "simple": -0.04, "creative": -0.06}

    # Instruction-tuned flagships (Claude, GPT-4+, Gemini Pro)
    if any(x in m for x in ["claude-3", "claude-4", "claude-opus", "claude-sonnet",
                              "gpt-4", "gpt-5", "gemini-2.5-pro", "gemini-exp"]):
        return {"creative": +0.04, "simple": +0.03, "critical": +0.02}

    # Fast / mini models (good at simple, worse at complex)
    if any(x in m for x in ["mini", "nano", "flash", "haiku", "tiny", "small", "3b", "1b"]):
        return {"simple": +0.03, "creative": +0.02, "reasoning": -0.04, "critical": -0.05}

    return {}


async def fetch_openrouter_catalog(api_key: str | None = None) -> list[dict]:
    """
    Fetch the live OpenRouter model catalog and convert to heuristic capability scores.

    Uses pricing + context_length + model family as quality proxies.
    The public endpoint (no auth) returns all 300+ available models including
    GPT-5, Claude 4, Gemini 2.5, Grok 3, and every new model added to OR.

    Weights are intentionally low (0.10) so these heuristic scores only matter
    for models not covered by HF benchmarks or the curated baseline.
    If a better source already has the model, OR heuristics contribute only 10%.
    """
    headers: dict = {"User-Agent": "Diksuchi/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        resp = await client.get(OR_MODELS_URL)
        resp.raise_for_status()
        data = resp.json()

    results: list[dict] = []
    for m in data.get("data", []):
        model_id = (m.get("id") or "").strip()
        if not model_id:
            continue

        pricing = m.get("pricing") or {}
        base    = _price_to_quality(pricing)
        bias    = _family_bias(model_id)

        def _s(bucket: str) -> float:
            return round(max(0.05, min(1.0, base + bias.get(bucket, 0.0))), 3)

        entry = {
            "model":     model_id,
            "source":    "openrouter_catalog",
            "side":      "cloud",
            "simple":    _s("simple"),
            "coding":    _s("coding"),
            "reasoning": _s("reasoning"),
            "creative":  _s("creative"),
            "critical":  _s("critical"),
            "overall":   round(base, 3),
            # repurpose params_b to store context length (÷1000 to keep same numeric range)
            "params_b":  round(float(m.get("context_length") or 0) / 1000, 1),
        }
        results.append(entry)

    return results


# ─── Source 3: Bundled baseline ───────────────────────────────────────────────

def load_baseline() -> list[dict]:
    """Load bundled baseline ratings. Works offline, ships with Diksuchi."""
    if not BASELINE_FILE.exists():
        log.warning("📦  Baseline file not found: %s", BASELINE_FILE)
        return []
    data = json.loads(BASELINE_FILE.read_text())
    return data.get("models", [])


# ─── Merge ────────────────────────────────────────────────────────────────────

def merge_ratings(
    hf_models:  list[dict],
    baseline:   list[dict],
    self_bench: list[dict] | None = None,
    or_catalog: list[dict] | None = None,
    weights:    dict | None = None,
) -> dict:
    """
    Merge model scores from all sources.

    STEP 1 — Weighted blend of benchmark-grounded sources (HF + baseline + self-bench).
              Models appearing in multiple sources get a weighted average.
              Models in only one source get that source's score (normalized to 100%).

    STEP 2 — OR catalog seed pass (gap filler ONLY).
              Any model in the OR catalog that is NOT already in the blended result
              gets an entry with heuristic scores.
              OR catalog scores NEVER overwrite or blend with benchmark data.
              This ensures: new models that just appeared on OpenRouter get at least
              some score estimate, without dragging down scores for known models.

    Returns {model_id: scored_entry}.
    """
    w = weights or DEFAULT_WEIGHTS

    # ── Step 1: weighted blend of benchmark sources ───────────────────────────
    sources: list[tuple[list[dict], float]] = [
        (hf_models,         w.get("hf_leaderboard",  0.55)),
        (baseline,          w.get("bundled_baseline", 0.35)),
        (self_bench or [],  w.get("self_benchmark",  0.10)),
    ]

    acc: dict[str, dict] = {}

    for records, sw in sources:
        for m in records:
            model_id = (m.get("model") or "").strip()
            if not model_id:
                continue
            if model_id not in acc:
                acc[model_id] = {
                    "model":      model_id,
                    "sources":    [],
                    "weight_sum": 0.0,
                    "side":       m.get("side", "cloud"),
                    "simple":    0.0, "coding":    0.0, "reasoning": 0.0,
                    "creative":  0.0, "critical":  0.0, "overall":   0.0,
                }
            e = acc[model_id]
            e["sources"].append(m.get("source", "unknown"))
            e["weight_sum"] += sw
            if m.get("side"):
                e["side"] = m["side"]
            for field in ("simple", "coding", "reasoning", "creative", "critical", "overall"):
                e[field] += m.get(field, 0.0) * sw

    # Normalize weighted sums → final scores
    merged: dict[str, dict] = {}
    for model_id, e in acc.items():
        ws = e.pop("weight_sum", 1.0) or 1.0
        for field in ("simple", "coding", "reasoning", "creative", "critical", "overall"):
            e[field] = round(e[field] / ws, 3)
        e["best_bucket"] = max(
            ("simple", "coding", "reasoning", "creative", "critical"),
            key=lambda f: e[f],   # noqa: B023
        )
        merged[model_id] = e

    # ── Step 2: OR catalog seed pass (gap filler) ─────────────────────────────
    # Add heuristic scores ONLY for models not covered by any benchmark source.
    if or_catalog:
        _BUCKET_FIELDS = ("simple", "coding", "reasoning", "creative", "critical")
        for m in or_catalog:
            model_id = (m.get("model") or "").strip()
            if not model_id or model_id in merged:
                continue   # already have benchmark/baseline data — don't overwrite
            entry: dict[str, Any] = {
                "model":   model_id,
                "sources": [m.get("source", "openrouter_catalog")],
                "side":    m.get("side", "cloud"),
            }
            for field in (*_BUCKET_FIELDS, "overall"):
                entry[field] = round(float(m.get(field, 0.0)), 3)
            entry["best_bucket"] = max(_BUCKET_FIELDS, key=lambda f: entry[f])  # noqa: B023
            merged[model_id] = entry

    return merged


# ─── Main entry point ─────────────────────────────────────────────────────────

async def refresh_ratings(
    weights: dict | None = None,
    or_api_key: str | None = None,
) -> dict:
    """
    Fetch all sources → merge → persist to model_ratings.json.
    Called by: dashboard button / make ratings / POST /v1/ratings/refresh

    Args:
        weights:    Optional weight overrides (dict matching DEFAULT_WEIGHTS keys).
        or_api_key: Optional OpenRouter API key. The public catalog endpoint works
                    without auth, but a key may unlock higher rate limits.
    """
    t0 = time.perf_counter()
    errors: list[str] = []

    # ── Source 1: HF Leaderboard (v2 → v1 fallback) ──────────────────────────
    hf_models: list[dict] = []
    try:
        log.info("📡  [1/3] Fetching HF Open LLM Leaderboard (v2 with v1 fallback)…")
        hf_models = await fetch_hf_leaderboard()
        log.info("   ✅ HF Leaderboard: %d models", len(hf_models))
    except Exception as exc:
        msg = f"HF Leaderboard fetch failed: {exc}"
        errors.append(msg)
        log.warning("   ⚠️ %s", msg)

    # ── Source 2: OpenRouter live catalog ────────────────────────────────────
    or_catalog: list[dict] = []
    try:
        log.info("📡  [2/3] Fetching OpenRouter live catalog…")
        or_catalog = await fetch_openrouter_catalog(api_key=or_api_key)
        log.info("   ✅ OpenRouter catalog: %d models", len(or_catalog))
    except Exception as exc:
        msg = f"OpenRouter catalog fetch failed: {exc}"
        errors.append(msg)
        log.warning("   ⚠️ %s", msg)

    # ── Source 3: Bundled baseline ────────────────────────────────────────────
    baseline = load_baseline()
    log.info("   📦 Baseline: %d models", len(baseline))

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = merge_ratings(hf_models, baseline, or_catalog=or_catalog, weights=weights)

    # Count how many OR-only models were seeded (appeared only in OR catalog)
    or_seeded = sum(
        1 for e in merged.values()
        if e.get("sources") == ["openrouter_catalog"]
    )

    output = {
        "updated_at":  datetime.now(timezone.utc).isoformat(),
        "model_count": len(merged),
        "sources": {
            "hf_leaderboard":         len(hf_models),
            "openrouter_catalog_raw": len(or_catalog),
            "openrouter_seeded":      or_seeded,   # models ONLY from OR catalog
            "baseline":               len(baseline),
        },
        "errors": errors,
        "models": merged,
    }

    RATINGS_FILE.write_text(json.dumps(output, indent=2))
    ms = (time.perf_counter() - t0) * 1000
    log.info(
        "✅  Ratings saved → %s  (%d models: %d benchmark-grounded, %d OR-seeded, %.0fms)",
        RATINGS_FILE.name, len(merged),
        len(merged) - or_seeded, or_seeded, ms,
    )
    return output


# ─── Read helpers (used at request-time) ─────────────────────────────────────

def load_ratings() -> dict:
    """Load persisted ratings. Returns {} if never fetched."""
    if not RATINGS_FILE.exists():
        return {}
    try:
        return json.loads(RATINGS_FILE.read_text()).get("models", {})
    except Exception:
        return {}


def best_model_for_bucket(bucket: str, available_models: list[str]) -> str | None:
    """
    Given a bucket and a list of available model IDs, return the highest-rated one.
    Uses fuzzy matching so "llama-3.1-8b-instruct:free" matches "Llama-3.1-8B-Instruct".
    """
    ratings = load_ratings()
    if not ratings or not available_models:
        return None

    scored: list[tuple[str, float]] = []
    for model_id in available_models:
        entry = ratings.get(model_id)

        if not entry:
            # fuzzy: find any rating key that overlaps with model_id
            m_lower = model_id.lower()
            for key, val in ratings.items():
                k_lower = key.lower()
                # match on significant substring (avoid accidental short matches)
                if len(k_lower) > 8 and (m_lower in k_lower or k_lower in m_lower):
                    entry = val
                    break

        if entry:
            scored.append((model_id, float(entry.get(bucket, 0.0))))

    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# ─── CLI helper ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _main():
        result = await refresh_ratings()
        s = result["sources"]
        print(f"\n✅ Done — {result['model_count']} models rated")
        print(f"   HF Leaderboard:            {s['hf_leaderboard']} models (benchmark-grounded)")
        print(f"   OpenRouter catalog fetched: {s['openrouter_catalog_raw']} models")
        print(f"   OpenRouter seeded:          {s['openrouter_seeded']} new models (gap-fill only)")
        print(f"   Baseline:                   {s['baseline']} models")
        if result["errors"]:
            print(f"   Errors: {result['errors']}")

    asyncio.run(_main())
