"""
self_benchmark.py — Self-benchmark runner for Diksuchi model rating.

Sends 10 canonical prompts (2 per capability bucket) to each active provider,
then uses the cheapest available local model as a judge to score responses 1–10.
Fills rating gaps for new/unknown models not on HF Leaderboard or baseline.

Trigger via:  make benchmark  |  POST /v1/ratings/benchmark
Writes results to: self_benchmark_results.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

log = logging.getLogger("diksuchi")

RESULTS_FILE = Path(__file__).parent / "self_benchmark_results.json"

# ─── Canonical benchmark prompts ─────────────────────────────────────────────
# 2 prompts per bucket — designed to clearly exercise each capability.
# Judge scores on correctness, depth, and helpfulness (1–10).

BENCHMARK_PROMPTS: dict[str, list[dict]] = {
    "simple": [
        {
            "id": "simple_1",
            "prompt": "What is the capital of Japan?",
            "expected_keywords": ["tokyo", "Tokyo"],
        },
        {
            "id": "simple_2",
            "prompt": "Translate 'hello world' to Spanish.",
            "expected_keywords": ["hola", "mundo"],
        },
    ],
    "coding": [
        {
            "id": "coding_1",
            "prompt": (
                "Write a Python function called `flatten` that takes a nested list "
                "(e.g. [1, [2, [3, 4]], 5]) and returns a flat list ([1, 2, 3, 4, 5]). "
                "Include a brief docstring."
            ),
            "expected_keywords": ["def flatten", "def ", "return", "docstring"],
        },
        {
            "id": "coding_2",
            "prompt": (
                "Here is a buggy Python function:\n\n"
                "def divide(a, b):\n    return a / b\n\n"
                "What is the bug? Show the fixed version with proper error handling."
            ),
            "expected_keywords": ["ZeroDivisionError", "try", "except", "ValueError"],
        },
    ],
    "reasoning": [
        {
            "id": "reasoning_1",
            "prompt": (
                "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. "
                "How much does the ball cost? Show your reasoning step by step."
            ),
            "expected_keywords": ["$0.05", "0.05", "five cents"],
        },
        {
            "id": "reasoning_2",
            "prompt": (
                "Compare and contrast REST and GraphQL APIs. When would you choose one over the other? "
                "Consider: flexibility, performance, caching, and developer experience."
            ),
            "expected_keywords": ["REST", "GraphQL", "overfetch", "schema", "endpoint"],
        },
    ],
    "creative": [
        {
            "id": "creative_1",
            "prompt": (
                "Write a two-sentence product description for a reusable water bottle "
                "targeting eco-conscious millennials. Make it punchy and memorable."
            ),
            "expected_keywords": [],  # creative — judge scores on quality
        },
        {
            "id": "creative_2",
            "prompt": (
                "Draft a short email (3–4 sentences) to a client explaining that "
                "their project deadline is moving from Friday to Monday. "
                "Be professional but friendly."
            ),
            "expected_keywords": ["Monday", "apologize", "delay"],
        },
    ],
    "critical": [
        {
            "id": "critical_1",
            "prompt": (
                "Review this Python authentication code for security issues:\n\n"
                "def login(username, password):\n"
                "    query = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"\n"
                "    return db.execute(query)\n\n"
                "List every vulnerability and how to fix each one."
            ),
            "expected_keywords": ["SQL injection", "injection", "parameterized", "hash", "bcrypt"],
        },
        {
            "id": "critical_2",
            "prompt": (
                "You are designing a payment processing system that must handle 10,000 "
                "transactions per second with 99.99% uptime. Outline the key architectural "
                "decisions you would make and the trade-offs involved."
            ),
            "expected_keywords": ["queue", "idempotent", "ACID", "database", "scale"],
        },
    ],
}

# Judge prompt — sent to the local model to score each response
JUDGE_TEMPLATE = """You are an expert AI evaluator. Score the following response on a scale of 1 to 10.

Task: {prompt}

Response: {response}

Scoring criteria:
- 9-10: Excellent, accurate, complete, well-explained
- 7-8:  Good, mostly correct with minor gaps
- 5-6:  Acceptable but missing important elements
- 3-4:  Partial, significant errors or omissions
- 1-2:  Wrong or irrelevant

Reply with ONLY a single integer from 1 to 10. No explanation."""


def _extract_score(text: str) -> float:
    """Extract numeric score from judge response."""
    text = text.strip()
    # try first token as integer
    match = re.search(r"\b([1-9]|10)\b", text)
    if match:
        return float(match.group(1))
    return 5.0   # neutral fallback


def _scores_to_buckets(results: list[dict]) -> dict[str, float]:
    """Average scores per bucket, normalize to 0–1."""
    bucket_scores: dict[str, list[float]] = {}
    for r in results:
        bucket = r["bucket"]
        bucket_scores.setdefault(bucket, []).append(r["score"])

    return {
        bucket: round(sum(scores) / len(scores) / 10.0, 3)
        for bucket, scores in bucket_scores.items()
    }


# ─── Benchmark runner ─────────────────────────────────────────────────────────

class SelfBenchmark:
    def __init__(self, ollama_url: str = "http://localhost:11434",
                 local_model: str = "gemma3:1b"):
        self.ollama_url = ollama_url
        self.judge_model = local_model

    async def _call_ollama(self, model: str, prompt: str) -> str:
        """Call Ollama for generation or judging."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")

    async def _call_provider(self, base_url: str, model: str,
                             prompt: str, api_key: str = "",
                             extra_headers: dict | None = None) -> str:
        """Call an OpenAI-compatible endpoint."""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _judge(self, prompt: str, response: str) -> float:
        """Score a response using the local judge model."""
        judge_prompt = JUDGE_TEMPLATE.format(prompt=prompt, response=response[:1500])
        try:
            answer = await self._call_ollama(self.judge_model, judge_prompt)
            return _extract_score(answer)
        except Exception as exc:
            log.warning("Judge call failed: %s — using neutral score", exc)
            return 5.0

    async def run_for_provider(
        self,
        provider_name: str,
        base_url: str,
        model: str,
        api_key: str = "",
        extra_headers: dict | None = None,
        is_ollama: bool = False,
    ) -> dict:
        """Run all benchmark prompts against one provider/model pair."""
        log.info("🔬  Benchmarking %s / %s…", provider_name, model)
        results = []
        errors = 0

        for bucket, prompts in BENCHMARK_PROMPTS.items():
            for item in prompts:
                try:
                    t0 = time.perf_counter()
                    if is_ollama:
                        response = await self._call_ollama(model, item["prompt"])
                    else:
                        response = await self._call_provider(
                            base_url, model, item["prompt"], api_key, extra_headers
                        )
                    latency = round((time.perf_counter() - t0) * 1000)

                    score = await self._judge(item["prompt"], response)
                    log.debug(
                        "  %s/%s → score=%.0f  latency=%dms",
                        bucket, item["id"], score, latency,
                    )
                    results.append({
                        "prompt_id": item["id"],
                        "bucket":    bucket,
                        "score":     score,
                        "latency_ms": latency,
                    })
                except Exception as exc:
                    log.warning("  ⚠️ %s/%s failed: %s", bucket, item["id"], exc)
                    errors += 1
                    results.append({
                        "prompt_id": item["id"],
                        "bucket":    bucket,
                        "score":     0.0,
                        "latency_ms": 0,
                        "error":     str(exc),
                    })

        bucket_scores = _scores_to_buckets(results)
        return {
            "model":          model,
            "provider":       provider_name,
            "source":         "self_benchmark",
            "benchmarked_at": datetime.now(timezone.utc).isoformat(),
            "errors":         errors,
            "results":        results,
            **bucket_scores,
            "overall": round(sum(bucket_scores.values()) / max(len(bucket_scores), 1), 3),
            "best_bucket": max(bucket_scores, key=bucket_scores.__getitem__) if bucket_scores else "simple",
        }

    async def run_all(self, providers: list[dict]) -> list[dict]:
        """Run benchmark for all given providers concurrently (with semaphore)."""
        sem = asyncio.Semaphore(2)   # max 2 providers in parallel

        async def limited(p):
            async with sem:
                return await self.run_for_provider(**p)

        tasks = [limited(p) for p in providers]
        return await asyncio.gather(*tasks, return_exceptions=False)


async def run_benchmark(providers: list[dict],
                        ollama_url: str = "http://localhost:11434",
                        judge_model: str = "gemma3:1b") -> list[dict]:
    """
    Main entry point.
    providers: list of dicts with keys: provider_name, base_url, model, api_key,
               extra_headers (optional), is_ollama (optional bool)
    """
    bm = SelfBenchmark(ollama_url=ollama_url, local_model=judge_model)
    t0 = time.perf_counter()
    results = await bm.run_all(providers)
    ms = (time.perf_counter() - t0) * 1000

    output = {
        "benchmarked_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms":    round(ms),
        "models":         results,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    log.info("✅  Self-benchmark complete — %d models in %.0fms", len(results), ms)
    return results


def load_self_benchmark_results() -> list[dict]:
    """Load previously run self-benchmark results."""
    if not RESULTS_FILE.exists():
        return []
    try:
        return json.loads(RESULTS_FILE.read_text()).get("models", [])
    except Exception:
        return []


# ─── CLI helper ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Example: benchmark local Ollama + OpenRouter free
    example_providers = [
        {
            "provider_name": "ollama",
            "base_url": "http://localhost:11434",
            "model": "gemma3:1b",
            "api_key": "",
            "is_ollama": True,
        },
        {
            "provider_name": "openrouter_free",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "extra_headers": {
                "HTTP-Referer": "https://github.com/manikantaDasari/diksuchi",
                "X-Title": "Diksuchi",
            },
            "is_ollama": False,
        },
    ]

    async def _main():
        results = await run_benchmark(example_providers)
        for r in results:
            print(f"\n{r['provider']} / {r['model']}")
            for bucket in ("simple", "coding", "reasoning", "creative", "critical"):
                bar = "█" * int(r.get(bucket, 0) * 20)
                print(f"  {bucket:10s}  {bar:<20s} {r.get(bucket, 0):.2f}")

    asyncio.run(_main())
