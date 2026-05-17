"""
test_runner.py
──────────────────────────────────────────────────────────────────────────────
Automated test suite for the AI Router.
No pytest required — plain Python. Prints a colour-coded report.

Usage:
    python3 tests/test_runner.py                  # all tests
    python3 tests/test_runner.py routing          # only routing tests
    python3 tests/test_runner.py chat             # only chat tests
    python3 tests/test_runner.py providers        # only provider tests

Requires:
    pip install requests
    Router must be running (python main.py)
    Mock backends optional but recommended for chat tests
"""

from __future__ import annotations
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import requests

ROUTER = "http://localhost:8080"
FILTER = sys.argv[1] if len(sys.argv) > 1 else "all"

# ─── Colours ──────────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; C = "\033[96m"; Y = "\033[93m"; W = "\033[0m"


@dataclass
class Result:
    name: str
    group: str
    passed: bool
    error: str = ""
    duration_ms: float = 0.0


results: list[Result] = []


def test(name: str, group: str = "misc"):
    """Decorator to register a test function."""
    def decorator(fn: Callable):
        def wrapper():
            if FILTER not in ("all", group):
                return
            t0 = time.perf_counter()
            try:
                fn()
                dur = (time.perf_counter() - t0) * 1000
                results.append(Result(name, group, True, duration_ms=round(dur, 1)))
                print(f"  {G}✓{W}  {name}  {Y}({dur:.0f}ms){W}")
            except AssertionError as e:
                dur = (time.perf_counter() - t0) * 1000
                msg = str(e)
                results.append(Result(name, group, False, msg, round(dur, 1)))
                print(f"  {R}✗{W}  {name}")
                print(f"      {R}{msg}{W}")
            except Exception as e:
                dur = (time.perf_counter() - t0) * 1000
                msg = f"{type(e).__name__}: {e}"
                results.append(Result(name, group, False, msg, round(dur, 1)))
                print(f"  {R}✗{W}  {name}")
                print(f"      {R}{msg}{W}")
        wrapper()
        return fn
    return decorator


def section(title: str):
    if FILTER == "all" or any(title.lower().startswith(FILTER) for _ in [1]):
        print(f"\n{C}━━━  {title}  ━━━{W}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def preview(prompt: str, model: str = "", headers: dict | None = None) -> dict:
    r = requests.post(
        f"{ROUTER}/v1/route/preview",
        json={"messages": [{"role": "user", "content": prompt}], "model": model},
        headers=headers or {},
        timeout=10,
    )
    assert r.status_code == 200, f"preview returned {r.status_code}: {r.text[:200]}"
    return r.json()


def chat(prompt: str, model: str = "", headers: dict | None = None, timeout: int = 30) -> dict:
    r = requests.post(
        f"{ROUTER}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        headers={"Content-Type": "application/json", **(headers or {})},
        timeout=timeout,
    )
    assert r.status_code == 200, f"chat returned {r.status_code}: {r.text[:300]}"
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Health
# ══════════════════════════════════════════════════════════════════════════════

section("1. Health & Connectivity")

@test("Router is reachable", "health")
def _():
    r = requests.get(f"{ROUTER}/health", timeout=5)
    assert r.status_code == 200, f"Got {r.status_code}"
    d = r.json()
    assert d["status"] == "ok", f"status={d['status']}"

@test("Root endpoint returns endpoint map", "health")
def _():
    r = requests.get(f"{ROUTER}/", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "chat_completions" in d["endpoints"]
    assert "list_providers" in d["endpoints"]

@test("Stats endpoint is accessible", "health")
def _():
    r = requests.get(f"{ROUTER}/v1/stats", timeout=5)
    assert r.status_code == 200
    d = r.json()
    for key in ("total", "local", "cloud", "errors", "fallbacks", "active_local", "active_cloud"):
        assert key in d, f"Missing key: {key}"

@test("Config endpoint returns sanitised config", "health")
def _():
    r = requests.get(f"{ROUTER}/v1/config", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "providers" in d
    assert "routing" in d
    # API keys must not leak
    for p in d["providers"].get("cloud", {}).get("list", []):
        assert "api_key_env" not in p, f"api_key_env leaked for {p.get('name')}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Routing Rules (no LLM needed — uses /v1/route/preview)
# ══════════════════════════════════════════════════════════════════════════════

section("2. Routing Rules  (dry-run, no LLM)")

@test("Short prompt → local", "routing")
def _():
    d = preview("Hi")
    assert d["backend"] == "local", f"Expected local, got {d['backend']} (rule={d['rule_name']})"

@test("Long prompt (>800 tokens) → cloud", "routing")
def _():
    long_prompt = "word " * 810
    d = preview(long_prompt)
    assert d["backend"] == "cloud", f"Expected cloud, got {d['backend']}"
    assert d["token_count"] >= 800, f"token_count={d['token_count']}"

@test("Complexity keyword → cloud", "routing")
def _():
    d = preview("Write a comprehensive financial model for a SaaS startup")
    assert d["backend"] == "cloud", f"Expected cloud, got {d['backend']}"

@test("Simple keyword → local", "routing")
def _():
    d = preview("Quick question: what is 2+2?")
    assert d["backend"] == "local", f"Expected local, got {d['backend']}"

@test("gpt-4o model hint → cloud", "routing")
def _():
    d = preview("Hello", model="gpt-4o")
    assert d["backend"] == "cloud"
    assert d["rule_name"] == "model_hint_cloud"

@test("llama3.2 model hint → local", "routing")
def _():
    d = preview("Hello", model="llama3.2")
    assert d["backend"] == "local"
    assert d["rule_name"] == "model_hint_local"

@test("X-Route-To: cloud header overrides rules", "routing")
def _():
    d = preview("Quick question", headers={"X-Route-To": "cloud"})
    assert d["backend"] == "cloud"
    assert d["rule_name"] == "force_cloud_header"

@test("X-Route-To: local header overrides rules", "routing")
def _():
    d = preview("Write a comprehensive analysis", headers={"X-Route-To": "local"})
    assert d["backend"] == "local"
    assert d["rule_name"] == "force_local_header"

@test("Preview response has all required fields", "routing")
def _():
    d = preview("Test prompt")
    for key in ("backend", "provider", "rule_name", "reason", "token_count", "model_requested", "prompt_preview"):
        assert key in d, f"Missing field: {key}"

@test("Token count is plausible", "routing")
def _():
    d = preview("This is a ten word sentence that should have roughly ten tokens")
    assert 5 <= d["token_count"] <= 20, f"token_count={d['token_count']} seems wrong"

@test("Provider field is populated in preview", "routing")
def _():
    d = preview("Hello")
    assert d["provider"], "provider should not be empty"

@test("Empty messages → valid response", "routing")
def _():
    r = requests.post(f"{ROUTER}/v1/route/preview",
                      json={"messages": [], "model": ""}, timeout=10)
    assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# 3. Chat Completions (needs mock or real backends)
# ══════════════════════════════════════════════════════════════════════════════

section("3. Chat Completions  (needs backends)")

@test("Chat returns valid OpenAI-format response", "chat")
def _():
    d = chat("Say hello.", headers={"X-Route-To": "local"})
    assert "choices" in d and len(d["choices"]) > 0
    assert "message" in d["choices"][0]
    assert "content" in d["choices"][0]["message"]
    assert d["choices"][0]["message"]["content"], "Empty response"

@test("Response contains x_router metadata", "chat")
def _():
    d = chat("ping", headers={"X-Route-To": "local"})
    assert "x_router" in d, "x_router metadata missing"
    xr = d["x_router"]
    for key in ("backend", "provider", "rule", "tokens"):
        assert key in xr, f"x_router missing key: {key}"

@test("x_router.backend = local when forced local", "chat")
def _():
    d = chat("test", headers={"X-Route-To": "local"})
    assert d["x_router"]["backend"] == "local"

@test("x_router.backend = cloud when forced cloud", "chat")
def _():
    d = chat("test", headers={"X-Route-To": "cloud"})
    assert d["x_router"]["backend"] == "cloud"

@test("x_router.fallback = False on normal call", "chat")
def _():
    d = chat("test", headers={"X-Route-To": "local"})
    assert d["x_router"].get("fallback") == False

@test("Chat increments stats.total", "chat")
def _():
    before = requests.get(f"{ROUTER}/v1/stats").json()["total"]
    chat("increment test", headers={"X-Route-To": "local"})
    after = requests.get(f"{ROUTER}/v1/stats").json()["total"]
    assert after > before, f"total didn't increment: {before} → {after}"

@test("Chat with system message works", "chat")
def _():
    r = requests.post(f"{ROUTER}/v1/chat/completions",
        json={"model": "llama3.2", "messages": [
            {"role": "system", "content": "You are a helpful bot."},
            {"role": "user",   "content": "Say hi."},
        ]},
        headers={"Content-Type": "application/json", "X-Route-To": "local"},
        timeout=30,
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"]

@test("Invalid JSON returns 400", "chat")
def _():
    r = requests.post(f"{ROUTER}/v1/chat/completions",
                      data="NOT_JSON",
                      headers={"Content-Type": "application/json"},
                      timeout=5)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"

@test("Request log is updated after chat", "chat")
def _():
    chat("log test", headers={"X-Route-To": "local"})
    log = requests.get(f"{ROUTER}/v1/log?limit=5").json()
    assert len(log["entries"]) > 0
    entry = log["entries"][0]
    for key in ("id", "ts", "backend", "provider", "rule", "tokens", "duration_ms"):
        assert key in entry, f"Log entry missing: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Provider Management
# ══════════════════════════════════════════════════════════════════════════════

section("4. Provider Management")

@test("GET /v1/providers returns local + cloud", "providers")
def _():
    r = requests.get(f"{ROUTER}/v1/providers", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "local" in d and "cloud" in d
    assert len(d["local"]["providers"]) >= 1
    assert len(d["cloud"]["providers"]) >= 1

@test("Active provider is marked in provider list", "providers")
def _():
    d = requests.get(f"{ROUTER}/v1/providers").json()
    active_local = d["local"]["active"]
    actives = [p for p in d["local"]["providers"] if p["active"]]
    assert len(actives) == 1, f"Expected exactly 1 active local provider, got {len(actives)}"
    assert actives[0]["name"] == active_local

@test("Switch cloud provider to groq", "providers")
def _():
    r = requests.post(f"{ROUTER}/v1/providers/cloud/activate/groq")
    assert r.status_code == 200
    assert r.json()["active_cloud"] == "groq"
    # Confirm stats also reflects new active
    stats = requests.get(f"{ROUTER}/v1/stats").json()
    assert stats["active_cloud"] == "groq"

@test("Switch cloud provider back to openai", "providers")
def _():
    r = requests.post(f"{ROUTER}/v1/providers/cloud/activate/openai")
    assert r.status_code == 200
    assert r.json()["active_cloud"] == "openai"

@test("Activating unknown provider returns 404", "providers")
def _():
    r = requests.post(f"{ROUTER}/v1/providers/cloud/activate/nonexistent_xyz")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"

@test("Switching local provider to lmstudio", "providers")
def _():
    r = requests.post(f"{ROUTER}/v1/providers/local/activate/lmstudio")
    assert r.status_code == 200
    assert r.json()["active_local"] == "lmstudio"

@test("Restore local provider to ollama", "providers")
def _():
    r = requests.post(f"{ROUTER}/v1/providers/local/activate/ollama")
    assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# 5. Stats Integrity
# ══════════════════════════════════════════════════════════════════════════════

section("5. Stats Integrity")

@test("local + cloud counts ≤ total", "stats")
def _():
    d = requests.get(f"{ROUTER}/v1/stats").json()
    assert d["local"] + d["cloud"] <= d["total"] + 5  # small tolerance for race

@test("Percentages are between 0 and 100", "stats")
def _():
    d = requests.get(f"{ROUTER}/v1/stats").json()
    for key in ("local_pct", "cloud_pct", "fallback_pct"):
        assert 0 <= d[key] <= 100, f"{key}={d[key]} out of range"

@test("rules_hit is a dict of rule names to counts", "stats")
def _():
    d = requests.get(f"{ROUTER}/v1/stats").json()
    assert isinstance(d["rules_hit"], dict)
    for k, v in d["rules_hit"].items():
        assert isinstance(k, str)
        assert isinstance(v, int) and v >= 0


# ══════════════════════════════════════════════════════════════════════════════
# Print summary
# ══════════════════════════════════════════════════════════════════════════════

relevant = [r for r in results if FILTER == "all" or r.group == FILTER]
if not relevant:
    print(f"\n  {Y}No tests matched filter '{FILTER}'{W}")
    sys.exit(0)

passed  = [r for r in relevant if r.passed]
failed  = [r for r in relevant if not r.passed]
avg_dur = sum(r.duration_ms for r in relevant) / len(relevant)

print(f"\n{C}━━━  Summary  ━━━{W}")
print(f"  Tests run:  {len(relevant)}")
print(f"  {G}Passed:  {len(passed)}{W}")
if failed:
    print(f"  {R}Failed:  {len(failed)}{W}")
    print(f"\n  {R}Failures:{W}")
    for r in failed:
        print(f"    {R}✗{W} [{r.group}] {r.name}")
        print(f"      {R}{r.error}{W}")
print(f"  Avg duration: {avg_dur:.0f}ms")

if failed:
    print(f"\n  {R}Some tests failed.{W}\n")
    sys.exit(1)
else:
    print(f"\n  {G}All {len(passed)} tests passed! ✓{W}\n")
