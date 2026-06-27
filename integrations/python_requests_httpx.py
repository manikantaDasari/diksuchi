"""
python_requests_httpx.py
─────────────────────────
Use the router with plain requests or httpx — no OpenAI SDK needed.
Useful when you want minimal dependencies or build your own wrapper.
"""

import json
import httpx
import requests

ROUTER = "http://localhost:8080"


# ─── requests ────────────────────────────────────────────────────────────────

def chat_requests(prompt: str, model: str = "llama3.2") -> str:
    resp = requests.post(
        f"{ROUTER}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    router_info = data.get("x_router", {})
    print(f"  → backend={router_info.get('backend')}  provider={router_info.get('provider')}  rule={router_info.get('rule')}")
    return data["choices"][0]["message"]["content"]


# ─── httpx (sync) ────────────────────────────────────────────────────────────

def chat_httpx(prompt: str, model: str = "") -> str:
    with httpx.Client(base_url=ROUTER, timeout=120) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ─── httpx async + streaming ─────────────────────────────────────────────────

import asyncio

async def stream_httpx(prompt: str, model: str = "llama3.2"):
    """Stream tokens from the router using SSE."""
    async with httpx.AsyncClient(base_url=ROUTER, timeout=120) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
                except (json.JSONDecodeError, KeyError):
                    pass
    print()


# ─── Utility: get current routing stats ──────────────────────────────────────

def get_stats() -> dict:
    return requests.get(f"{ROUTER}/v1/stats").json()


def get_providers() -> dict:
    return requests.get(f"{ROUTER}/v1/providers").json()


def switch_provider(side: str, name: str) -> dict:
    """Switch active local or cloud provider at runtime."""
    return requests.post(f"{ROUTER}/v1/providers/{side}/activate/{name}").json()


def route_preview(prompt: str, model: str = "") -> dict:
    """See where a request would route WITHOUT sending it."""
    return requests.post(
        f"{ROUTER}/v1/route/preview",
        json={"messages": [{"role": "user", "content": prompt}], "model": model},
    ).json()


# ─── Example: build a simple CLI chatbot ─────────────────────────────────────

def cli_chatbot():
    print("=== Diksuchi CLI Chatbot ===")
    stats = get_stats()
    print(f"Router online. Active: local={stats['active_local']}  cloud={stats['active_cloud']}")
    print("Type 'exit' to quit, 'stats' for routing stats\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "stats":
            s = get_stats()
            print(f"  Total={s['total']}  Local={s['local']} ({s['local_pct']}%)  Cloud={s['cloud']} ({s['cloud_pct']}%)  Fallbacks={s['fallbacks']}")
            continue

        history.append({"role": "user", "content": user_input})

        resp = requests.post(
            f"{ROUTER}/v1/chat/completions",
            json={"model": "", "messages": history},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        router_info = data.get("x_router", {})

        print(f"[{router_info.get('backend','?')}/{router_info.get('provider','?')}] ", end="")
        print(f"Assistant: {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    # Quick test
    print("=== requests ===")
    result = chat_requests("What is Python's GIL in one sentence?")
    print(result, "\n")

    print("=== httpx ===")
    result = chat_httpx("Name three sorting algorithms.")
    print(result, "\n")

    print("=== streaming ===")
    asyncio.run(stream_httpx("Count from 1 to 5 slowly."))

    print("\n=== route preview ===")
    preview = route_preview("Write a comprehensive analysis of machine learning architectures")
    print(f"Would route to: {preview['backend']}/{preview['provider']} — {preview['reason']}")

    print("\n=== stats ===")
    s = get_stats()
    print(f"Local: {s['local']} ({s['local_pct']}%)  Cloud: {s['cloud']} ({s['cloud_pct']}%)")
