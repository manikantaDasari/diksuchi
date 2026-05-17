"""
python_openai_sdk.py
────────────────────
Use the official OpenAI Python SDK against your local AI Router.
The router is fully OpenAI-compatible, so this is a one-line change:
  base_url="http://localhost:8080/v1"

Run:  pip install openai
      python python_openai_sdk.py
"""

from openai import OpenAI

# ── Drop-in replacement: just change base_url ─────────────────────────────────
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="router",          # any non-empty string; router handles the real keys
)

# ── 1. Basic chat ─────────────────────────────────────────────────────────────
def basic_chat():
    response = client.chat.completions.create(
        model="llama3.2",      # router auto-picks local/cloud based on rules
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user",   "content": "What is 2 + 2?"},
        ],
    )
    print("[basic_chat]")
    print(response.choices[0].message.content)
    print("Backend:", response.model_extra.get("x_router", {}).get("backend"))
    print("Provider:", response.model_extra.get("x_router", {}).get("provider"))
    print()


# ── 2. Streaming ──────────────────────────────────────────────────────────────
def streaming_chat():
    print("[streaming_chat]")
    stream = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print("\n")


# ── 3. Force a specific backend via header ────────────────────────────────────
def force_cloud():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize quantum entanglement."}],
        extra_headers={"X-Route-To": "cloud"},   # override routing rules
    )
    print("[force_cloud]")
    print(response.choices[0].message.content[:200])
    print()


# ── 4. Force a specific provider via header ───────────────────────────────────
def force_provider():
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Write a Python hello world."}],
        extra_headers={
            "X-Route-To": "cloud",
            "X-Provider":  "groq",   # hint (router uses it if you wire the header rule)
        },
    )
    print("[force_provider]")
    print(response.choices[0].message.content[:300])
    print()


# ── 5. Async usage ────────────────────────────────────────────────────────────
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="router",
)

async def async_chat():
    response = await async_client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hello in 3 languages."}],
    )
    print("[async_chat]")
    print(response.choices[0].message.content)
    print()


# ── 6. Code generation with routing preview ───────────────────────────────────
import httpx, json

def route_preview(prompt: str, model: str = "") -> dict:
    """Check where a prompt would route WITHOUT sending it."""
    r = httpx.post(
        "http://localhost:8080/v1/route/preview",
        json={"messages": [{"role": "user", "content": prompt}], "model": model},
    )
    return r.json()

def smart_code_gen(task: str):
    # Check routing first
    preview = route_preview(task)
    print(f"[smart_code_gen] Routing → {preview['backend']}/{preview['provider']}  ({preview['reason']})")

    response = client.chat.completions.create(
        model="",
        messages=[
            {"role": "system", "content": "Write clean, commented Python code only."},
            {"role": "user",   "content": task},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    basic_chat()
    streaming_chat()

    code = smart_code_gen("Write a function that reads a CSV and returns a pandas DataFrame")
    print("[code result]\n", code[:500])

    asyncio.run(async_chat())
