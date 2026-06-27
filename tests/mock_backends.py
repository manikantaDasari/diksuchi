"""
mock_backends.py
─────────────────────────────────────────────────────────────────────────────
Fake local (Ollama) + cloud (OpenAI) backends for testing Diksuchi
without needing real LLMs or API keys.

Run this BEFORE starting the router, then update config.yaml:
  local.base_url  → http://localhost:11435   (fake Ollama)
  cloud.base_url  → http://localhost:11436/v1 (fake OpenAI)

Or just run the included start_test_env.sh which patches config automatically.

Usage:
    pip install fastapi uvicorn
    python tests/mock_backends.py
"""

import json
import random
import time
import asyncio
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ─── Fake Ollama (port 11435) ─────────────────────────────────────────────────

ollama_app = FastAPI(title="Mock Ollama")

OLLAMA_RESPONSES = [
    "I am the local Ollama mock. Your message was received.",
    "Mock local response: Everything looks great!",
    "Fake Ollama here — routing is working correctly.",
    "Local inference mock: request processed successfully.",
]

@ollama_app.get("/api/tags")
async def ollama_tags():
    """Health endpoint — used by Docker Compose healthcheck."""
    return {
        "models": [
            {"name": "llama3.2", "size": 2_000_000_000},
            {"name": "mistral",  "size": 4_000_000_000},
            {"name": "phi4",     "size": 3_000_000_000},
        ]
    }

@ollama_app.post("/api/chat")
async def ollama_chat(request: Request):
    body = await request.json()
    model    = body.get("model", "llama3.2")
    messages = body.get("messages", [])
    stream   = body.get("stream", False)
    user_msg = messages[-1].get("content", "") if messages else ""

    reply = random.choice(OLLAMA_RESPONSES) + f" (echo: '{user_msg[:40]}')"

    if stream:
        async def stream_gen():
            words = reply.split()
            for i, word in enumerate(words):
                chunk = {
                    "model": model,
                    "message": {"role": "assistant", "content": word + " "},
                    "done": i == len(words) - 1,
                    "prompt_eval_count": len(user_msg.split()),
                    "eval_count": i + 1,
                }
                yield json.dumps(chunk) + "\n"
                await asyncio.sleep(0.03)
        return StreamingResponse(stream_gen(), media_type="application/x-ndjson")

    return JSONResponse({
        "model": model,
        "message": {"role": "assistant", "content": reply},
        "done": True,
        "prompt_eval_count": len(user_msg.split()),
        "eval_count": len(reply.split()),
    })


# ─── Fake OpenAI (port 11436) ─────────────────────────────────────────────────

openai_app = FastAPI(title="Mock OpenAI")

CLOUD_RESPONSES = [
    "I am the cloud OpenAI mock. Request routed to cloud successfully.",
    "Mock cloud response: Complex task handled by cloud backend.",
    "Fake OpenAI here — cloud routing is working.",
    "Cloud inference mock: API call processed.",
]

def _make_openai_response(model: str, content: str, req_id: str) -> dict:
    words = content.split()
    return {
        "id": req_id or f"chatcmpl-mock{random.randint(1000,9999)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     random.randint(10, 50),
            "completion_tokens": len(words),
            "total_tokens":      random.randint(60, 120),
        },
    }

@openai_app.post("/v1/chat/completions")
@openai_app.post("/chat/completions")   # also handle without /v1 prefix
async def openai_chat(request: Request):
    body     = await request.json()
    model    = body.get("model", "gpt-4o-mini")
    messages = body.get("messages", [])
    stream   = body.get("stream", False)
    user_msg = messages[-1].get("content", "") if messages else ""

    reply = random.choice(CLOUD_RESPONSES) + f" (echo: '{user_msg[:40]}')"
    req_id = f"chatcmpl-mock{random.randint(10000,99999)}"

    if stream:
        async def stream_gen():
            words = reply.split()
            # Opening chunk
            yield f"data: {json.dumps({'id':req_id,'object':'chat.completion.chunk','created':int(time.time()),'model':model,'choices':[{'index':0,'delta':{'role':'assistant','content':''},'finish_reason':None}]})}\n\n"
            for i, word in enumerate(words):
                chunk = {
                    "id": req_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.03)
            yield f"data: {json.dumps({'id':req_id,'object':'chat.completion.chunk','created':int(time.time()),'model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    return JSONResponse(_make_openai_response(model, reply, req_id))

# Anthropic mock
@openai_app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body    = await request.json()
    model   = body.get("model", "claude-3-5-haiku-20241022")
    msgs    = body.get("messages", [])
    user_msg = msgs[-1].get("content", "") if msgs else ""
    reply   = f"Mock Anthropic response for model {model}. (echo: '{user_msg[:40]}')"
    return JSONResponse({
        "id": f"msg_mock{random.randint(1000,9999)}",
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": reply}],
        "model": model, "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": len(reply.split())},
    })

@openai_app.get("/v1/models")
@openai_app.get("/models")
async def openai_models():
    return {"object": "list", "data": [
        {"id": "gpt-4o-mini", "object": "model"},
        {"id": "gpt-4o",      "object": "model"},
    ]}


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import threading

    def run_ollama():
        uvicorn.run(ollama_app, host="0.0.0.0", port=11435, log_level="warning")

    def run_openai():
        uvicorn.run(openai_app, host="0.0.0.0", port=11436, log_level="warning")

    t1 = threading.Thread(target=run_ollama, daemon=True)
    t2 = threading.Thread(target=run_openai, daemon=True)
    t1.start()
    t2.start()

    print("=" * 60)
    print("  Mock Backends running")
    print("  Fake Ollama  → http://localhost:11435")
    print("  Fake OpenAI  → http://localhost:11436/v1")
    print()
    print("  Now edit config.yaml:")
    print("    local.base_url:  http://localhost:11435")
    print("    cloud.base_url:  http://localhost:11436/v1")
    print()
    print("  Then start the router:  python main.py")
    print("  Then run tests:         bash tests/test_manual.sh")
    print("                          python tests/test_runner.py")
    print("=" * 60)
    print(f"  Started at {datetime.now().strftime('%H:%M:%S')}  |  Ctrl+C to stop")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMock backends stopped.")
