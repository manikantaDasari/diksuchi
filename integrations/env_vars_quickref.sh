#!/bin/bash
# env_vars_quickref.sh
# ──────────────────────────────────────────────────────────────────────────────
# Any tool or library that reads OPENAI_API_BASE / OPENAI_BASE_URL
# can be redirected to your AI Router with just environment variables.
# No code changes needed.
# ──────────────────────────────────────────────────────────────────────────────

export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_API_KEY="router"       # non-empty; router manages real keys

# Now run ANY OpenAI-SDK-based tool and it will go through your router:
#   aider --model gpt-4o
#   python my_app.py
#   uvicorn my_fastapi_app:app
#   jupyter notebook

# ── Per-tool env vars ─────────────────────────────────────────────────────────
# Some tools use their own variable names:

# LangChain
export OPENAI_API_BASE="http://localhost:8080/v1"   # older LangChain versions

# LlamaIndex
export OPENAI_API_BASE="http://localhost:8080/v1"

# Semantic Kernel (Python)
export AZURE_OPENAI_ENDPOINT="http://localhost:8080/v1"

# AutoGen
export OAI_CONFIG_LIST='[{"model":"llama3.2","api_key":"router","base_url":"http://localhost:8080/v1"}]'

echo "Environment configured. Router: $OPENAI_BASE_URL"
