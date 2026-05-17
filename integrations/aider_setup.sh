#!/bin/bash
# aider_setup.sh
# ──────────────────────────────────────────────────────────────────────────────
# Aider (https://github.com/paul-gauthier/aider) is a CLI AI coding agent.
# Point it at your AI Router so all requests get local-first routing.
#
# Install:  pip install aider-chat
# Usage:    bash aider_setup.sh
# ──────────────────────────────────────────────────────────────────────────────

ROUTER="http://localhost:8080/v1"

echo "Starting Aider → AI Router ($ROUTER)"
echo ""

# ── Option 1: Auto-mode (let router decide) ───────────────────────────────────
# Aider uses the model name you give it. If the router rule for "llama3.2"
# matches local, it goes local; otherwise cloud.
aider \
  --openai-api-base "$ROUTER" \
  --openai-api-key  "router" \
  --model           "openai/llama3.2" \
  "$@"

# ── Option 2: Force cloud (copy-paste to use instead of Option 1) ─────────────
# aider \
#   --openai-api-base "$ROUTER" \
#   --openai-api-key  "router" \
#   --model           "openai/gpt-4o" \
#   --header          "X-Route-To: cloud" \
#   "$@"

# ── Option 3: Always use Groq for maximum speed ───────────────────────────────
# aider \
#   --openai-api-base "$ROUTER" \
#   --openai-api-key  "router" \
#   --model           "openai/llama-3.1-8b-instant" \
#   --header          "X-Route-To: cloud" \
#   "$@"
