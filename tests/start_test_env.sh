#!/bin/bash
# start_test_env.sh
# ──────────────────────────────────────────────────────────────────────────────
# One command to spin up the full test environment:
#   1. Patches config.yaml to point at mock backends
#   2. Starts mock backends in background
#   3. Starts the AI Router
#
# Usage:
#   cd ai-router
#   bash tests/start_test_env.sh
#
# Stop:  Ctrl+C  (kills all background processes)
# ──────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
CONFIG="$ROOT/config.yaml"
BACKUP="$ROOT/config.yaml.bak"

cleanup() {
  echo ""
  echo "Stopping test environment..."
  # Restore original config
  if [ -f "$BACKUP" ]; then
    mv "$BACKUP" "$CONFIG"
    echo "config.yaml restored."
  fi
  kill %1 %2 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

# 1. Backup config and patch URLs to point at mock backends
cp "$CONFIG" "$BACKUP"
python3 - "$CONFIG" << 'PYEOF'
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

# Patch local provider (ollama) to use mock
for p in cfg['providers']['local']['list']:
    if p['name'] == 'ollama':
        p['base_url'] = 'http://localhost:11435'

# Patch cloud provider (openai) to use mock
for p in cfg['providers']['cloud']['list']:
    if p['name'] == 'openai':
        p['base_url'] = 'http://localhost:11436/v1'
    if p['name'] == 'anthropic':
        p['base_url'] = 'http://localhost:11436'

# Make sure OPENAI_API_KEY doesn't block startup (mock ignores it)
with open(sys.argv[1], 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print("config.yaml patched → mock backends")
PYEOF

# 2. Start mock backends
echo "Starting mock backends..."
python3 "$SCRIPT_DIR/mock_backends.py" &
MOCK_PID=$!
sleep 2

# 3. Create a minimal .env if missing
if [ ! -f "$ROOT/.env" ]; then
  echo "OPENAI_API_KEY=mock-key-not-used" > "$ROOT/.env"
  echo ".env created with mock key"
fi

# 4. Start the router
echo "Starting AI Router..."
cd "$ROOT"
python3 main.py &
ROUTER_PID=$!
sleep 2

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Test environment ready!"
echo "  Mock Ollama  → http://localhost:11435"
echo "  Mock OpenAI  → http://localhost:11436/v1"
echo "  AI Router    → http://localhost:8080"
echo "  Dashboard    → open tests/../dashboard.html"
echo ""
echo "  Run curl tests:    bash tests/test_manual.sh"
echo "  Run Python tests:  python3 tests/test_runner.py"
echo "══════════════════════════════════════════════════════"
echo "  Press Ctrl+C to stop everything"

wait
