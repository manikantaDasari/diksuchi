#!/bin/bash
# test_manual.sh
# ──────────────────────────────────────────────────────────────────────────────
# Manual curl tests for every AI Router endpoint.
# Run AFTER starting the router (and optionally mock backends).
#
# Usage:
#   bash tests/test_manual.sh            # run all tests
#   bash tests/test_manual.sh health     # run only the health test
#   bash tests/test_manual.sh preview    # run only routing previews
# ──────────────────────────────────────────────────────────────────────────────

ROUTER="http://localhost:8080"
PASS=0; FAIL=0
FILTER="${1:-all}"

# ── Helpers ───────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

section() { echo -e "\n${CYAN}━━━  $1  ━━━${NC}"; }
pass()    { echo -e "  ${GREEN}✓ PASS${NC}  $1"; ((PASS++)); }
fail()    { echo -e "  ${RED}✗ FAIL${NC}  $1"; ((FAIL++)); }
info()    { echo -e "  ${YELLOW}ℹ${NC}  $1"; }

run() {
  # run <test_name> <filter_group> <curl_args...>
  local name="$1"; local group="$2"; shift 2
  [[ "$FILTER" != "all" && "$FILTER" != "$group" ]] && return
  echo -e "\n  ${CYAN}▶ $name${NC}"
  local resp
  resp=$(curl -s -w "\n__STATUS__%{http_code}" "$@")
  local status=$(echo "$resp" | tail -1 | sed 's/__STATUS__//')
  local body=$(echo "$resp" | head -n -1)
  echo "  Status: $status"
  echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body" | head -c 600
  if [[ "$status" == "200" ]]; then pass "$name"; else fail "$name (HTTP $status)"; fi
}

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║   AI Router Manual Test Suite        ║"
echo "  ║   Target: $ROUTER   ║"
echo "  ╚══════════════════════════════════════╝${NC}"


# ════════════════════════════════════════════════════════════
section "1. Health & Info"
# ════════════════════════════════════════════════════════════

run "Health check" "health" \
  "$ROUTER/health"

run "Root info" "health" \
  "$ROUTER/"

run "Stats (fresh)" "health" \
  "$ROUTER/v1/stats"


# ════════════════════════════════════════════════════════════
section "2. Route Preview (dry-run — no LLM needed)"
# ════════════════════════════════════════════════════════════

run "Preview: short prompt → local" "preview" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}]}'

run "Preview: long prompt → cloud" "preview" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"'"$(python3 -c "print('word ' * 810)")"'"}]}'

run "Preview: gpt-4o model hint → cloud" "preview" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"model":"gpt-4o"}'

run "Preview: llama3.2 model hint → local" "preview" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"model":"llama3.2"}'

run "Preview: complexity keyword → cloud" "preview" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a comprehensive analysis of distributed systems"}]}'

run "Preview: simple keyword → local" "preview" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Quick question: what is 2+2?"}]}'

run "Preview: X-Route-To cloud header" "preview" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -H "X-Route-To: cloud" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'

run "Preview: X-Route-To local header" "preview" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -H "X-Route-To: local" \
  -d '{"messages":[{"role":"user","content":"Write a comprehensive analysis"}]}'


# ════════════════════════════════════════════════════════════
section "3. Chat Completions (needs backends running)"
# ════════════════════════════════════════════════════════════

run "Chat: auto-route (short → local)" "chat" \
  -X POST "$ROUTER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Say hello in one word."}]}'

run "Chat: force local via header" "chat" \
  -X POST "$ROUTER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Route-To: local" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"What is Python?"}]}'

run "Chat: force cloud via header" "chat" \
  -X POST "$ROUTER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Route-To: cloud" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What is Python?"}]}'

run "Chat: complexity keyword → cloud" "chat" \
  -X POST "$ROUTER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a comprehensive financial model"}]}'

run "Chat: system + user messages" "chat" \
  -X POST "$ROUTER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Say hi."}]}'

run "Chat: verify x_router metadata in response" "chat" \
  -X POST "$ROUTER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ping"}]}'


# ════════════════════════════════════════════════════════════
section "4. Streaming"
# ════════════════════════════════════════════════════════════

echo ""
echo -e "  ${CYAN}▶ Streaming: local (watch SSE chunks)${NC}"
[[ "$FILTER" == "all" || "$FILTER" == "stream" ]] && \
  curl -s -N -X POST "$ROUTER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-Route-To: local" \
    -d '{"model":"llama3.2","messages":[{"role":"user","content":"Count 1 to 5"}],"stream":true}' \
  | head -20
echo ""

echo -e "  ${CYAN}▶ Streaming: cloud (watch SSE chunks)${NC}"
[[ "$FILTER" == "all" || "$FILTER" == "stream" ]] && \
  curl -s -N -X POST "$ROUTER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-Route-To: cloud" \
    -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Count 1 to 5"}],"stream":true}' \
  | head -20
echo ""


# ════════════════════════════════════════════════════════════
section "5. Provider Management"
# ════════════════════════════════════════════════════════════

run "List all providers" "providers" \
  "$ROUTER/v1/providers"

run "Activate cloud provider: openai" "providers" \
  -X POST "$ROUTER/v1/providers/cloud/activate/openai"

run "Activate cloud provider: groq" "providers" \
  -X POST "$ROUTER/v1/providers/cloud/activate/groq"

run "Verify active cloud changed to groq" "providers" \
  "$ROUTER/v1/stats"

run "Restore cloud provider: openai" "providers" \
  -X POST "$ROUTER/v1/providers/cloud/activate/openai"

run "Activate unknown provider (expect 404)" "providers" \
  -X POST "$ROUTER/v1/providers/cloud/activate/does_not_exist"


# ════════════════════════════════════════════════════════════
section "6. Stats & Log"
# ════════════════════════════════════════════════════════════

run "Stats after tests" "stats" \
  "$ROUTER/v1/stats"

run "Request log (last 10)" "stats" \
  "$ROUTER/v1/log?limit=10"

run "Config (sanitised)" "stats" \
  "$ROUTER/v1/config"


# ════════════════════════════════════════════════════════════
section "7. Edge Cases"
# ════════════════════════════════════════════════════════════

run "Empty messages array" "edge" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -d '{"messages":[],"model":""}'

run "Invalid JSON body (expect 400)" "edge" \
  -X POST "$ROUTER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d 'NOT_JSON'

run "Very long model name" "edge" \
  -X POST "$ROUTER/v1/route/preview" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o-2024-11-20-preview-turbo"}'


# ════════════════════════════════════════════════════════════
echo -e "\n${CYAN}━━━  Results  ━━━${NC}"
echo -e "  ${GREEN}Passed: $PASS${NC}   ${RED}Failed: $FAIL${NC}"
TOTAL=$((PASS + FAIL))
echo -e "  Total:  $TOTAL"
if [ $FAIL -eq 0 ]; then
  echo -e "\n  ${GREEN}All tests passed! ✓${NC}\n"
else
  echo -e "\n  ${RED}$FAIL test(s) failed.${NC}\n"
  exit 1
fi
