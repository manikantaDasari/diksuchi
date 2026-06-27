# ─────────────────────────────────────────────────────────────────────────────
#  Diksuchi (दिक्सूची) — Local-First AI Router — Makefile
#
#  Quick start (no Docker needed):
#    make dev         — installs everything + starts the server (one command)
#
#  Intelligence:
#    make ratings     — fetch model ratings from HF Leaderboard (user-triggered)
#    make benchmark   — run self-benchmark against active providers
#
#  Docker (full stack):
#    make up          — build & start the full stack (Ollama + router)
#    make down        — stop and remove containers
#    make restart     — restart the router only (after config change)
#    make logs        — tail router logs
#    make build       — rebuild the router Docker image
#    make push        — build multi-arch image and push to GHCR
#    make test        — run the test suite (requires local Python venv)
#    make clean       — remove stopped containers and dangling images
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_NAME    ?= diksuchi
IMAGE_TAG     ?= latest
GHCR_USER     ?= manikantaDasari
PLATFORMS     ?= linux/amd64,linux/arm64

# Models to pull automatically on `make up` and `make models`
# gemma3:1b  → ~815 MB, CPU-only, runs on old/low-RAM machines  ← default
# To pull more:  make models OLLAMA_MODELS="gemma3:1b llama3.2"
OLLAMA_MODELS ?= gemma3:1b

# Full GHCR image reference
GHCR_IMAGE    := ghcr.io/$(GHCR_USER)/$(IMAGE_NAME):$(IMAGE_TAG)

.DEFAULT_GOAL := help

# ── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Diksuchi — available targets"
	@echo ""
	@echo "  ── Quick start (no Docker) ─────────────────────────────"
	@echo "  make dev         Install everything + start the server"
	@echo ""
	@echo "  ── Model intelligence ──────────────────────────────────"
	@echo "  make ratings     Fetch model ratings from HF Leaderboard"
	@echo "  make baseline    Rebuild bundled baseline from raw benchmark data"
	@echo "  make benchmark   Self-benchmark active providers (local judge)"
	@echo ""
	@echo "  ── Docker (full stack) ─────────────────────────────────"
	@echo "  make up          Start stack + pull all models automatically"
	@echo "  make models      Pull / refresh all OLLAMA_MODELS"
	@echo "  make down        Stop and remove containers"
	@echo "  make restart     Restart router only"
	@echo "  make logs        Tail router logs"
	@echo "  make build       Build router Docker image locally"
	@echo "  make push        Build multi-arch + push to GHCR"
	@echo "  make test        Run test suite (local Python venv)"
	@echo "  make clean       Remove stopped containers & dangling images"
	@echo "  make gpu         Start full stack with NVIDIA GPU support"
	@echo ""

# ── Local dev (no Docker) ────────────────────────────────────────────────────
.PHONY: install
install:
	@[ -d .venv ] || python3 -m venv .venv
	@.venv/bin/pip install -q --upgrade pip
	@.venv/bin/pip install -q -r requirements.txt
	@echo "✓ Dependencies installed"

.PHONY: dev
dev:
	@echo ""
	@echo "  ╔══════════════════════════════════════════╗"
	@echo "  ║        Diksuchi — starting up            ║"
	@echo "  ╚══════════════════════════════════════════╝"
	@echo ""
	@# ── Step 1: .env ──────────────────────────────────────────────────────────
	@if [ ! -f .env ]; then \
	  echo "  ⚠  No .env found — copying .env.example"; \
	  cp .env.example .env; \
	  echo "  ✎  Add your OPENAI_API_KEY to .env if you want cloud routing"; \
	  echo ""; \
	fi
	@# ── Step 2: Python venv ───────────────────────────────────────────────────
	@if [ ! -d .venv ]; then \
	  echo "  ▶  Creating virtual environment…"; \
	  python3 -m venv .venv; \
	fi
	@# ── Step 3: Dependencies ──────────────────────────────────────────────────
	@echo "  ▶  Checking dependencies…"
	@.venv/bin/pip install -q --upgrade pip
	@.venv/bin/pip install -q -r requirements.txt
	@echo "  ✓  All dependencies ready"
	@echo ""
	@echo "  ▶  Router → http://localhost:8081   (Ctrl+C to stop)"
	@echo ""
	@# ── Step 4: Launch ────────────────────────────────────────────────────────
	@.venv/bin/python main.py

# ── Model intelligence ────────────────────────────────────────────────────────
.PHONY: ratings
ratings:
	@echo ""
	@echo "  📡  Fetching model ratings from HF Open LLM Leaderboard…"
	@echo "  (This calls the public HF dataset API — no auth required)"
	@echo ""
	@[ -d .venv ] || python3 -m venv .venv
	@.venv/bin/pip install -q --upgrade pip
	@.venv/bin/pip install -q -r requirements.txt
	@.venv/bin/python rating_fetcher.py
	@echo ""
	@echo "  ✓  Ratings saved to model_ratings.json"
	@echo "  ✓  Restart the router to use updated ratings"
	@echo ""

.PHONY: baseline
baseline:
	@echo ""
	@echo "  🏗️  Rebuilding bundled baseline from model_benchmarks_raw.json…"
	@echo "  Source: model_benchmarks_raw.json (edit this file to update scores)"
	@echo ""
	@[ -d .venv ] || python3 -m venv .venv
	@.venv/bin/pip install -q --upgrade pip
	@.venv/bin/python baseline_builder.py
	@echo ""
	@echo "  ✓  model_ratings_baseline.json regenerated"
	@echo "  ✓  Run 'make ratings' or restart the router to use updated baseline"
	@echo ""

.PHONY: baseline-verify
baseline-verify:
	@[ -d .venv ] || python3 -m venv .venv
	@.venv/bin/python baseline_builder.py --verify

.PHONY: benchmark
benchmark:
	@echo ""
	@echo "  🔬  Running self-benchmark against active providers…"
	@echo "  (Sends 10 prompts to each provider, scored by local judge)"
	@echo ""
	@[ -d .venv ] || python3 -m venv .venv
	@.venv/bin/pip install -q --upgrade pip
	@.venv/bin/pip install -q -r requirements.txt
	@.venv/bin/python self_benchmark.py
	@echo ""
	@echo "  ✓  Results saved to self_benchmark_results.json"
	@echo "  ✓  Run 'make ratings' to merge with other sources"
	@echo ""

# ── Stack lifecycle ───────────────────────────────────────────────────────────
.PHONY: up
up:
	@[ -f .env ] || (echo "⚠  .env not found — copying .env.example"; cp .env.example .env)
	@echo ""
	@echo "  ╔══════════════════════════════════════════╗"
	@echo "  ║        Diksuchi — starting up            ║"
	@echo "  ╚══════════════════════════════════════════╝"
	@echo ""
	@echo "  ▶  Starting containers…"
	@docker compose up -d --build
	@echo ""
	@echo "  ▶  Waiting for Ollama to be ready…"
	@until docker compose exec -T ollama ollama list >/dev/null 2>&1; do \
	  printf "."; sleep 3; \
	done
	@echo " ✓"
	@echo ""
	@echo "  ▶  Pulling models: $(OLLAMA_MODELS)"
	@for model in $(OLLAMA_MODELS); do \
	  echo ""; \
	  echo "  ── $$model ──────────────────────────────────"; \
	  docker compose exec -T ollama ollama pull $$model; \
	done
	@echo ""
	@echo "  ✓  All models ready"
	@echo "  ✓  Router → http://localhost:$${ROUTER_PORT:-8080}"
	@echo ""

.PHONY: down
down:
	docker compose down

.PHONY: restart
restart:
	docker compose restart router

.PHONY: gpu
gpu:
	@[ -f .env ] || (echo "⚠  .env not found — copying .env.example"; cp .env.example .env)
	@docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
	@echo "  ▶  Waiting for Ollama (GPU)…"
	@until docker compose exec -T ollama ollama list >/dev/null 2>&1; do printf "."; sleep 3; done
	@echo " ✓"
	@$(MAKE) models

# ── Logs ─────────────────────────────────────────────────────────────────────
.PHONY: logs
logs:
	docker compose logs -f router

.PHONY: logs-all
logs-all:
	docker compose logs -f

# ── Model management ─────────────────────────────────────────────────────────
.PHONY: models
models:
	@echo "  ▶  Pulling models: $(OLLAMA_MODELS)"
	@for model in $(OLLAMA_MODELS); do \
	  echo ""; \
	  echo "  ── $$model ──────────────────────────────────"; \
	  docker compose exec -T ollama ollama pull $$model; \
	done
	@echo ""
	@echo "  ✓  Done"

.PHONY: model-list
model-list:
	@docker compose exec -T ollama ollama list

# ── Image build ───────────────────────────────────────────────────────────────
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# ── Multi-arch push to GHCR ──────────────────────────────────────────────────
.PHONY: push
push:
	docker buildx build \
	  --platform $(PLATFORMS) \
	  --tag $(GHCR_IMAGE) \
	  --push \
	  .
	@echo "✓ Pushed $(GHCR_IMAGE)"

# ── Tests ────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	@[ -d .venv ] || python3 -m venv .venv
	.venv/bin/pip install -q -r requirements.txt pytest httpx
	.venv/bin/pytest tests/ -v

# ── Cleanup ──────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	docker compose down --remove-orphans
	docker image prune -f
	@echo "✓ Cleaned up"
