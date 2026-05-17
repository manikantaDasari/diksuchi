# ─────────────────────────────────────────────────────────────────────────────
#  AI Router — Makefile
#
#  Common tasks:
#    make up        — build & start the full stack (Ollama + router)
#    make down      — stop and remove containers
#    make restart   — restart the router only (after config change)
#    make logs      — tail router logs
#    make pull      — pull/refresh the default Ollama model
#    make build     — rebuild the router Docker image
#    make push      — build multi-arch image and push to GHCR
#    make test      — run the test suite (requires local Python venv)
#    make clean     — remove stopped containers and dangling images
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_NAME   ?= ai-router
IMAGE_TAG    ?= latest
GHCR_USER    ?= manikantaDasari
PLATFORMS    ?= linux/amd64,linux/arm64
OLLAMA_MODEL ?= llama3.2

# Full GHCR image reference
GHCR_IMAGE   := ghcr.io/$(GHCR_USER)/$(IMAGE_NAME):$(IMAGE_TAG)

.DEFAULT_GOAL := help

# ── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  AI Router — available targets"
	@echo ""
	@echo "  make up          Start full stack (Ollama + router)"
	@echo "  make down        Stop and remove containers"
	@echo "  make restart     Restart router only"
	@echo "  make logs        Tail router logs"
	@echo "  make pull        Pull / refresh the Ollama model"
	@echo "  make build       Build router Docker image locally"
	@echo "  make push        Build multi-arch + push to GHCR"
	@echo "  make test        Run test suite (local Python venv)"
	@echo "  make clean       Remove stopped containers & dangling images"
	@echo "  make gpu         Start full stack with NVIDIA GPU support"
	@echo ""

# ── Stack lifecycle ───────────────────────────────────────────────────────────
.PHONY: up
up:
	@[ -f .env ] || (echo "⚠  .env not found — copying .env.example"; cp .env.example .env)
	docker compose up -d --build
	@echo "✓ Router running at http://localhost:$${ROUTER_PORT:-8080}"

.PHONY: down
down:
	docker compose down

.PHONY: restart
restart:
	docker compose restart router

.PHONY: gpu
gpu:
	@[ -f .env ] || (echo "⚠  .env not found — copying .env.example"; cp .env.example .env)
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# ── Logs ─────────────────────────────────────────────────────────────────────
.PHONY: logs
logs:
	docker compose logs -f router

.PHONY: logs-all
logs-all:
	docker compose logs -f

# ── Model management ─────────────────────────────────────────────────────────
.PHONY: pull
pull:
	docker compose exec ollama ollama pull $(OLLAMA_MODEL)

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
