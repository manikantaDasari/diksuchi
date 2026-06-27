# ─────────────────────────────────────────────────────────────────────────────
#  Diksuchi — Local-First AI Router — Production Dockerfile
#
#  Multi-stage build:
#    Stage 1 (builder) — install Python deps into an isolated venv
#    Stage 2 (runtime) — copy only the venv + source, run as non-root user
#
#  Build:  docker build -t diksuchi .
#  Run:    docker run -d -p 8081:8081 -e OPENAI_API_KEY=sk-xxx diksuchi
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build-time tools (gcc needed for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv so the runtime stage only copies what it needs
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first — this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: minimal runtime image ───────────────────────────────────────────
FROM python:3.12-slim AS runtime

# curl is needed for the HEALTHCHECK command
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — principle of least privilege
RUN useradd --no-create-home --shell /bin/false --system router

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source (no tests, no integrations, no .env)
COPY main.py router_engine.py config.yaml ./

# .env is bind-mounted or passed via -e at runtime — never bake secrets
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Drop to non-root
USER router

EXPOSE 8080

# Container-level health check (also used by docker-compose depends_on)
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python", "main.py"]
