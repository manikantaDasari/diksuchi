<div align="center">

<!-- Logo -->
<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 72 72">
  <rect width="72" height="72" rx="16" fill="#E1F64A"/>
  <line x1="36" y1="14" x2="36" y2="58" stroke="#111110" stroke-width="2" stroke-linecap="round" opacity="0.25"/>
  <line x1="14" y1="36" x2="58" y2="36" stroke="#111110" stroke-width="2" stroke-linecap="round" opacity="0.25"/>
  <polygon points="36,14 31,40 36,36 41,40" fill="#111110"/>
  <polygon points="36,58 31,34 36,38 41,34" fill="#111110" opacity="0.3"/>
</svg>

# Diksuchi

**Always pointing to the right model.**

[![Docker Pulls](https://img.shields.io/docker/pulls/ghcr.io/manikantaDasari/diksuchi?style=flat-square&color=E1F64A&labelColor=1A1A1A)](https://github.com/manikantaDasari/diksuchi/pkgs/container/diksuchi)
[![MIT License](https://img.shields.io/badge/license-MIT-E1F64A?style=flat-square&labelColor=1A1A1A)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-E1F64A?style=flat-square&labelColor=1A1A1A)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-35%2F35-13BD7C?style=flat-square&labelColor=1A1A1A)](tests/)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20compatible-white?style=flat-square&labelColor=1A1A1A)](https://platform.openai.com/docs/api-reference)

</div>

---

**Diksuchi** (Sanskrit: दिक्सूची — *direction needle*) is a local-first AI API router. Drop it between your code and your AI providers — it automatically picks the right model for each request based on complexity, task type, time of day, token count, and 13 other rules. Zero code changes required in your app.

```
Your App  →  Diksuchi  →  Local (Ollama / llama3.2)   [fast, free, private]
                       →  Cloud (GPT-4o / Claude / Gemini)  [powerful, when needed]
```

---

## Why Diksuchi?

| Without Diksuchi | With Diksuchi |
|---|---|
| Hard-code a single model everywhere | Each request gets the best model automatically |
| Pay cloud rates for simple chats | 60–80% of simple requests stay local |
| Switch models = rewrite API calls | Zero code changes — one endpoint |
| No insight into routing decisions | Live dashboard with full audit trail |

---

## Quick Start

### Option A — Docker (recommended, one command)

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  ghcr.io/manikantaDasari/diksuchi:latest
```

Your router is live at `http://localhost:8080/v1/chat/completions` — fully OpenAI-compatible.

### Option B — With local Ollama (full local+cloud stack)

```bash
# 1. Clone
git clone https://github.com/manikantaDasari/diksuchi
cd diksuchi/ai-router

# 2. Configure
cp .env.example .env
# Edit .env and add your API keys

# 3. Launch
make up
```

That's it. Ollama pulls automatically. Point your app at `http://localhost:8080`.

### Option C — Python directly

```bash
pip install -r requirements.txt
cp .env.example .env  # add your keys
python main.py
```

---

## Point your app at Diksuchi

Replace your existing model endpoint with Diksuchi's URL. No other changes:

**Python / OpenAI SDK**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="any-string"   # Diksuchi handles real auth
)

response = client.chat.completions.create(
    model="auto",          # Diksuchi picks the model
    messages=[{"role": "user", "content": "Explain binary search"}]
)
```

**curl**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Routing Rules

Diksuchi evaluates 13 rules in priority order:

| Priority | Rule | Routes to |
|---|---|---|
| 1 | `X-Router-Override: local` / `cloud` header | Explicit override |
| 2 | Model name hint (`gpt-*`, `claude-*`, `llama*`) | Matched provider |
| 3 | Stack trace detected in prompt (regex) | Cloud (GPT-4o) |
| 4 | Credentials / secrets detected | Local (private) |
| 5 | Git diff in prompt | Cloud (Claude) |
| 6 | High-complexity keywords | Cloud |
| 7 | Simple/chat keywords | Local |
| 8 | Night hours IST (10pm – 7am) | Local (save cost) |
| 9 | Prompt > 2000 tokens | Cloud |
| 10–13 | Fallback chain | Configurable default |

All rules, providers, and thresholds are defined in `config.yaml` — no code changes needed to tune routing behavior.

---

## Configuration

```yaml
# config.yaml

providers:
  local:
    name: Ollama
    base_url: http://ollama:11434/v1
    default_model: llama3.2
    api_key: ""

  cloud:
    name: OpenAI
    base_url: https://api.openai.com/v1
    default_model: gpt-4o
    api_key: ${OPENAI_API_KEY}

routing:
  default_provider: local

  rules:
    - name: header_override
      condition: header
      header_name: X-Router-Override
      priority: 100

    - name: complexity_high
      condition: keyword
      keywords: ["analyze", "debug", "architecture", "optimize", "refactor"]
      match: any
      route_to: cloud
      priority: 50

    - name: time_of_day
      condition: time_range
      start_hour_ist: 22
      end_hour_ist: 7
      route_to: local
      priority: 20
```

Full schema with all 13 rule types documented in [`config.yaml`](config.yaml).

---

## Dashboard

Open `http://localhost:8080` after starting Diksuchi.

- **Live routing feed** — see every request and where it was routed in real time  
- **Cost saved** — running total of estimated savings from local routing  
- **Provider split** — local vs. cloud percentage over time  
- **Routing preview** — test any prompt and see which rule fires  

---

## Continue.dev Integration (VS Code)

Route your in-editor AI assistant through Diksuchi:

```yaml
# ~/.continue/config.yaml
models:
  - name: Diksuchi Router
    provider: openai
    model: auto
    apiBase: http://localhost:8080/v1
    apiKey: local
```

All VS Code AI completions now flow through your routing rules — complex code questions go cloud, quick autocomplete stays local.

Full config in [`integrations/continue_dev_config.yaml`](integrations/continue_dev_config.yaml).

---

## Docker Reference

```bash
# Start (Ollama + Router)
make up

# GPU acceleration (NVIDIA)
make gpu

# View router logs
make logs

# Stop everything
make down

# Run tests
make test

# Build & push to GHCR
make push
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GROQ_API_KEY` | — | Groq API key |
| `OLLAMA_MODEL` | `llama3.2` | Local model to pull and use |
| `ROUTER_PORT` | `8080` | Port to expose |

Copy `.env.example` → `.env` and fill in keys.

---

## Supported Providers

| Provider | Type | Notes |
|---|---|---|
| **Ollama** | 🟡 Local | Any model: llama3.2, mistral, phi3, gemma2 |
| **OpenAI** | ☁️ Cloud | GPT-4o, GPT-4o-mini, o1, o3-mini |
| **Anthropic** | ☁️ Cloud | Claude Opus, Sonnet, Haiku |
| **Google Gemini** | ☁️ Cloud | Gemini 1.5 Pro, Flash |
| **Groq** | ☁️ Cloud | Llama3.3-70B at 500+ t/s |
| **Mistral** | ☁️ Cloud | Mistral Large, Nemo |
| **Cohere** | ☁️ Cloud | Command R+ |
| **Together AI** | ☁️ Cloud | Open model hosting |
| **OpenRouter** | ☁️ Cloud | 200+ model gateway |
| **LM Studio** | 🟡 Local | Any GGUF model |

---

## Project Structure

```
ai-router/
├── main.py                 # FastAPI async proxy (OpenAI-compatible)
├── router_engine.py        # Rule evaluation engine
├── config.yaml             # All providers + routing rules
├── dashboard.html          # Monitoring dashboard
├── requirements.txt
├── Dockerfile              # Multi-stage production image
├── docker-compose.yml      # Ollama + Router stack
├── docker-compose.gpu.yml  # NVIDIA GPU override
├── .env.example
├── Makefile
├── tests/                  # 35 unit + integration tests
│   └── ...
└── integrations/
    └── continue_dev_config.yaml
```

---

## Contributing

1. Fork → feature branch → PR against `main`
2. Run `make test` before pushing — all 35 tests must pass
3. Routing rule changes go in `config.yaml`, not `router_engine.py` (unless adding a new condition type)
4. Keep the OpenAI-compatible response envelope — downstream apps depend on it

Bug reports and rule suggestions welcome via Issues.

---

## License

MIT — use it, fork it, ship it.

---

<div align="center">
  <sub>Built with FastAPI · Runs on Ollama · Ships on Docker</sub><br/>
  <sub><b>Diksuchi</b> — दिक्सूची — Always pointing to the right model.</sub>
</div>
