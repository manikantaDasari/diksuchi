# Cursor → Diksuchi Setup

Cursor (cursor.sh) is an AI-first code editor built on VS Code.
It supports custom OpenAI-compatible API endpoints.

## Steps

1. Open Cursor → **Settings** (Cmd+,)
2. Search for **"OpenAI API Key"** or go to:
   `Cursor Settings → Models → OpenAI API Key`
3. Fill in:

| Field        | Value                      |
|--------------|----------------------------|
| API Key      | `router` (any non-empty)   |
| Base URL     | `http://localhost:8080/v1` |
| Model        | `gpt-4o` or `llama3.2`    |

4. Make sure your router is running: `python main.py`

## Routing behaviour

| What you type in Cursor     | What happens                              |
|-----------------------------|-------------------------------------------|
| Short inline edit           | Router sees short prompt → **local**      |
| "Explain this entire file"  | Router sees long prompt → **cloud**       |
| Model = `gpt-4o`            | Model hint → forced to **cloud**          |
| Model = `llama3.2`          | Model hint → forced to **local**          |
| Business hours (IST)        | Time rule → **local**                     |
| Evening / weekend           | Time rule → **cloud**                     |

## Per-request override

Cursor doesn't expose custom headers yet, but you can pre-select
a model name to hint the router:

- `llama3.2` → always local (Ollama)
- `gpt-4o` or `gpt-4o-mini` → always cloud (OpenAI)
- `llama-3.1-8b-instant` → always cloud/Groq (if Groq is active)
