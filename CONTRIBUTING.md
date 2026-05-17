# Contributing to Diksuchi

Thanks for taking the time. Diksuchi is a small, focused tool — contributions that keep it simple and reliable are most welcome.

## What we're looking for

- New routing rule types or conditions
- Additional provider integrations
- Bug fixes with a failing test that proves the fix
- Dashboard improvements
- Documentation improvements

## What to avoid

- Breaking the OpenAI-compatible response envelope
- Adding runtime dependencies without strong justification
- Changes that require cloud credentials to test

## Setup

```bash
git clone https://github.com/manikantaDasari/diksuchi
cd diksuchi
cp .env.example .env
pip install -r requirements.txt
make test   # all 35 should pass
```

## Making changes

1. Fork → feature branch off `main` (`git checkout -b feat/your-rule`)
2. Make your change
3. Add or update tests in `tests/test_runner.py`
4. Run `make test` — all 35 tests must pass
5. Open a PR against `main` using the PR template

## Routing rule changes

All routing rules live in `config.yaml` — not in `router_engine.py`. Only touch `router_engine.py` if you're adding a new **condition type** (e.g. a new `condition:` key). Document new condition types inline in `config.yaml`.

## Commit style

```
feat: add task_type routing condition
fix: regex rules now use re.MULTILINE
docs: update provider table in README
chore: bump python base image to 3.12.4
```

## Code of conduct

Be direct and constructive. No personal attacks. Disagreements about routing logic are fine — keep them technical.
