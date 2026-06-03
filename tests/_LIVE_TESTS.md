# Live integration tests

Tests in this list require live external services (network, API keys,
real model providers). They're **not** in the canonical CI regression
because they fail without those services. Run them manually when you
want to verify the live integration paths.

## Tests requiring live model providers

These call real LLMs and need valid API keys in `credentials.json`:

- `_smoke_spawn.py` — runs a real subagent dispatch end-to-end; needs Cerebras key
- `_smoke_verification_command.py` — verifies the verification_command
  override path with a live Cerebras call

## How to run a live test

```bash
# Make sure credentials.json has the required keys
.venv/bin/python -u tests/_smoke_spawn.py
.venv/bin/python -u tests/_smoke_verification_command.py
```

## How to run the canonical regression (live tests excluded)

```bash
# The bash regression script auto-skips entries matching _LIVE_TESTS_PATTERN
ALL_TESTS=$(ls tests/_smoke_*.py)
LIVE=$(grep -lE '"""LIVE-TEST' tests/_smoke_*.py)
for t in $ALL_TESTS; do
  if echo "$LIVE" | grep -q "$t"; then continue; fi
  .venv/bin/python -u "$t"
done
```

Each live test has `"""LIVE-TEST` as its first line so the bash filter
can identify it without maintaining a separate list file.
