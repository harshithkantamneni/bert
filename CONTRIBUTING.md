# Contributing to bert

Thanks for your interest. bert is a long-context retrieval MCP server; the scope is deliberately narrow (see the README's "What bert is / is NOT" and `benchmarks/BENCHMARK_SYNTHESIS.md`). Contributions that sharpen the retrieval/memory core, the benchmark rigor, or the MCP surface are most welcome; please keep the honest-positioning discipline (claims must trace to a benchmark result).

## Setup

```bash
git clone <your-fork> bert && cd bert
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # full stack (incl. the ML retrieval deps)
```

Provider keys (any subset) go in your own `~/.bert-lab/credentials.json` (mode 600) or the environment — never commit keys; `.gitignore` already excludes `credentials.json` and `.env`.

## Tests

```bash
# Fast, offline, no keys (what CI runs) — benchmark methodology + metric logic:
PYTHONPATH=. pytest tests/benchmarks/ -q

# Full suite (needs the ML extras; real-API tests skip without keys):
PYTHONPATH=. pytest tests/ -q
```

TDD is expected for new behavior: write the failing test first, watch it fail, then the minimal implementation. Lint with `ruff check`.

## Ground rules

- **Honesty over hype.** Every performance claim in docs/README must trace to a result in `benchmarks/`. Nulls are reported as headlines, not buried.
- **No secrets, ever.** Keys live in the user's local credentials file; the security test (`tests/_smoke_gg_a_0_credentials.py`) guards against committed key prefixes.
- **Keep the proof path intact.** Changes touching `core/proof_packet.py` must keep a `packet_export` → `bert verify` round-trip passing.

## License

By contributing you agree your contributions are licensed under the project's MIT license.
