# Cycle 1 plan — note capture command

## Goal

Implement `note capture <text> [--tag T]*` writing to `~/.notes/YYYY-MM-DD.md`.

## Steps

1. **Scaffold `note.py`** — argparse + main entry. Single file, stdlib only.
2. **Implement capture** — open today's file in append mode, write frontmatter+text.
3. **Write tests** — `test_capture.py` covers: latency, file format, tag extraction, empty-text rejection.

## Acceptance criteria

- `note "hello"` runs in <100ms (measured)
- `note "thought" --tag deep --tag idea` writes frontmatter with both tags
- `note ""` fails with non-zero exit + clear error
- All 5 tests pass via `python -m pytest tests/`
