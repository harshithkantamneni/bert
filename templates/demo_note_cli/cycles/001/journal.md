# Cycle 1 journal — note-cli

**Date:** 2026-05-13
**Cycle target:** Ship `note capture` command end-to-end.

## What landed

- `code/note.py` — 75 lines, stdlib only. `capture()` + `main()` entrypoints.
- `tests/test_capture.py` — 5 tests, all green.
- Latency budget hit: capture runs in 1.8ms on cold disk (≪ 100ms target).

## What was harder than expected

The empty-text rejection looked trivial but caught a subtle bug: stripping
whitespace before the `if not text` check is critical, otherwise `"   "`
sneaks through as truthy. The test for that case caught it before the proof
packet was sealed.

## Open questions for cycle 2

- Should `note query --tag <x>` filter by tag? (likely yes — it's the
  natural next step)
- Markdown frontmatter format: YAML-style `---` block is verbose for
  CLI capture. Consider a tighter format that's still parseable.
- File rotation: one-file-per-day is fine for a day or two; need a
  monthly archive plan before it grows.

## Next-cycle hint

`note query --tag <x>` — implement tag-indexed retrieval. Tests must
verify: empty-archive case, single-tag match, multi-tag intersection,
case-insensitive matching.
