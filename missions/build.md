# Mission — Build: JSONL Histogram CLI Utility

Build a small Python CLI utility named `jhist` that produces a
histogram from any field in a JSONL file. The deliverable is working
code + tests, not prose.

## Functional requirements

- `jhist <file.jsonl> --field <key>` prints a text histogram of values
  found at that key.
- Supports nested keys via dotted paths: `--field user.profile.age`.
- Supports `--bins N` to bucket numeric values into N bins.
- Categorical fields (strings): show top-K by count, default K=20.
- Numeric fields: auto-detect, show bucketed histogram.
- Missing key (in a given row): count under `(missing)` bucket.
- Returns exit code 0 on success, 1 on file-not-found, 2 on
  unparseable JSONL line.

## Quality requirements

- Single file: `tools/jhist.py` (or `jhist/__init__.py` if it grows)
- Standard library only — no pandas / no plotting libs
- ≥3 unit tests under `tests/test_jhist.py` covering: categorical
  field, numeric field with bins, missing-key handling
- Tests run via `.venv/bin/python -m pytest tests/test_jhist.py -q`
- The cycle's finding file is the place to capture the design
  decisions and the verification command output, NOT the code itself
  (code lives in `tools/`)

## Procedure per cycle

- **Researcher**: design the CLI surface, edge cases, test plan;
  prototype the parser; write the first cuts of code + tests; capture
  the decisions in the finding file. Use memory_search to check what
  prior cycles already implemented.
- **Strategist**: review researcher's code; run the tests; identify
  what's broken or missing; specify the next concrete change.

## Constraints

- No external libraries.
- No `print()` for actual histogram output — use a function that
  returns a string so tests can assert on it. The CLI calls that
  function and prints.
- Code must actually run on this machine — the verification command
  invokes pytest and rejects on failure.

## Expected output

After 5 cycles, `tools/jhist.py` should be a working utility with
passing tests. The finding files chronicle the design progression.

## Non-goals

- GUI / plotting
- Streaming / online statistics
- Format support beyond JSONL
