# Design: `memory_ingest` MCP tool + auto-resync

Status: approved 2026-06-05

## Goal

Let an AI coding host (Claude Code, Cursor, Codex) point bert at an **existing
external project** and have bert build and maintain the retrieval index over it,
entirely from inside the MCP session. Today the ingest primitive
(`memory.ingest_corpus`) exists but is only reachable in Python / the benchmark
harness; there is no MCP tool, and the index does not refresh as the source
project changes.

## Non-goals

- Connecting to a live external **database** (Postgres/MySQL/etc.). bert's store
  stays local sqlite + sqlite-vec; "ingest" means files/text only.
- A long-running background file-watcher daemon. Considered and rejected for this
  use case (see Approach). Freshness is delivered lazily at query time instead.
- In-place indexing of the source tree. Files are sharded into the lab corpus
  (the existing, path-safe mechanism); the source tree is never modified.

## Background (current state, verified)

- `memory.ingest_corpus(src_dir, *, dest="findings/corpus", exts=(...),
  eager_index=False)` walks an external tree and writes each file as a `.md`
  shard under the active lab's `findings/corpus/<relpath>.md` via `create()`,
  which enforces a path-traversal gate (writes only under `memories/` or
  `findings/`). It currently re-shards **every** file on each call (not
  incremental) and does not remove shards for deleted source files.
- `_index_corpus()` lazily re-embeds `.md` shards whose mtime exceeds the stored
  `indexed_mtime`, GCs chunks for vanished shards, and is TTL-cached (5s) and
  mtime-driven. This is the proven pattern we extend.
- MCP tools are registered with `srv.register_tool(name, description,
  input_schema, handler)`; handlers are `def _t_*(args: dict) -> dict`. Labs are
  resolved with `_resolve_lab()` and scoped with
  `lab_context.set_active_lab_path()`.

## Approach

A background watcher only lives while the MCP process is up, needs
re-shard-on-change + debounce + thread lifecycle, can thrash on large repos, and
is the hardest piece to test, all for the marginal benefit of staying fresh when
nobody is querying. Freshness only matters at query time. So we instead make
ingest **incremental** and **auto-resync registered sources at search time**
(TTL-gated), reusing bert's existing lazy-reindex philosophy. Automatic like a
watcher, but deterministic and testable, with no process to leak.

## Components

**A. `ingest_corpus` becomes incremental + GC** (`core/memory.py`)
- Skip a source file when its shard exists and `source.mtime <= shard.mtime`.
- Re-shard only changed source files (a fresh `create()` bumps shard mtime, so
  `_index_corpus` re-embeds it).
- GC: delete shards under `dest/` whose originating source file no longer exists.
- Return type stays `int` (= files written this call) for backward compatibility
  with the benchmark callers (`b9_rag_runner`, `run_b9_wall`, `run_b10_niah`).

**B. Source registry** (`core/memory.py`)
- `register_ingest_source(source, *, exts, dest)` appends/updates an entry in
  `<lab>/state/ingest_sources.json`: `[{ "source": abs, "exts": [...],
  "dest": "findings/corpus" }]` (dedup by resolved source path).
- `list_ingest_sources()` reads it back. Scoped to the active lab.

**C. `resync_sources()`** (`core/memory.py`)
- For each registered root, run the incremental `ingest_corpus` (+ index).
- TTL-gated (reuse the existing 5s cache, keyed per lab) so frequent searches do
  not re-walk the external tree more than once per window.
- Returns `{sources, files_written}`.

**D. `memory_ingest` MCP tool** (`tools/mcp/bert_lab.py`)
- Input: `lab` (required), `source` (required, external dir), optional `exts`
  (string list).
- Handler `_t_memory_ingest`: resolve lab (404 if missing) → validate `source`
  (exists, is a directory) → enforce safety caps/exclusions → set active lab →
  `ingest_corpus(source, eager_index=True)` → `register_ingest_source(...)` →
  return `{ok, lab, source, files_ingested, files_skipped, files_removed,
  chunks_indexed, db_bytes, truncated?}`.

**E. Auto-resync in `_t_memory_search`** (`tools/mcp/bert_lab.py`)
- Before searching (grep *or* vector), call `resync_sources()` under the active
  lab (TTL-gated). Results then reflect the current project with no agent effort.
- Kept in the MCP layer so `core/memory.search()` stays decoupled.

**F. Safety** (public tool taking a filesystem path)
- Source must be an existing directory.
- Caps (defaults): `BERT_INGEST_MAX_FILES=5000`, `BERT_INGEST_MAX_BYTES=100MB`.
  On hitting a cap, ingest what fits and return `truncated: true` with counts
  (no silent truncation).
- Exclude directories: `.git`, `node_modules`, `.venv`, `__pycache__`, `dist`,
  `build`, `target`, `.next`, and any dotdir.
- Do not follow symlinks that resolve outside the source root.
- Shard writes continue through the path-safe `create()` gate.

## Error handling

- Missing/invalid lab → `{ok: false, error: "lab not found: ..."}`.
- Source missing / not a directory → `{ok: false, error: ...}`.
- Per-file read errors are skipped (counted), never fatal.
- Registry/resync failures are caught and logged; a resync failure must never
  break `memory_search` (degrade to searching the existing index).

## Testing (TDD, written first)

Unit (`tests/`):
1. incremental ingest: unchanged source file is skipped on re-call.
2. changed source file is re-sharded and re-embedded.
3. deleted source file → its shard is GC'd (and chunks drop).
4. registry round-trip: `register_ingest_source` then `list_ingest_sources`.
5. `resync_sources` picks up an out-of-band source change.
6. caps + exclusions: `.git`/`node_modules` skipped; cap → `truncated`.
7. MCP tool happy path; bad-source and bad-lab errors.

Integration: full `tools/eval/industry_eval.sh` / smoke suite green; CI green.

## Future work (explicitly out of scope now)

- Opt-in CLI watcher (`bert memory ingest --watch`) for very long sessions.
- URL / git-remote sources.
- Chunk-granular (not file-granular) incremental re-embed.
