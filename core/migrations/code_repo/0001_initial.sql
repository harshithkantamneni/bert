-- CodeRepoAdapter baseline schema (v1) — Phase D1
--
-- Stores parsed code: files + symbols + references + tests + git history.
-- Designed to be the structural backbone for code labs (refactor, audit,
-- migration, performance). Embeddings go on docstrings ONLY (not code
-- bodies) — structural lookups (callers, callees, tests covering) beat
-- semantic search for code work.

CREATE TABLE IF NOT EXISTS files (
  id            INTEGER PRIMARY KEY,
  path          TEXT UNIQUE NOT NULL,
  language      TEXT,                 -- 'python' | 'typescript' | 'go' | ...
  mtime         INTEGER,
  content_hash  TEXT,
  loc           INTEGER               -- non-blank line count
);

CREATE TABLE IF NOT EXISTS symbols (
  id              INTEGER PRIMARY KEY,
  file_id         INTEGER REFERENCES files(id) ON DELETE CASCADE,
  kind            TEXT,                -- 'function' | 'class' | 'method' | 'const' | 'type'
  name            TEXT NOT NULL,
  qualified_name  TEXT,                -- 'module.Class.method'
  signature       TEXT,
  start_byte      INTEGER,
  end_byte        INTEGER,
  start_line      INTEGER,
  end_line        INTEGER,
  docstring       TEXT
);

CREATE TABLE IF NOT EXISTS symbol_refs (
  caller_id   INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  callee_id   INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  ref_kind    TEXT,                    -- 'call' | 'import' | 'inherit' | 'type_ref'
  file_id     INTEGER REFERENCES files(id),
  line        INTEGER,
  PRIMARY KEY (caller_id, callee_id, ref_kind, line)
);

CREATE TABLE IF NOT EXISTS tests (
  test_symbol_id    INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  covers_symbol_id  INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  coverage_pct      REAL,
  PRIMARY KEY (test_symbol_id, covers_symbol_id)
);

CREATE TABLE IF NOT EXISTS commits (
  sha        TEXT PRIMARY KEY,
  ts         INTEGER,
  author     TEXT,
  message    TEXT
);

CREATE TABLE IF NOT EXISTS commit_changes (
  commit_sha   TEXT REFERENCES commits(sha) ON DELETE CASCADE,
  file_id      INTEGER REFERENCES files(id) ON DELETE CASCADE,
  symbol_id    INTEGER REFERENCES symbols(id),
  change_kind  TEXT,                  -- 'add' | 'modify' | 'delete' | 'rename'
  PRIMARY KEY (commit_sha, file_id, symbol_id)
);

CREATE INDEX IF NOT EXISTS ix_symbols_file       ON symbols(file_id);
CREATE INDEX IF NOT EXISTS ix_symbols_name       ON symbols(name);
CREATE INDEX IF NOT EXISTS ix_symbols_qual       ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS ix_symbol_refs_caller ON symbol_refs(caller_id);
CREATE INDEX IF NOT EXISTS ix_symbol_refs_callee ON symbol_refs(callee_id);
CREATE INDEX IF NOT EXISTS ix_files_language     ON files(language);
CREATE INDEX IF NOT EXISTS ix_commits_ts         ON commits(ts);

-- docstring_fts is created on-demand (requires FTS5)
