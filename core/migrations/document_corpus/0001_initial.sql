-- DocumentCorpusAdapter baseline schema (v1)
--
-- Phase B lands the concrete adapter that uses these tables. This
-- migration ships the schema so adapters created after Phase A6 can
-- bootstrap into a known-good state.

CREATE TABLE IF NOT EXISTS documents (
  id            INTEGER PRIMARY KEY,
  path          TEXT UNIQUE NOT NULL,
  kind          TEXT,                -- 'pdf' | 'html' | 'md' | 'arxiv' | 'web'
  title         TEXT,
  source_url    TEXT,
  ingest_ts     INTEGER NOT NULL,
  content_hash  TEXT,
  size_bytes    INTEGER,
  language      TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id            INTEGER PRIMARY KEY,
  doc_id        INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  chunk_idx     INTEGER NOT NULL,
  content       TEXT NOT NULL,
  char_start    INTEGER,
  char_end      INTEGER,
  section_path  TEXT,
  mtime_indexed INTEGER
);

CREATE INDEX IF NOT EXISTS ix_chunks_doc      ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_chunks_section  ON chunks(section_path);
CREATE INDEX IF NOT EXISTS ix_documents_kind  ON documents(kind);

-- chunks_vec and chunks_fts virtual tables are created on-demand by
-- the adapter (require sqlite-vec / fts5 extensions loaded at runtime).
