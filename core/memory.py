"""Memory subsystem — vector search over the markdown corpus (MVP).

Lazy-indexes memories/ + findings/ via sqlite-vec + sentence-transformers
(all-MiniLM-L6-v2, 22 MB, 384-dim, normalized embeddings → cosine via L2).
On each search, the corpus is re-walked and any file with mtime newer than
its indexed mtime is re-embedded. Old chunks for changed files are deleted
before reinsertion so we never serve stale content.

Storage: `memory.db` at lab root (sqlite-vec virtual table for embeddings,
plus a regular `chunks` table for path/content/mtime). Persisted across cycles.
The DB is gitignored (rebuildable from the markdown sources).

Scope cut for MVP: graph queries, status transitions, coherence detection,
compaction shapers, and the full 11-op API (view/str_replace/insert/delete/
rename/extract/stats) land later. This file ships only `search` and `create`.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import time
from pathlib import Path

from core import log

# Force HuggingFace Hub offline mode by default. The embedder model
# (all-MiniLM-L6-v2, 22 MB) is downloaded once into ~/.cache/huggingface
# and used offline thereafter — bert is free-tier-runtime per P-009 with
# data-stays-on-machine privacy positioning, so the library's default
# "phone home to check for model updates" behavior is wrong for bert.
# Without this, SentenceTransformer.__init__ can hang for minutes on
# slow/blocked HF Hub network (observed 2026-05-07: process sat at
# 3 sec CPU over 6 min wall-clock with zero observable network IO before
# we set this flag and got a 7.7 sec reindex). setdefault preserves
# user override (HF_HUB_OFFLINE=0 still allowed for first-time setup
# when the model needs initial download — handled by Phase C4
# onboarding wizard).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

LAB_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = LAB_ROOT / "memory.db"
INDEX_DIRS = [LAB_ROOT / "memories", LAB_ROOT / "findings"]


# ── Per-lab scoping (WS0b) ───────────────────────────────────────────
# When a benchmark / per-lab run sets an active lab via lab_context, the DB and
# index dirs follow it so retrieval hits the LAB's corpus, not bert's global
# memory.db. With no active lab, these fall back to the module constants above
# (DB_PATH/INDEX_DIRS/LAB_ROOT) — so production and the existing test suite
# (which monkeypatches those constants) are unchanged.

def _active_root() -> Path:
    try:
        from core import lab_context
        p = lab_context.get_active_lab_path()
    except Exception:  # noqa: BLE001 — lab_context optional; never break memory
        p = None
    return p if p is not None else LAB_ROOT


def _db_path() -> Path:
    from core import lab_context
    try:
        p = lab_context.get_active_lab_path()
    except Exception:  # noqa: BLE001
        p = None
    return (p / "memory.db") if p is not None else DB_PATH


def _index_dirs() -> list[Path]:
    from core import lab_context
    try:
        p = lab_context.get_active_lab_path()
    except Exception:  # noqa: BLE001
        p = None
    return [p / "memories", p / "findings"] if p is not None else list(INDEX_DIRS)
EMBED_DIM = 384
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 100

LOG = log.get_logger("bert.memory")

# Singletons (sentence-transformers init is ~3-5s + ~100MB RAM; sqlite conn
# is cheap but loading the vec extension is a one-time cost).
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None   # which db path _conn is bound to (reopen on change)
_embedder = None  # type: ignore[var-annotated]


# ── DB / embedder bootstrap ─────────────────────────────────────────


def _get_conn() -> sqlite3.Connection:
    global _conn, _conn_path
    dbp = str(_db_path())
    # Reuse the cached connection only if it's bound to the current lab's DB.
    # (_conn=None reset by tests, or an active-lab switch, forces a reopen.)
    if _conn is not None and _conn_path == dbp:
        return _conn
    import sqlite_vec
    Path(dbp).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(dbp)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            content TEXT NOT NULL,
            indexed_mtime REAL NOT NULL,
            UNIQUE(path, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
        """
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{EMBED_DIM}])"
    )
    _conn = conn
    _conn_path = dbp
    return conn


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        from core import config
        token = config.load().get("HF_TOKEN")  # auth → faster HF Hub access
        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        LOG.info(
            "loading embedder %s (offline=%s; first call; ~3-5s cached, "
            "~30-60s if model needs download)", EMBED_MODEL_NAME, offline
        )
        try:
            _embedder = SentenceTransformer(EMBED_MODEL_NAME, token=token)
        except Exception as e:
            # Likely cache miss on first-ever invocation. Retry with HF
            # Hub network enabled for one-time download. Subsequent loads
            # use the cache and the offline default kicks back in.
            if offline:
                LOG.warning(
                    "offline embedder load failed (%s); retrying with "
                    "HF_HUB_OFFLINE=0 for one-time download", e
                )
                os.environ["HF_HUB_OFFLINE"] = "0"
                try:
                    _embedder = SentenceTransformer(EMBED_MODEL_NAME, token=token)
                finally:
                    os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                raise
    return _embedder


def _embed_batch(texts: list[str]) -> list[bytes]:
    """Embed → list of float32 byte-blobs in sqlite-vec format."""
    model = _get_embedder()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [struct.pack(f"{EMBED_DIM}f", *e.tolist()) for e in embs]


# ── Chunking ────────────────────────────────────────────────────────


def _chunk(text: str, max_chars: int = CHUNK_CHARS,
           overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunking. Merges paragraphs up to max_chars; hard-splits
    paragraphs longer than max_chars with overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paragraphs:
        if not cur:
            cur = p
        elif len(cur) + len(p) + 2 <= max_chars:
            cur = cur + "\n\n" + p
        else:
            chunks.append(cur)
            cur = p
        if len(cur) > max_chars:
            # Hard-split the current chunk
            chunks.extend(cur[i : i + max_chars] for i in range(0, len(cur), max_chars - overlap))
            cur = ""
    if cur:
        chunks.append(cur)
    return chunks


# ── Indexing ────────────────────────────────────────────────────────


def _list_corpus_files() -> list[Path]:
    """Live corpus files to index. Excludes any */archive/* path (demand paging,
    page-out half): findings/memories the consolidator archived as stale are
    historical — they must not be embedded or surface in live search. Already-
    indexed files that get archived are GC'd on the next re-index (they leave
    this listing, so _index_corpus's orphan removal drops their chunks)."""
    files: list[Path] = []
    for d in _index_dirs():
        if d.exists():
            files.extend(f for f in d.rglob("*.md") if "archive" not in f.parts)
    return sorted(files)


_INDEX_CORPUS_CACHE: dict[str, float] = {}    # cache_key → next_check_ts
_INDEX_CORPUS_TTL_S = 5.0   # how long to trust a "0 changes" verdict


def _invalidate_index_corpus_cache() -> None:
    """Ingest paths call this to force a fresh corpus walk on the next
    search. Cheap (one dict.clear). Equivalent to a TLB shootdown."""
    _INDEX_CORPUS_CACHE.clear()

def _index_corpus() -> int:
    """Re-walk the corpus and (re-)embed any file with mtime > indexed_mtime.
    Also removes index entries for files that no longer exist on disk.

    Returns the number of chunks (re-)indexed (deletions don't count).

    Hot-path optimization (2026-05-26): the corpus walk + DB query is
    4ms per call but almost always finds 0 changes. We cache the
    "checked at" timestamp and skip the walk for `_INDEX_CORPUS_TTL_S`
    seconds. If a corpus mutation happens during that window, the
    next walk catches it (mtime comparison is per-file). For workloads
    that ingest then immediately search, ingest can call
    `_invalidate_index_corpus_cache()` directly.

    BB.1 — Two new escape hatches:
      1. `BERT_SKIP_INDEXER=1` short-circuits this function. Used by
         robustness tests that don't care about freshness.
      2. Graceful degradation: sqlite errors (corrupted DB, locked
         schema, missing extension) are caught and logged; the function
         returns 0 instead of hanging or crashing. Robustness tests
         specifically corrupt the DB to verify this.
    """
    import os as _os
    if _os.environ.get("BERT_SKIP_INDEXER") == "1":
        return 0
    # TTL cache: if we walked recently and found 0 changes, trust that
    # for a few seconds. Ingest paths bump the cache.
    cache_key = str(_active_root())   # per-lab TTL cache key
    now = time.time()
    next_check = _INDEX_CORPUS_CACHE.get(cache_key, 0.0)
    if now < next_check:
        return 0
    try:
        conn = _get_conn()
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
        LOG.warning("indexer: _get_conn failed (db corrupt or locked) — "
                    "skipping re-index: %s", exc)
        return 0
    on_disk = {str(f.relative_to(_active_root())): f for f in _list_corpus_files()}

    # 1. Garbage-collect: drop chunks for files no longer on disk
    try:
        indexed_paths = {r[0] for r in conn.execute(
            "SELECT DISTINCT path FROM chunks"
        ).fetchall()}
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
        LOG.warning("indexer: schema query failed (db corrupt?) — "
                    "skipping re-index: %s", exc)
        return 0
    orphans = indexed_paths - on_disk.keys()
    if orphans:
        LOG.info("garbage-collecting %d orphaned files from index", len(orphans))
        for rel in orphans:
            old_ids = [r[0] for r in conn.execute(
                "SELECT id FROM chunks WHERE path = ?", [rel]
            ).fetchall()]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", old_ids)
                conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})", old_ids)
        conn.commit()

    # 2. Find files needing (re-)indexing
    to_index: list[tuple[Path, str, float]] = []
    for rel, f in on_disk.items():
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        row = conn.execute(
            "SELECT MAX(indexed_mtime) FROM chunks WHERE path = ?", [rel]
        ).fetchone()
        existing = (row[0] or 0) if row else 0
        if mtime > existing:
            to_index.append((f, rel, mtime))

    if not to_index:
        # 0-changes verdict — extend the TTL cache so subsequent
        # searches in the next TTL window skip this walk entirely.
        _INDEX_CORPUS_CACHE[cache_key] = now + _INDEX_CORPUS_TTL_S
        return 0

    LOG.info("indexing %d changed files", len(to_index))
    n_chunks = 0
    for f, rel, mtime in to_index:
        # Delete prior chunks for this file (both regular and vec rows)
        old_ids = [r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE path = ?", [rel]
        ).fetchall()]
        if old_ids:
            placeholders = ",".join("?" * len(old_ids))
            conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", old_ids)
            conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})", old_ids)

        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, IsADirectoryError):
            continue
        chunks = _chunk(text)
        if not chunks:
            continue
        embs = _embed_batch(chunks)
        for i, (ch, emb) in enumerate(zip(chunks, embs, strict=False)):
            cur = conn.execute(
                "INSERT INTO chunks (path, chunk_idx, content, indexed_mtime) "
                "VALUES (?, ?, ?, ?)",
                [rel, i, ch, mtime],
            )
            chunk_id = cur.lastrowid
            conn.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                [chunk_id, emb],
            )
        n_chunks += len(chunks)

    conn.commit()
    LOG.info("indexed %d chunks across %d files", n_chunks, len(to_index))
    return n_chunks


# ── Public API ──────────────────────────────────────────────────────


def search(query: str, k: int = 5) -> list[dict]:
    """Vector search across the indexed memories + findings corpus.

    Returns list of {path, chunk_idx, content, distance} sorted by distance
    ascending (lower = more similar). Auto-triggers re-index for stale files.
    """
    if not query or not query.strip():
        return []
    _index_corpus()
    conn = _get_conn()
    q_emb = _embed_batch([query])[0]
    k = max(1, min(k, 20))
    rows = conn.execute(
        """
        SELECT chunks.path, chunks.chunk_idx, chunks.content, vec.distance
        FROM (
            SELECT rowid, distance FROM vec_chunks
            WHERE embedding MATCH ? AND k = ?
        ) AS vec
        JOIN chunks ON chunks.id = vec.rowid
        ORDER BY vec.distance
        """,
        [q_emb, k],
    ).fetchall()
    return [
        {"path": r[0], "chunk_idx": r[1], "content": r[2], "distance": float(r[3])}
        for r in rows
    ]


def create(path: str, content: str) -> dict:
    """Write a memory file (path scoped under memories/ or findings/).

    Atomic write (tmp + rename). Returns dict with ok, path, bytes, error.
    Re-indexing happens lazily on the next search() call (mtime-driven).
    """
    root = _active_root()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    try:
        rel = p.resolve().relative_to(root.resolve())
    except ValueError:
        return {
            "ok": False, "path": str(p), "bytes": 0,
            "error": f"path resolves outside lab root: {p}",
        }
    rel_s = str(rel)
    if not (rel_s.startswith("memories/") or rel_s.startswith("findings/")):
        return {
            "ok": False, "path": rel_s, "bytes": 0,
            "error": (
                "memory_create requires path under memories/ or findings/. "
                "For state, code, or other artifacts, use Write instead."
            ),
        }
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)
    return {"ok": True, "path": rel_s, "bytes": len(content), "error": ""}


def ingest_corpus(src_dir, *, dest: str = "findings/corpus",
                  exts=(".py", ".md", ".txt", ".rst", ".js", ".ts", ".go",
                        ".java", ".yaml", ".yml", ".json", ".toml"),
                  eager_index: bool = False) -> int:
    """Ingest an EXTERNAL corpus tree into the active lab so RAG can retrieve
    over it (WS0b). Each source file is written as a shard under the active lab's
    `dest/` (so create()'s findings/ gate accepts it) preserving relative path.
    Returns the number of files written. With eager_index=True, embeds them now;
    otherwise the next search() lazily indexes (the TTL cache is invalidated).

    Requires an active lab (lab_context.set_active_lab_path) or it writes under
    the repo root — pass an isolated benchmark lab to avoid polluting bert's DB."""
    src = Path(src_dir).expanduser().resolve()
    written = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = f.relative_to(src)
        # Shard path: dest/<relpath>.md so create()'s .md indexer picks it up.
        shard = f"{dest}/{rel}.md"
        res = create(shard, content)
        if res.get("ok"):
            written += 1
    _invalidate_index_corpus_cache()
    if eager_index:
        _index_corpus()
    return written


def stats() -> dict:
    """Quick corpus stats — files indexed, chunks indexed, db size."""
    conn = _get_conn()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_files = conn.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
    _dbp = _db_path()
    db_bytes = _dbp.stat().st_size if _dbp.exists() else 0
    return {
        "files_indexed": total_files,
        "chunks_indexed": total_chunks,
        "db_bytes": db_bytes,
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_dim": EMBED_DIM,
    }


def cli(op: str, args: list[str]) -> int:
    """Minimal CLI wrapper for `python lab.py memory <op>`."""
    if op == "search":
        if not args:
            print("usage: memory search <query> [k]")
            return 1
        query = args[0]
        k = int(args[1]) if len(args) > 1 else 5
        for hit in search(query, k):
            print(f"[{hit['distance']:.3f}] {hit['path']}#chunk{hit['chunk_idx']}")
            preview = hit["content"][:200].replace("\n", " ")
            print(f"   {preview}")
        return 0
    if op == "stats":
        s = stats()
        for k, v in s.items():
            print(f"  {k}: {v}")
        return 0
    if op == "index":
        n = _index_corpus()
        print(f"indexed {n} new/changed chunks")
        return 0
    print(f"memory: op '{op}' not yet implemented in MVP "
          f"(supported: search, stats, index)")
    return 1


__all__ = ["search", "create", "stats", "cli"]
