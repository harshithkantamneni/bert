"""Memory subsystem — vector search over the markdown corpus (MVP).

Lazy-indexes memories/ + findings/ via sqlite-vec + sentence-transformers
(BAAI/bge-base-en-v1.5, 768-dim, normalized embeddings → cosine via L2).
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

import json
import os
import sqlite3
import struct
import time
from pathlib import Path

from core import log

# Force HuggingFace Hub offline mode by default. The embedder model
# (BAAI/bge-base-en-v1.5, ~440 MB) is downloaded once into ~/.cache/huggingface
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
# Embedder. Default upgraded from all-MiniLM-L6-v2 (2020, 384-dim) to
# BAAI/bge-base-en-v1.5 (2023, 768-dim) — a materially stronger retrieval
# encoder (see benchmarks/results/B2_BEIR_RESULT.md). Both the model and dim
# are env-overridable so the BEIR harness can sweep encoders and pick on
# measured nDCG@10. EMBED_DIM is baked into the vec0 schema below, so changing
# it requires rebuilding memory.db (gitignored + rebuildable from the markdown).
EMBED_MODEL_NAME = os.environ.get("BERT_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIM = int(os.environ.get("BERT_EMBED_DIM", "768"))


def _default_affixes(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) for a retrieval encoder.

    Modern retrieval encoders are asymmetric: the query is encoded with a short
    instruction the passages don't get. bge-*-en-v1.5 prefixes only the query;
    the e5 family prefixes both. Symmetric models (MiniLM) use neither. These
    are the model authors' documented retrieval instructions — omitting the bge
    query instruction silently costs several nDCG points.
    """
    m = model_name.lower()
    if "bge" in m and "v1.5" in m:
        return ("Represent this sentence for searching relevant passages: ", "")
    if "e5" in m:  # intfloat/e5-* family
        return ("query: ", "passage: ")
    return ("", "")


_q_pref, _p_pref = _default_affixes(EMBED_MODEL_NAME)
EMBED_QUERY_PREFIX = os.environ.get("BERT_EMBED_QUERY_PREFIX", _q_pref)
EMBED_PASSAGE_PREFIX = os.environ.get("BERT_EMBED_PASSAGE_PREFIX", _p_pref)
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


def _embed_batch(texts: list[str], *, is_query: bool = False) -> list[bytes]:
    """Embed → list of float32 byte-blobs in sqlite-vec format.

    is_query selects the asymmetric affix: queries get EMBED_QUERY_PREFIX,
    passages get EMBED_PASSAGE_PREFIX. For symmetric encoders both are empty,
    so indexing and search encode identically (the historical behavior).
    """
    model = _get_embedder()
    prefix = EMBED_QUERY_PREFIX if is_query else EMBED_PASSAGE_PREFIX
    if prefix:
        texts = [prefix + t for t in texts]
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
    q_emb = _embed_batch([query], is_query=True)[0]
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


# ── External-corpus ingest (incremental) ────────────────────────────

_DEFAULT_INGEST_EXTS = (
    ".py", ".md", ".txt", ".rst", ".js", ".ts", ".go", ".java",
    ".yaml", ".yml", ".json", ".toml",
)
_INGEST_EXCLUDE_DIRS = {
    "node_modules", "__pycache__", "dist", "build", "target",
    "site-packages", ".git", ".venv", ".next", ".idea",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
_INGEST_SOURCES_FILE = "state/ingest_sources.json"
_RESYNC_CACHE: dict[str, float] = {}    # cache_key → next_check_ts
_RESYNC_TTL_S = 5.0


def _ingest_cap(env_name: str, default: int, override: int | None) -> int:
    if override is not None:
        return override
    import os as _os
    try:
        return int(_os.environ.get(env_name, str(default)))
    except ValueError:
        return default


def _is_excluded(rel: Path, exclude_dirs: set[str]) -> bool:
    return any(
        part in exclude_dirs or part.startswith(".") for part in rel.parts
    )


def ingest_corpus_report(
    src_dir,
    *,
    dest: str = "findings/corpus",
    exts=_DEFAULT_INGEST_EXTS,
    eager_index: bool = False,
    max_files: int | None = None,
    max_bytes: int | None = None,
    exclude_dirs: set[str] | None = None,
) -> dict:
    """Incrementally ingest an EXTERNAL source tree into the active lab as `.md`
    shards under `dest/`, so RAG can retrieve over it.

    Incremental: a source file is re-sharded only when its mtime exceeds its
    existing shard's mtime; unchanged in-corpus files are counted as `skipped`.
    Shards whose source file no longer exists are garbage-collected (skipped
    when a cap truncates the run, since the source list is then incomplete).

    Returns {written, skipped, removed, truncated, total_seen}. Writes go through
    create(), which enforces the path-traversal gate (dest is under findings/).
    Requires an active lab (lab_context.set_active_lab_path); otherwise shards
    land under the repo root — pass an isolated lab to avoid polluting the DB."""
    src = Path(src_dir).expanduser().resolve()
    root = _active_root()
    excl = exclude_dirs if exclude_dirs is not None else set(_INGEST_EXCLUDE_DIRS)
    cap_files = _ingest_cap("BERT_INGEST_MAX_FILES", 5000, max_files)
    cap_bytes = _ingest_cap("BERT_INGEST_MAX_BYTES", 100 * 1024 * 1024, max_bytes)
    exts_l = tuple(e.lower() for e in exts)

    written = skipped = removed = total = total_bytes = 0
    truncated = False
    expected: set[str] = set()

    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        try:
            rel = f.relative_to(src)
        except ValueError:
            continue
        if _is_excluded(rel, excl) or f.suffix.lower() not in exts_l:
            continue
        try:
            if f.is_symlink() and not str(f.resolve()).startswith(str(src)):
                continue
        except OSError:
            continue
        if total >= cap_files:
            truncated = True
            break
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        nbytes = len(content.encode("utf-8", "replace"))
        if total_bytes + nbytes > cap_bytes:
            truncated = True
            break
        total += 1
        total_bytes += nbytes

        # Shard path: dest/<relpath>.md so create()'s .md indexer picks it up.
        shard_rel = f"{dest}/{rel}.md"
        expected.add(shard_rel)
        shard_abs = root / shard_rel
        try:
            src_mtime = f.stat().st_mtime
        except OSError:
            continue
        if shard_abs.exists() and src_mtime <= shard_abs.stat().st_mtime:
            skipped += 1
            continue
        if create(shard_rel, content).get("ok"):
            written += 1

    # Garbage-collect shards whose source vanished — only on a complete walk
    # (a truncated run has an incomplete `expected` set).
    if not truncated:
        corpus_root = root / dest
        if corpus_root.exists():
            for shard in corpus_root.rglob("*.md"):
                try:
                    rel_s = str(shard.relative_to(root))
                except ValueError:
                    continue
                if rel_s not in expected:
                    try:
                        shard.unlink()
                        removed += 1
                    except OSError:
                        pass

    _invalidate_index_corpus_cache()
    if eager_index and (written or removed):
        _index_corpus()
    return {
        "written": written,
        "skipped": skipped,
        "removed": removed,
        "truncated": truncated,
        "total_seen": total,
    }


def ingest_corpus(src_dir, *, dest: str = "findings/corpus",
                  exts=_DEFAULT_INGEST_EXTS, eager_index: bool = False) -> int:
    """Backward-compatible wrapper around ingest_corpus_report; returns the
    number of files (re-)written this call."""
    return ingest_corpus_report(
        src_dir, dest=dest, exts=exts, eager_index=eager_index,
    )["written"]


# ── Ingested-source registry + search-time auto-resync ──────────────


def _ingest_sources_path() -> Path:
    return _active_root() / _INGEST_SOURCES_FILE


def list_ingest_sources() -> list[dict]:
    """Source roots previously registered for the active lab (for auto-resync)."""
    p = _ingest_sources_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def register_ingest_source(source, *, exts=None,
                           dest: str = "findings/corpus") -> dict:
    """Record an external source root so search-time auto-resync can keep it
    current. Idempotent: re-registering the same resolved path updates in place."""
    src_abs = str(Path(source).expanduser().resolve())
    entry = {"source": src_abs, "exts": list(exts) if exts else None, "dest": dest}
    sources = [s for s in list_ingest_sources() if s.get("source") != src_abs]
    sources.append(entry)
    p = _ingest_sources_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    tmp.replace(p)
    return {"ok": True, "count": len(sources)}


def resync_sources(*, force: bool = False, eager_index: bool = True) -> dict:
    """Incrementally re-ingest every registered source root for the active lab.
    TTL-gated (skipped within _RESYNC_TTL_S of the last run) unless force=True,
    so frequent searches don't re-walk large trees. Best-effort: a failing
    source is skipped, never raised."""
    cache_key = str(_active_root())
    now = time.time()
    if not force and now < _RESYNC_CACHE.get(cache_key, 0.0):
        return {"sources": 0, "written": 0, "skipped": 0,
                "removed": 0, "cached": True}
    written = skipped = removed = n_sources = 0
    for entry in list_ingest_sources():
        src = entry.get("source")
        if not src or not Path(src).is_dir():
            continue
        exts = entry.get("exts")
        try:
            rep = ingest_corpus_report(
                src,
                dest=entry.get("dest", "findings/corpus"),
                exts=tuple(exts) if exts else _DEFAULT_INGEST_EXTS,
                eager_index=eager_index,
            )
        except Exception:  # noqa: BLE001 — one bad source must not break resync
            continue
        n_sources += 1
        written += rep["written"]
        skipped += rep["skipped"]
        removed += rep["removed"]
    _RESYNC_CACHE[cache_key] = now + _RESYNC_TTL_S
    return {"sources": n_sources, "written": written,
            "skipped": skipped, "removed": removed, "cached": False}


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
