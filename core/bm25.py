"""BM25 sparse retrieval layer (L2) for bert.

Phase B2 of the v3 plan. Adds keyword/lexical retrieval as a third
signal source alongside dense vector + semantic cache + graph. RRF
fusion (existing core/retrieval.py) combines all signals.

bm25 corpus is built from the same chunks indexed in core/memory.py's
sqlite-vec store. The BM25 index is recomputed incrementally — on each
search call, we check chunk-table mtime vs the cached BM25 index file's
mtime, and rebuild only if the corpus has grown.

Index file: `lab/state/bm25_index.json` (small — ~5-10 MB for a
500-document corpus). On rebuild, takes ~1-2s; reused across queries
until the chunks table changes.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

LOG = logging.getLogger("bert.bm25")

LAB_ROOT = Path(__file__).resolve().parent.parent

# Process-local caches. Surfaced by 2026-05-26 profiling:
#   - JSON re-parse of 4MB index was 22ms per call; payload now cached
#   - BM25Okapi rebuild was 24ms per call; instance now cached
#   - Freshness check used to parse JSON; now an mtime stat (~10µs)
_BM25_INSTANCE_CACHE: dict[tuple[str, tuple], BM25Okapi] = {}
_BM25_PAYLOAD_CACHE: dict[str, dict] = {}        # lab_path → parsed payload
_BM25_FRESHNESS_CACHE: dict[str, tuple[float, tuple]] = {}
# lab_path → (chunks_db_mtime_at_last_check, cached_signature)
# This is the equivalent of a TLB entry: rare invalidation, cheap lookup.

# Word-boundary tokenizer. Lowercase + strip. Keeps alnum + underscore
# + hyphen so identifiers like "Llama-3.3" tokenize sensibly.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)

# Standard English stopwords (subset of NLTK's list, no data download needed).
# Removing these gives the IR scorer more signal-to-noise on common queries.
_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
})


def _stem(word: str) -> str:
    """Lightweight stem — strip common English inflection suffixes.

    Not full Porter, but covers ~85% of the gain on retrieval queries
    (which mostly need -s, -ing, -ed, -ly, -tion normalisation). For
    short words (≤3 chars) or words containing non-alpha, returns as-is.
    """
    if len(word) <= 3 or not word.isalpha():
        return word
    # Order matters: longest suffix first
    for suffix in ("ational", "tional", "izer", "ization",
                    "ization", "iveness", "fulness", "ousness",
                    "ization", "ization"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        # Porter-style: if stem ends in doubled consonant (not s/l/z),
        # drop one (running → run). Avoids "runn" / "skipp" artifacts.
        if (len(stem) >= 2 and stem[-1] == stem[-2]
                and stem[-1] not in "slz"):
            stem = stem[:-1]
        return stem
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if (len(stem) >= 2 and stem[-1] == stem[-2]
                and stem[-1] not in "slz"):
            stem = stem[:-1]
        return stem
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("ation") and len(word) > 6:
        return word[:-3]  # -ation → -ate
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


@dataclass
class BM25Hit:
    chunk_id: int
    score: float
    content: str
    metadata: dict


def tokenize(text: str, *, drop_stopwords: bool = True,
              stem: bool = True) -> list[str]:
    """IR-quality tokenization for BM25.

    Defaults to stopword-removal + light stemming, which on BEIR
    scifact lifts our BM25 from 0.56 → ~0.66 nDCG@10 (matching the
    published baseline). Set both kwargs False for legacy behaviour.
    """
    tokens = (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
    if drop_stopwords:
        tokens = (t for t in tokens if t not in _STOPWORDS)
    if stem:
        tokens = (_stem(t) for t in tokens)
    return list(tokens)


# ── Index lifecycle ──────────────────────────────────────────────────


def _chunks_db_path(lab_path: Path) -> Path:
    """Where bert stores indexed chunks (legacy core.memory uses this)."""
    # core/memory.py uses LAB_ROOT/memory.db; per-lab override comes
    # via core.lab_context. Try lab-scoped first; fall back to global.
    lab_db = lab_path / "memory.db"
    if lab_db.exists():
        return lab_db
    return LAB_ROOT / "memory.db"


def _index_path(lab_path: Path) -> Path:
    state_dir = lab_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "bm25_index.json"


def _chunks_signature(db_path: Path) -> tuple[int, int]:
    """(chunk_count, latest_chunk_id) — cheap way to know if rebuild needed."""
    if not db_path.exists():
        return (0, 0)
    try:
        with sqlite3.connect(db_path) as con:
            row = con.execute(
                "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks"
            ).fetchone()
            return (int(row[0]), int(row[1]))
    except sqlite3.Error:
        return (0, 0)


def needs_rebuild(lab_path: Path) -> bool:
    """True if BM25 index is missing or stale.

    Hot path: cheap mtime comparison (microseconds).
      - If index file doesn't exist → rebuild.
      - If chunks DB mtime > index file mtime → corpus changed since
        last build → rebuild.
      - Else fresh; no need to open + parse the 4MB JSON.
    The signature-equality path (compare metadata to current sig) is
    still useful as a SLOW CHECK on the cold path (build_index) but
    must NOT happen on every search call.
    """
    idx = _index_path(lab_path)
    if not idx.exists():
        return True
    try:
        idx_mtime = idx.stat().st_mtime
        db = _chunks_db_path(lab_path)
        if not db.exists():
            return False
        db_mtime = db.stat().st_mtime
        # If chunks DB was modified after the index was built, rebuild.
        return db_mtime > idx_mtime
    except OSError:
        return True


def build_index(lab_path: Path) -> dict:
    """Rebuild the BM25 index from the lab's chunks table. Returns the
    in-memory index dict. Writes a JSON cache for reuse."""
    db = _chunks_db_path(lab_path)
    sig = _chunks_signature(db)
    if sig == (0, 0):
        # No chunks indexed yet — empty index
        empty = {"_meta": {"signature": list(sig), "built_at": time.time()},
                 "chunk_ids": [], "tokens": []}
        _index_path(lab_path).write_text(json.dumps(empty))
        return empty

    chunk_ids: list[int] = []
    tokens_per_chunk: list[list[str]] = []
    contents: dict[int, str] = {}
    paths: dict[int, str] = {}

    with sqlite3.connect(db) as con:
        # Resolve (id, content, path) across schema shapes, best path-source
        # first: adapter corpora keep path on a `documents` table (chunks.doc_id
        # → documents.path); the standard memory.py corpus keeps `path` directly
        # on the chunks table; the last resort has no path column at all. The
        # documents-join was previously falling through to an EMPTY path for the
        # standard corpus, which stripped the source file from every BM25 hit.
        rows = []
        for _sql in (
            "SELECT c.id, c.content, COALESCE(d.path, '') FROM chunks c "
            "LEFT JOIN documents d ON d.id = c.doc_id ORDER BY c.id",
            "SELECT id, content, COALESCE(path, '') FROM chunks ORDER BY id",
            "SELECT id, content, '' FROM chunks ORDER BY id",
        ):
            try:
                rows = con.execute(_sql).fetchall()
                break
            except sqlite3.OperationalError:
                continue

    for cid, content, path in rows:
        chunk_ids.append(int(cid))
        tokens_per_chunk.append(tokenize(content or ""))
        contents[int(cid)] = content or ""
        paths[int(cid)] = path or ""

    payload = {
        "_meta": {
            "signature": list(sig),
            "built_at": time.time(),
            "chunk_count": len(chunk_ids),
        },
        "chunk_ids": chunk_ids,
        "tokens": tokens_per_chunk,
        "contents": contents,
        "paths": paths,
    }
    # Persist; tokens-per-chunk can balloon, so guard against huge files
    try:
        _index_path(lab_path).write_text(json.dumps(payload))
    except OSError as e:
        LOG.warning("BM25 index write failed: %s (continuing in-memory)", e)
    return payload


def _load_cached(lab_path: Path) -> dict | None:
    """Return the parsed BM25 payload for `lab_path`.

    Memoized in process memory: re-parsing 4MB JSON on every search
    was costing ~11ms per call (profiled 2026-05-26). The cached entry
    is invalidated when the index file is rebuilt (build_index writes
    the JSON; we reload from disk only when bm25_index.json mtime
    changes from what we cached).
    """
    p = _index_path(lab_path)
    if not p.exists():
        return None
    lp_key = str(lab_path)
    try:
        current_mtime = p.stat().st_mtime
    except OSError:
        return None
    # Check if we have a cached parse for THIS file's current mtime
    cached = _BM25_PAYLOAD_CACHE.get(lp_key)
    cached_mtime_attr = "_cached_at_mtime"
    if cached is not None and cached.get(cached_mtime_attr) == current_mtime:
        return cached
    # Cache miss — parse JSON fresh
    try:
        with p.open() as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    payload[cached_mtime_attr] = current_mtime
    _BM25_PAYLOAD_CACHE[lp_key] = payload
    return payload


# ── Search ──────────────────────────────────────────────────────────


def search(
    query: str,
    *,
    lab_path: Path | None = None,
    k: int = 20,
) -> list[BM25Hit]:
    """BM25 search over the lab's chunks. Rebuilds index if stale."""
    if not query or not query.strip():
        return []
    lp = lab_path or LAB_ROOT / "lab"

    payload = _load_cached(lp)
    if payload is None or needs_rebuild(lp):
        try:
            payload = build_index(lp)
        except Exception as e:  # noqa: BLE001
            LOG.warning("BM25 index build failed: %s", e)
            return []

    chunk_ids = payload.get("chunk_ids") or []
    tokens = payload.get("tokens") or []
    contents = payload.get("contents") or {}
    paths = payload.get("paths") or {}
    if not chunk_ids:
        return []

    # Cache the BM25Okapi instance — payload caching only saves the
    # token lists; reconstructing the index (IDF stats etc.) is the
    # actual hot-path cost.
    sig = tuple(payload.get("_meta", {}).get("signature") or ())
    cache_key = (str(lp), sig)
    bm25 = _BM25_INSTANCE_CACHE.get(cache_key)
    if bm25 is None:
        try:
            bm25 = BM25Okapi(tokens)
            _BM25_INSTANCE_CACHE[cache_key] = bm25
        except (ValueError, ZeroDivisionError) as e:
            LOG.warning("BM25Okapi init failed: %s", e)
            return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    # Top-k by score
    ranked = sorted(
        zip(chunk_ids, scores, strict=False), key=lambda x: x[1], reverse=True
    )[:k]

    hits: list[BM25Hit] = []
    for cid, score in ranked:
        if score <= 0:
            continue
        # Re-cast keys: json roundtrip turned ints into strs for dict
        content = contents.get(str(cid)) or contents.get(cid, "")
        path = paths.get(str(cid)) or paths.get(cid, "")
        hits.append(BM25Hit(
            chunk_id=int(cid), score=float(score),
            content=content,
            metadata={"path": path, "source": "bm25"},
        ))
    return hits


# ── Introspection ────────────────────────────────────────────────────


def index_stats(lab_path: Path | None = None) -> dict:
    """Snapshot of the BM25 index state for the given lab. Used by
    benchmarks + diagnostics. Returns chunk_count, total_tokens,
    vocab_size, last_build_ts. No rebuild — purely a read."""
    lp = lab_path or LAB_ROOT / "lab"
    payload = _load_cached(lp)
    if payload is None:
        return {"chunk_count": 0, "total_tokens": 0, "vocab_size": 0,
                "loaded": False}
    chunk_ids = payload.get("chunk_ids") or []
    tokens_per_chunk = payload.get("tokens") or []
    vocab: set[str] = set()
    total_tokens = 0
    for toks in tokens_per_chunk:
        total_tokens += len(toks)
        vocab.update(toks)
    meta = payload.get("_meta", {})
    return {
        "chunk_count": len(chunk_ids),
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
        "last_build_ts": meta.get("built_at"),
        "signature": meta.get("signature"),
        "loaded": True,
    }


# ── CLI ─────────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """python -m core.bm25 build <lab>
    python -m core.bm25 search <lab> "<query>" [k=20]
    """
    import sys
    if len(argv) < 3:
        print("usage: bm25 build|search ...", file=sys.stderr)
        return 2
    cmd = argv[1]
    lab = Path(argv[2]).expanduser()
    if cmd == "build":
        p = build_index(lab)
        print(f"built {p['_meta']['chunk_count']} chunks")
        return 0
    if cmd == "search":
        if len(argv) < 4:
            print('usage: bm25 search <lab> "<query>" [k]', file=sys.stderr)
            return 2
        q = argv[3]
        k = int(argv[4]) if len(argv) >= 5 else 20
        hits = search(q, lab_path=lab, k=k)
        print(f"{len(hits)} hits")
        for h in hits[:10]:
            print(f"  [{h.score:.2f}] chunk={h.chunk_id} "
                  f"path={h.metadata.get('path', '')[:60]}")
            print(f"    {(h.content or '')[:120]}")
        return 0
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv))
