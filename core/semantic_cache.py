"""Semantic dispatch cache (F.12 / post-F follow-up).

Closes the last gap in bert's caching story: near-duplicate dispatches
hit a cached answer instead of burning tokens. Sits *above* the
prefix-cache layer (Ollama native + Gemini implicit + Groq automatic)
that H1 + E.4 already wired.

Design

  Exact-match caches need byte-identical inputs. Bert's actual
  pattern is more like "asks similar questions over and over from
  different cycle contexts" — the prefix is stable, but the per-call
  delta is plain prose that varies slightly each time. Exact-match
  misses; semantic match wins.

  Embeddings come from Ollama's nomic-embed-text:latest (free, local,
  768-dim). Cosine similarity ≥ threshold → hit. Default threshold
  0.95; tighter than the typical 0.9 because verdict drift is more
  costly than a wasted dispatch.

  SQLite-backed at lab/state/semantic_cache.db; each row is
  {id, role, prompt_hash, embedding (BLOB), output (TEXT),
  output_meta (JSON), written_at, ttl_secs, hit_count}.

Role discipline

  This is the LOAD-BEARING design decision. Caching applies *only*
  to roles where stale answers are cheap. Default `CACHEABLE_ROLES`:

    researcher        fact / synthesis lookup — same question often
                      asked across cycles
    threshing         eligibility check — same dispatch shape repeats
    implementer       code / patch generation — pattern-heavy
    clearness_phase1  query drafting — open queries don't carry verdict
                      risk
    strategist        option enumeration — when explicitly approved

  Roles explicitly EXCLUDED:

    evaluator         verdicts must be fresh; P-VS-02 requires
                      independent cross-family judgment
    clearness_phase2  verdict + concerns — same reason
    director          orchestration decisions — context-dependent

  Caller can override the default set; the discipline is enforced by
  `get_or_compute()` which falls through to compute_fn for any role
  not in cacheable_roles.

Telemetry

  hit_count incremented on every hit. cache_stats(role) surfaces
  per-role hit rates over a 24h window so the /api/semantic-cache
  endpoint can show whether the layer is actually saving dispatches
  vs serving stale answers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("bert.semantic_cache")
LAB_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = LAB_ROOT / "lab" / "state" / "semantic_cache.db"
_LOCK = threading.Lock()

# Default cache TTL (24h). Caller can override per get_or_compute call.
DEFAULT_TTL_SECS = 86400
# Default similarity threshold. Above this = hit. Below = miss + new entry.
# Calibrated against nomic-embed-text:latest 2026-05-13. Two findings
# drive this number:
#
#   (1) The embedding model has a weakness: pairs like "synthesize on
#       KVComm" vs "synthesize on LatentMAS" produce byte-IDENTICAL
#       vectors (cosine = 1.0) because it under-weights short topic
#       suffixes. Cosine threshold alone can't fix this. The anchor-
#       term guard below catches it — that's our safety net for the
#       topic-collision failure mode.
#
#   (2) Meaningful matches on bert's actual prompts cluster around
#       0.90-0.96. Examples:
#         "should this proceed forward" / "...onward"        → 0.94
#         "threshing dispatch eligibility" / "...check"       → 0.96
#         "evaluate the KVComm threshing pass/decision"       → 0.81
#       Setting threshold > 0.93 misses the obvious paraphrases we
#       want to hit. 0.90 catches them while still being well above
#       the noise floor (~0.6 for unrelated prompts).
#
# Net: 0.90 cosine threshold + anchor-term guard = both halves of the
# safety story. Verdict roles still excluded from CACHEABLE_ROLES.
DEFAULT_THRESHOLD = 0.90
# Default embedding model — bert ships with nomic-embed-text:latest on Ollama.
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Cacheable roles — explicit allow-list, conservative defaults after
# 2026-05-13 calibration. Excluded with rationale:
#   evaluator         verdicts need fresh cross-family judgment (P-VS-02)
#   clearness_phase2  same — verdict + concerns
#   director          orchestration is context-dependent
#   researcher        embedding-model collapse on "synthesize on X" patterns
#   strategist        same — option exploration should be fresh
#   clearness_phase1  queries need fresh framing per dispatch
CACHEABLE_ROLES: frozenset[str] = frozenset({
    "threshing", "threshing_pass",
    "implementer",
})


@dataclass
class CacheHit:
    id: str
    role: str
    output: str
    output_meta: dict
    similarity: float
    age_secs: float
    hit_count: int


@dataclass
class CacheStats:
    role: str
    rows: int
    hits_24h: int
    misses_24h: int
    hit_rate: float
    avg_similarity_on_hit: float


# ── SQLite + serialization plumbing ──────────────────────────────────


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            prompt_text TEXT NOT NULL DEFAULT '',
            embedding BLOB NOT NULL,
            output TEXT NOT NULL,
            output_meta_json TEXT NOT NULL DEFAULT '{}',
            written_at REAL NOT NULL,
            ttl_secs INTEGER NOT NULL DEFAULT 86400,
            hit_count INTEGER NOT NULL DEFAULT 0,
            last_hit_at REAL
        )
    """)
    # Forward-compat: existing DBs created before prompt_text column
    _cols = {r[1] for r in conn.execute("PRAGMA table_info(entries)").fetchall()}
    if "prompt_text" not in _cols:
        conn.execute("ALTER TABLE entries ADD COLUMN prompt_text TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_role_ts ON entries(role, written_at)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lookups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            ts REAL NOT NULL,
            hit INTEGER NOT NULL DEFAULT 0,
            similarity REAL,
            entry_id TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lookups_role_ts ON lookups(role, ts)")
    conn.commit()
    return conn


def _pack_embedding(vec: list[float]) -> bytes:
    """Serialize a float32 vector to bytes. struct format avoids
    pulling in numpy as a hard dep for the cache."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(b: bytes) -> list[float]:
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _new_id() -> str:
    return "sc-" + hashlib.sha256(
        f"{time.time()}-{id(object())}".encode()
    ).hexdigest()[:12]


# ── Embedding (Ollama) ───────────────────────────────────────────────


def embed_via_ollama(text: str, *, model: str = DEFAULT_EMBED_MODEL,
                     host: str = DEFAULT_OLLAMA_HOST,
                     timeout: float = 30.0) -> list[float]:
    """Get an embedding from Ollama's /api/embeddings endpoint.

    Raises on failure. Caller may pass embed_fn=... to override (tests
    use a deterministic stub).
    """
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        r.raise_for_status()
        data = r.json()
    vec = data.get("embedding") or []
    if not vec:
        raise RuntimeError(f"empty embedding from {model}")
    return [float(x) for x in vec]


# ── Similarity ───────────────────────────────────────────────────────


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# ── Anchor-term guard ────────────────────────────────────────────────


_ANCHOR_RE = None


def _anchor_terms(text: str) -> frozenset[str]:
    """Extract meaningful identifier-like tokens from a prompt.

    Catches the case where nomic-embed (or any weak embedder) produces
    near-identical vectors for prompts that differ only in a topic
    word. We require the SET of anchor terms in the candidate to match
    the SET in the request, regardless of cosine.

    What counts as an anchor:
      - CamelCase identifiers          (KVComm, LatentMAS, FalkorDB)
      - SCREAMING_SNAKE_CASE constants (OLLAMA_KEEP_ALIVE, P_VS_02)
      - dotted/dashed identifiers       (cycle-100, P-VS-02, x.y.z)
      - quoted strings                  ("foo bar")
      - numeric literals ≥ 3 digits     (cycle 100, 2026-05-13)

    Common English words, even capitalized at sentence start, are NOT
    anchors. The check is structural: tokens that look like names, ids,
    or quoted phrases — the things that change between "the same kind
    of question about X" and "the same kind of question about Y".
    """
    global _ANCHOR_RE
    if _ANCHOR_RE is None:
        import re
        # CamelCase, SCREAMING_CASE, hyphenated identifiers, dot paths,
        # quoted strings, multi-digit numbers
        _ANCHOR_RE = re.compile(
            r'"[^"]+"'                       # double-quoted
            r"|'[^']+'"                      # single-quoted
            r"|\b[A-Z][a-z]+[A-Z]\w*"        # CamelCase (≥2 humps)
            r"|\b[A-Z]{2,}[\w-]*"            # SCREAMING/UPPER + tail
            r"|\b[a-zA-Z]+[-_][a-zA-Z0-9_-]+"  # hyphenated / snake
            r"|\b\d{3,}"                     # multi-digit number
            r"|\b\w+\.\w+(?:\.\w+)*",        # dotted identifier
        )
    return frozenset(m.group(0).strip("\"'").lower()
                      for m in _ANCHOR_RE.finditer(text))


def anchors_match(request: str, candidate: str) -> bool:
    """True iff request and candidate share the same set of anchor
    terms. An EMPTY anchor set on both sides also counts as a match
    (both prompts are anchor-free generic prose)."""
    return _anchor_terms(request) == _anchor_terms(candidate)


# ── Cache API ────────────────────────────────────────────────────────


def _search(role: str, embedding: list[float], threshold: float,
            max_age_secs: float, request_prompt: str = "") -> CacheHit | None:
    """Return the best CacheHit above threshold AND satisfying the
    anchor-term guard, or None.

    Two-layer match:
      1. cosine similarity ≥ threshold  (semantic closeness)
      2. anchor_terms(request) == anchor_terms(candidate_prompt)
         (catches embedding-model collapse on topic-suffix patterns)
    """
    now = time.time()
    best: CacheHit | None = None
    best_sim = 0.0
    request_anchors = _anchor_terms(request_prompt) if request_prompt else None
    with _LOCK, _connect() as conn:
        # Linear scan over role's rows. Fine for cache sizes <10K;
        # swap in sqlite-vec when bert outgrows it.
        rows = conn.execute(
            "SELECT id, role, prompt_text, embedding, output, "
            "output_meta_json, written_at, hit_count FROM entries "
            "WHERE role=? AND (written_at + ttl_secs) > ?",
            (role, now),
        ).fetchall()
        for r in rows:
            cand_emb = _unpack_embedding(r[3])
            sim = cosine(embedding, cand_emb)
            if sim < threshold or sim <= best_sim:
                continue
            # Anchor guard: candidate's anchor terms must match request's
            if request_anchors is not None:
                cand_anchors = _anchor_terms(r[2] or "")
                if cand_anchors != request_anchors:
                    LOG.debug(
                        "semantic_cache: anchor mismatch sim=%.3f "
                        "req_anchors=%s cand_anchors=%s",
                        sim, request_anchors, cand_anchors,
                    )
                    continue
            best_sim = sim
            best = CacheHit(
                id=r[0], role=r[1], output=r[4],
                output_meta=json.loads(r[5] or "{}"),
                similarity=sim,
                age_secs=now - r[6],
                hit_count=r[7],
            )
    return best


def _record_lookup(role: str, *, hit: bool, similarity: float | None = None,
                   entry_id: str | None = None) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO lookups(role, ts, hit, similarity, entry_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (role, time.time(), 1 if hit else 0, similarity, entry_id),
        )
        if hit and entry_id:
            conn.execute(
                "UPDATE entries SET hit_count=hit_count+1, last_hit_at=? "
                "WHERE id=?",
                (time.time(), entry_id),
            )
        conn.commit()


def _store(role: str, prompt: str, embedding: list[float],
           output: str, output_meta: dict, ttl_secs: int) -> str:
    """Append a new cache entry; returns the entry id.

    Stores the prompt text (truncated to 4K chars) so the anchor-term
    guard can re-check it at retrieval time.
    """
    eid = _new_id()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO entries(id, role, prompt_hash, prompt_text, "
            "embedding, output, output_meta_json, written_at, ttl_secs, "
            "hit_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (eid, role, _hash(prompt), prompt[:4000],
             _pack_embedding(embedding), output, json.dumps(output_meta),
             time.time(), ttl_secs),
        )
        conn.commit()
    return eid


def get_or_compute(
    role: str,
    prompt: str,
    compute_fn: Callable[[], dict],
    *,
    cacheable_roles: frozenset[str] | set[str] | None = None,
    similarity_threshold: float = DEFAULT_THRESHOLD,
    ttl_secs: int = DEFAULT_TTL_SECS,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> tuple[dict, bool]:
    """Check the cache; if a similar prompt was answered recently for
    this role, return the cached response. Otherwise dispatch via
    compute_fn and cache the result.

    Returns (response_dict, was_hit). compute_fn must return a dict
    with at minimum {"text": str}; additional keys (tokens, latency,
    model) are stored as output_meta for telemetry.

    Roles not in cacheable_roles (default CACHEABLE_ROLES) bypass the
    cache entirely — get_or_compute always calls compute_fn for them.
    This is the load-bearing safety rail: never cache verdict roles.
    """
    allowed = cacheable_roles if cacheable_roles is not None else CACHEABLE_ROLES
    if role not in allowed:
        return compute_fn(), False

    embed = embed_fn or embed_via_ollama
    try:
        embedding = embed(prompt)
    except Exception as e:  # noqa: BLE001
        LOG.warning("semantic_cache: embed failed (%s); bypass", e)
        return compute_fn(), False

    hit = _search(role, embedding, similarity_threshold,
                  max_age_secs=ttl_secs * 2,  # double-ttl read window
                  request_prompt=prompt)
    if hit is not None:
        _record_lookup(role, hit=True, similarity=hit.similarity,
                       entry_id=hit.id)
        LOG.info("semantic_cache: HIT role=%s sim=%.3f age=%.0fs",
                 role, hit.similarity, hit.age_secs)
        # Output dict carries the cached fields + a hint flag
        response = dict(hit.output_meta or {})
        response["text"] = hit.output
        response["semantic_cache_hit"] = True
        response["semantic_cache_similarity"] = round(hit.similarity, 4)
        response["semantic_cache_age_secs"] = round(hit.age_secs, 1)
        return response, True

    # Miss — dispatch and cache
    _record_lookup(role, hit=False)
    response = compute_fn()
    output = response.get("text") if isinstance(response, dict) else None
    if not output:
        # Nothing to cache (provider error, empty response, etc.)
        return response, False
    meta = {k: v for k, v in (response or {}).items() if k != "text"}
    _store(role, prompt, embedding, output, meta, ttl_secs)
    return response, False


def cache_stats(role: str | None = None, *, window_secs: int = 86400) -> list[CacheStats]:
    """Per-role lookup statistics over the last `window_secs`."""
    now = time.time()
    cutoff = now - window_secs
    out: list[CacheStats] = []
    with _LOCK, _connect() as conn:
        if role:
            roles = [role]
        else:
            roles = [r[0] for r in conn.execute(
                "SELECT DISTINCT role FROM entries"
            ).fetchall()]
        for r in roles:
            (rows_count,) = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE role=?", (r,),
            ).fetchone()
            (hits,) = conn.execute(
                "SELECT COUNT(*) FROM lookups WHERE role=? AND ts > ? AND hit=1",
                (r, cutoff),
            ).fetchone()
            (misses,) = conn.execute(
                "SELECT COUNT(*) FROM lookups WHERE role=? AND ts > ? AND hit=0",
                (r, cutoff),
            ).fetchone()
            avg_row = conn.execute(
                "SELECT AVG(similarity) FROM lookups "
                "WHERE role=? AND ts > ? AND hit=1",
                (r, cutoff),
            ).fetchone()
            total = hits + misses
            out.append(CacheStats(
                role=r,
                rows=rows_count,
                hits_24h=hits,
                misses_24h=misses,
                hit_rate=round(hits / total, 3) if total else 0.0,
                avg_similarity_on_hit=round(avg_row[0] or 0.0, 4),
            ))
    return out


def prune_expired() -> int:
    """Delete rows past their TTL. Returns rows removed."""
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "DELETE FROM entries WHERE (written_at + ttl_secs) < ?",
            (time.time(),),
        )
        conn.commit()
    return cur.rowcount or 0


def clear(role: str | None = None) -> int:
    """Drop all entries (or all entries for one role). Returns rows
    removed. Mainly for tests and PI override."""
    with _LOCK, _connect() as conn:
        if role:
            cur = conn.execute("DELETE FROM entries WHERE role=?", (role,))
        else:
            cur = conn.execute("DELETE FROM entries")
            conn.execute("DELETE FROM lookups")
        conn.commit()
    return cur.rowcount or 0
