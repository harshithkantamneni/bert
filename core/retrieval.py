"""Hybrid retrieval merger — vector + KG + semantic cache.

Per Zep Graphiti's 63.8% vs Mem0 49.0% on LongMemEval (Mem0 State of
Agent Memory 2026): 15pp uplift from hybrid vs vector-only. The
ingredient is **Reciprocal Rank Fusion** (RRF) combining rank lists
from heterogeneous sources, optionally followed by a stronger
reranker on the top-K.

Bert's three retrieval signal sources:

  vector    core.memory.search (sqlite-vec; BGE-M3 / Qwen3-Embed)
            best at semantic similarity
  graph     core.graph_store.subgraph (KG 1-2 hop from seeds)
            best at relational + temporal queries (post-H.3 validity
            windows make this stronger)
  cache     core.semantic_cache hits — re-using prior answers when
            the question is near-duplicate

RRF (Cormack & Buettcher 2009): for each item appearing in any rank
list, score = sum over (1 / (k + rank_in_list_i)); k=60 is the
production default. Items appearing in multiple lists get boosted
without needing per-source weight tuning.

Optional reranker (ColBERT v2 shape): pluggable rerank_fn callback
applied to the top-K RRF candidates. Since bert ships nomic-embed-
text (single-vector, no late interaction), the default "rerank" is
a simple cosine re-scoring against the query embedding using the
same model — a meaningful approximation that catches false-positive
hits the bag-of-RRF couldn't reorder. Real ColBERT v2 lands when
PI installs it.

Returns a unified RetrievalResult list with source attribution so
the caller (and the canvas Diagnostics surface) can see which signal
fired for each hit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger("bert.retrieval")

LAB_ROOT = Path(__file__).resolve().parent.parent

# RRF constant (Cormack & Buettcher 2009 default).
DEFAULT_RRF_K = 60


@dataclass
class RetrievalCandidate:
    """One hit from a single signal source."""
    id: str
    text: str                          # short excerpt or label
    source: str                        # "vector" / "graph" / "cache"
    rank: int                          # 0-indexed rank in source list
    score: float                       # source-native score
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Merged hit — appears in 1+ source lists."""
    id: str
    text: str
    rrf_score: float
    sources: list[str] = field(default_factory=list)
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """Use rerank when present, else RRF."""
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


# ── RRF merger ───────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    *candidate_lists: list[RetrievalCandidate],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """Cormack-Buettcher RRF. Each list contributes (1 / (k + rank))
    to each item's score. Items in multiple lists sum their
    contributions.

    Returns the merged list sorted by rrf_score descending.
    """
    merged: dict[str, RetrievalResult] = {}
    for cands in candidate_lists:
        for c in cands:
            contribution = 1.0 / (k + c.rank)
            if c.id in merged:
                merged[c.id].rrf_score += contribution
                if c.source not in merged[c.id].sources:
                    merged[c.id].sources.append(c.source)
            else:
                merged[c.id] = RetrievalResult(
                    id=c.id, text=c.text,
                    rrf_score=contribution,
                    sources=[c.source],
                    metadata=dict(c.metadata),
                )
    return sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)


# ── Source adapters ──────────────────────────────────────────────────


def _vector_candidates(query: str, k: int = 20) -> list[RetrievalCandidate]:
    """Pull from core.memory.search (sqlite-vec). Gracefully empty
    when memory/sqlite_vec aren't available."""
    try:
        from core import memory
        rows = memory.search(query, k=k)
    except Exception as e:  # noqa: BLE001
        LOG.debug("retrieval: vector source unavailable (%s)", e)
        return []
    out: list[RetrievalCandidate] = []
    for i, r in enumerate(rows[:k]):
        # core.memory.search returns {path, chunk_idx, content, distance}.
        # (Older callers used {id|key, text, score}; accept both.) Reading the
        # wrong keys here silently zeroed the vector signal in hybrid fusion —
        # the candidate got empty text + an index id, so the reranker saw
        # nothing and recall cratered vs vector-only. Carry the FULL chunk
        # content (the reranker + reader need it; 240-char excerpts dropped the
        # answer span on real long chunks).
        content = str(r.get("content") or r.get("text") or "")
        rid = str(r.get("id") or r.get("key")
                  or (f"{r.get('path')}:{r.get('chunk_idx')}"
                      if r.get("path") is not None else i))
        if "score" in r:
            score = float(r["score"])
        elif "distance" in r:                  # lower distance = better -> similarity
            score = 1.0 / (1.0 + float(r["distance"]))
        else:
            score = 0.0
        out.append(RetrievalCandidate(
            id=f"vec:{rid}",
            text=content,
            source="vector", rank=i,
            score=score,
            metadata={"vector_id": rid, "content": content,
                      **{k: v for k, v in r.items()
                         if k not in ("id", "key", "text", "content", "score", "distance")}},
        ))
    return out


def _graph_candidates(seed_ids: list[str], hops: int = 2,
                      at: float | None = None) -> list[RetrievalCandidate]:
    """Pull from core.graph_store.subgraph 1-2 hop neighbors of seeds."""
    try:
        from core import graph_store
        nodes, _edges = graph_store.subgraph(seed_ids, hops=hops, at=at)
    except Exception as e:  # noqa: BLE001
        LOG.debug("retrieval: graph source unavailable (%s)", e)
        return []
    # Rank: BFS distance from seeds (approximate — fold edges into
    # depth count later). For now, all neighbors get rank 0..N by
    # iteration order which BFS already enforces.
    seed_set = set(seed_ids)
    out: list[RetrievalCandidate] = []
    rank = 0
    for n in nodes:
        if n.id in seed_set:
            continue
        out.append(RetrievalCandidate(
            id=f"kg:{n.id}",
            text=f"{n.type}: {n.label}",
            source="graph", rank=rank,
            score=1.0 / (1 + rank),
            metadata={"node_id": n.id, "node_type": n.type, **n.props},
        ))
        rank += 1
    return out


# ── A5/B2 — BM25 + token-graph PPR signal sources ─────────────────────


def _bm25_candidates(query: str, k: int = 20,
                     lab_path: Path | None = None
                     ) -> list[RetrievalCandidate]:
    """Pull from core.bm25 sparse retrieval (rank_bm25)."""
    try:
        from core import bm25 as _bm25
        hits = _bm25.search(query, lab_path=lab_path, k=k)
    except Exception as e:  # noqa: BLE001
        LOG.debug("retrieval: BM25 source unavailable (%s)", e)
        return []
    out: list[RetrievalCandidate] = []
    for i, h in enumerate(hits):
        content = h.content or ""
        out.append(RetrievalCandidate(
            id=f"bm25:{h.chunk_id}",
            text=content,                      # full chunk (was [:240], dropped the answer span)
            source="bm25", rank=i,
            score=float(h.score),
            metadata={"chunk_id": h.chunk_id, "content": content, **(h.metadata or {})},
        ))
    return out


def _ppr_candidates(query: str, lab_path: Path | None = None,
                    k: int = 20) -> list[RetrievalCandidate]:
    """Pull from core.token_graph PPR. Only fires if query contains
    canonical tokens (otherwise returns empty so RRF ignores it)."""
    try:
        from pathlib import Path as _P

        from core import token_graph as _tg
        lp = lab_path or _P("/")
        hits = _tg.search(query, lab_path=lp, k=k)
    except Exception as e:  # noqa: BLE001
        LOG.debug("retrieval: PPR source unavailable (%s)", e)
        return []
    out: list[RetrievalCandidate] = []
    for i, h in enumerate(hits):
        out.append(RetrievalCandidate(
            id=f"ppr:{h.chunk_id}",
            text="",  # token graph doesn't store chunk text; vector path
                      # surfaces the same chunk_id with content
            source="ppr", rank=i,
            score=float(h.score),
            metadata={"chunk_id": h.chunk_id, **(h.metadata or {})},
        ))
    return out


def _cache_candidates(query: str, k: int = 5,
                      cacheable_roles: set[str] | None = None
                      ) -> list[RetrievalCandidate]:
    """Pull from core.semantic_cache — answers bert previously gave to
    near-duplicate questions, surfaced as candidates so the merger can
    weight them alongside fresh retrieval."""
    try:
        from core import semantic_cache
    except Exception as e:  # noqa: BLE001
        LOG.debug("retrieval: cache source unavailable (%s)", e)
        return []
    # Use the role-agnostic stats() for telemetry; for retrieval we
    # need raw entries. Reach into the DB directly (single-process,
    # local-only).
    try:
        import sqlite3
        conn = sqlite3.connect(semantic_cache.DB_PATH, timeout=2.0)
        roles = cacheable_roles or semantic_cache.CACHEABLE_ROLES
        placeholders = ",".join("?" * len(roles))
        rows = conn.execute(
            f"SELECT id, role, prompt_text, output FROM entries "
            f"WHERE role IN ({placeholders}) "
            f"ORDER BY written_at DESC LIMIT ?",
            (*roles, k),
        ).fetchall()
        conn.close()
    except Exception as e:  # noqa: BLE001
        LOG.debug("retrieval: cache db read failed (%s)", e)
        return []
    out: list[RetrievalCandidate] = []
    for i, r in enumerate(rows):
        out.append(RetrievalCandidate(
            id=f"cache:{r[0]}",
            text=str(r[3] or "")[:240],
            source="cache", rank=i,
            score=1.0 / (1 + i),
            metadata={"cache_id": r[0], "role": r[1],
                       "prompt_excerpt": str(r[2] or "")[:120]},
        ))
    return out


# ── Public hybrid_retrieve ───────────────────────────────────────────


def hybrid_retrieve(
    query: str,
    *,
    seed_ids: list[str] | None = None,
    k_per_source: int = 20,
    rrf_k: int = DEFAULT_RRF_K,
    top_n: int = 10,
    hops: int = 2,
    at: float | None = None,
    rerank_fn: Callable[[str, list[RetrievalResult]], list[RetrievalResult]] | None = None,
) -> list[RetrievalResult]:
    """One-call entry point.

    Pulls from vector + BM25 + graph (if seed_ids given); merges via
    RRF; optionally reranks the top top_n with rerank_fn; returns the
    top_n list.

    Note: PPR + semantic-cache candidates were removed from the fusion
    after empirical evidence (B2 on BEIR scifact, plus runtime probes)
    showed PPR never fired on arbitrary corpora and the cache signal
    ordered by recency rather than relevance. The semantic-cache
    *anchor-term guard* (USP #1) remains live in core/semantic_cache.py
    for LLM-call dedup — only the broken retrieval-side wrapper is gone.
    PPR continues to be available standalone via core.token_graph for
    bert's own canonical-token labs but is no longer fused by default.
    """
    # Resolve the active lab so BM25 reads the right index.
    try:
        from core import lab_context
        lab_path = lab_context.get_active_lab_path() or (LAB_ROOT / "lab")
    except Exception:  # noqa: BLE001
        lab_path = LAB_ROOT / "lab"

    # Per-stage timing for retrieval.jsonl instrumentation.
    import time as _t
    t0 = _t.perf_counter()
    timings: dict[str, float] = {}

    vec_cands = _vector_candidates(query, k=k_per_source)
    timings["vector_ms"] = (_t.perf_counter() - t0) * 1000

    t1 = _t.perf_counter()
    graph_cands = (
        _graph_candidates(seed_ids, hops=hops, at=at)
        if seed_ids else []
    )
    timings["graph_ms"] = (_t.perf_counter() - t1) * 1000

    t2 = _t.perf_counter()
    bm25_cands = _bm25_candidates(query, k=k_per_source, lab_path=lab_path)
    timings["bm25_ms"] = (_t.perf_counter() - t2) * 1000

    t3 = _t.perf_counter()
    merged = reciprocal_rank_fusion(
        vec_cands, graph_cands, bm25_cands,
        k=rrf_k,
    )
    timings["rrf_ms"] = (_t.perf_counter() - t3) * 1000

    top = merged[: max(top_n, 1) * 5]  # rerank pool = 5× final size
    # B3 — If no rerank_fn passed, try the real cross-encoder; on
    # failure, fall back to default cosine reranker. Both are opt-in
    # in the sense that hybrid_retrieve still works with rerank_fn=None.
    rerank_used = False
    if rerank_fn is None:
        try:
            from core import reranker as _rr
            rerank_fn = _rr.get_cross_encoder_rerank_fn()
        except Exception as e:  # noqa: BLE001
            LOG.debug("retrieval: cross-encoder lookup failed (%s)", e)
            rerank_fn = None
    t4 = _t.perf_counter()
    if rerank_fn is not None and top:
        try:
            top = rerank_fn(query, top)
            rerank_used = True
        except Exception as e:  # noqa: BLE001
            LOG.warning("retrieval: rerank_fn raised (%s); using RRF order", e)
    timings["rerank_ms"] = (_t.perf_counter() - t4) * 1000

    # Sort by final_score (rerank if present, else RRF) and trim
    top.sort(key=lambda r: r.final_score, reverse=True)
    final = top[:top_n]

    timings["total_ms"] = (_t.perf_counter() - t0) * 1000

    # Instrumentation — best-effort emit; never blocks retrieval.
    try:
        _emit_retrieval_event(
            query=query, seed_ids=seed_ids, top_n=top_n,
            vec_cands=vec_cands, bm25_cands=bm25_cands,
            graph_cands=graph_cands, merged=merged, final=final,
            rerank_used=rerank_used, timings=timings,
            lab_path=lab_path,
        )
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger("bert.retrieval").warning(
            "retrieval emit FAILED query=%r exc=%s: %s",
            (query or "")[:80], type(e).__name__, e,
        )

    return final


def _emit_retrieval_event(
    *, query: str, seed_ids: list[str] | None, top_n: int,
    vec_cands: list, bm25_cands: list, graph_cands: list,
    merged: list, final: list,
    rerank_used: bool, timings: dict[str, float],
    lab_path: Path,
) -> None:
    """Emit a retrieval-call event to state/observability/retrieval.jsonl.

    Captures the full per-call shape that v3+ Phase 1 instrumentation
    needs to make subsequent architectural decisions empirically:
    query payload (truncated), top-K result ids + scores, which signals
    fired and contributed, per-stage latency. Filed under its own event
    class so it doesn't pollute tool_call.jsonl."""
    from core import observability as _obs
    # Helper to get top-N candidates from a per-source list for inspection
    def _top_summary(cands: list, n: int = 5) -> list[dict]:
        return [
            {"id": c.id, "rank": c.rank, "score": float(c.score) if c.score is not None else None}
            for c in cands[:n]
        ]
    # Query text truncation — keep first 500 chars to bound log size
    q_text = (query or "")[:500]
    payload = {
        "query": q_text,
        "query_len": len(query or ""),
        "lab": lab_path.name if lab_path else None,
        "seed_ids": seed_ids or [],
        "top_n_requested": top_n,
        "final_top_k": [
            {
                "id": r.id,
                "score": float(r.final_score) if r.final_score is not None else None,
                "sources": list(r.sources or []),
                "rerank_score": float(r.rerank_score) if r.rerank_score is not None else None,
            }
            for r in final
        ],
        "signal_summary": {
            "vector": {"n": len(vec_cands), "top": _top_summary(vec_cands)},
            "bm25":   {"n": len(bm25_cands), "top": _top_summary(bm25_cands)},
            "graph":  {"n": len(graph_cands), "top": _top_summary(graph_cands)},
            "merged_pool": len(merged),
        },
        "rerank_used": rerank_used,
        "timings_ms": timings,
    }
    _obs.emit("retrieval", payload)


# ── Default reranker (cosine on the query embedding) ─────────────────


def default_cosine_reranker(query: str, candidates: list[RetrievalResult]
                              ) -> list[RetrievalResult]:
    """Approximate ColBERT-style rerank — re-scores top-K against the
    query embedding using the same single-vector model bert uses for
    the semantic cache (nomic-embed-text). Real ColBERT v2 lands as a
    drop-in replacement when PI installs it.
    """
    try:
        from core import semantic_cache
        query_vec = semantic_cache.embed_via_ollama(query)
    except Exception as e:  # noqa: BLE001
        LOG.debug("retrieval: rerank skipped — embed unavailable (%s)", e)
        return candidates
    for c in candidates:
        try:
            cand_vec = semantic_cache.embed_via_ollama(c.text or c.id)
            c.rerank_score = semantic_cache.cosine(query_vec, cand_vec)
        except Exception:  # noqa: BLE001
            c.rerank_score = 0.0
    return candidates
