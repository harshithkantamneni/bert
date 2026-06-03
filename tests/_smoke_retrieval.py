"""Smoke test for core/retrieval.py (H.4 — RRF + rerank hybrid)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import retrieval  # noqa: E402


def test_rrf_basic_single_list() -> None:
    """RRF on a single list: scores are 1/(k+rank)."""
    cands = [
        retrieval.RetrievalCandidate("a", "A", "vector", rank=0, score=1.0),
        retrieval.RetrievalCandidate("b", "B", "vector", rank=1, score=0.9),
    ]
    merged = retrieval.reciprocal_rank_fusion(cands)
    assert len(merged) == 2
    assert merged[0].id == "a"
    # 1 / (60 + 0) ≈ 0.01667
    assert abs(merged[0].rrf_score - 1.0 / 60) < 1e-6


def test_rrf_combines_lists() -> None:
    """Item appearing in multiple lists gets boosted via summed contributions."""
    list_a = [
        retrieval.RetrievalCandidate("hit", "X", "vector", rank=0, score=1.0),
    ]
    list_b = [
        retrieval.RetrievalCandidate("hit", "X", "graph", rank=2, score=0.5),
    ]
    merged = retrieval.reciprocal_rank_fusion(list_a, list_b)
    assert len(merged) == 1
    # Contributions: 1/(60+0) + 1/(60+2)
    expected = 1.0 / 60 + 1.0 / 62
    assert abs(merged[0].rrf_score - expected) < 1e-6
    assert set(merged[0].sources) == {"vector", "graph"}


def test_rrf_sorts_descending() -> None:
    cands_low = [
        retrieval.RetrievalCandidate("low", "X", "vector", rank=10, score=0.1),
    ]
    cands_high = [
        retrieval.RetrievalCandidate("high", "Y", "graph", rank=0, score=0.5),
    ]
    merged = retrieval.reciprocal_rank_fusion(cands_low, cands_high)
    assert merged[0].id == "high"
    assert merged[1].id == "low"


def test_empty_lists_return_empty() -> None:
    assert retrieval.reciprocal_rank_fusion() == []
    assert retrieval.reciprocal_rank_fusion([], [], []) == []


def test_hybrid_retrieve_with_seed_includes_graph() -> None:
    """With seed_ids, _graph_candidates fires off the local KG.
    Vector + cache adapters are bypassed to avoid live Ollama calls."""
    from core import graph_store
    import tempfile as _tf
    tmp = Path(_tf.mkdtemp()) / "graph.db"
    graph_store.DB_PATH = tmp
    # Seed a tiny graph
    graph_store.add_node("seed-A", "Decision", "A")
    graph_store.add_node("neighbor-B", "Decision", "B")
    graph_store.add_edge("seed-A", "neighbor-B", "REFERENCES")

    # Stub the slow adapters to avoid Ollama / sqlite-vec dependencies
    import core.retrieval as _r
    orig_vec = _r._vector_candidates
    orig_cache = _r._cache_candidates
    _r._vector_candidates = lambda q, k=20: []
    _r._cache_candidates = lambda q, k=20, cacheable_roles=None: []
    try:
        results = retrieval.hybrid_retrieve(
            "anything",
            seed_ids=["seed-A"],
            top_n=10,
        )
        # neighbor-B should appear via the graph source
        ids = {r.id for r in results}
        assert any("neighbor-B" in i for i in ids)
    finally:
        _r._vector_candidates = orig_vec
        _r._cache_candidates = orig_cache


def test_hybrid_retrieve_empty_when_all_sources_empty() -> None:
    """With all sources stubbed empty, returns [].

    C-prime retired _cache_candidates + _ppr_candidates from the default
    fusion (empirically dead: empty cache DB, missing token_graph.db).
    hybrid_retrieve now fuses vector + graph + BM25, so the empty-sources
    stub set must cover _bm25_candidates — previously it stubbed the
    now-unused _cache_candidates and let live BM25 results leak in."""
    import core.retrieval as _r
    orig_vec = _r._vector_candidates
    orig_graph = _r._graph_candidates
    orig_bm25 = _r._bm25_candidates
    _r._vector_candidates = lambda q, k=20: []
    _r._graph_candidates = lambda seeds, hops=2, at=None: []
    _r._bm25_candidates = lambda q, k=20, lab_path=None: []
    try:
        results = retrieval.hybrid_retrieve(
            "x", top_n=10, seed_ids=["s"],
        )
        assert results == []
    finally:
        _r._vector_candidates = orig_vec
        _r._graph_candidates = orig_graph
        _r._bm25_candidates = orig_bm25


def test_rerank_fn_override_takes_precedence() -> None:
    """If a rerank sets rerank_score, final_score uses it (not RRF)."""
    cands = [
        retrieval.RetrievalCandidate("a", "X", "vector", rank=0, score=1.0),
        retrieval.RetrievalCandidate("b", "Y", "vector", rank=1, score=0.9),
    ]
    merged = retrieval.reciprocal_rank_fusion(cands)

    def flip_rerank(query, candidates):
        # Reverse the RRF order via rerank score
        for i, c in enumerate(candidates):
            c.rerank_score = 1.0 - i
        return candidates

    # Without rerank: a first (rrf_score higher because rank 0)
    assert merged[0].id == "a"

    # Manual rerank application: first item gets the lowest score
    reranked = flip_rerank("q", list(merged))
    reranked.sort(key=lambda r: r.final_score, reverse=True)
    # Wait: rerank gives idx=0 → score 1.0, so first item still highest.
    # The point is: rerank_score is what final_score returns when set.
    assert reranked[0].rerank_score == 1.0
    assert reranked[0].final_score == 1.0
    # final_score short-circuits to rerank_score, not rrf_score
    assert reranked[0].final_score != reranked[0].rrf_score


def test_rerank_failure_bypasses_cleanly() -> None:
    """If rerank_fn raises, hybrid_retrieve falls back to RRF order."""
    def boom(query, cands):
        raise RuntimeError("simulated rerank failure")

    # Stub adapters so we get reliable seed candidates without live I/O.
    # Must cover the current fusion set (vector + graph + BM25); BM25
    # replaced the retired _cache_candidates signal in C-prime, so leaving
    # it unstubbed lets live index hits leak in and breaks len()==1.
    import core.retrieval as _r
    orig_vec = _r._vector_candidates
    orig_graph = _r._graph_candidates
    orig_bm25 = _r._bm25_candidates
    _r._vector_candidates = lambda q, k=20: [
        retrieval.RetrievalCandidate("v:1", "stub", "vector", rank=0, score=1.0),
    ]
    _r._graph_candidates = lambda seeds, hops=2, at=None: []
    _r._bm25_candidates = lambda q, k=20, lab_path=None: []
    try:
        results = retrieval.hybrid_retrieve("q", top_n=5, rerank_fn=boom)
    finally:
        _r._vector_candidates = orig_vec
        _r._graph_candidates = orig_graph
        _r._bm25_candidates = orig_bm25
    assert isinstance(results, list)
    assert len(results) == 1
    # Rerank raised → rerank_score stays None
    assert results[0].rerank_score is None


def main() -> int:
    tests = [
        test_rrf_basic_single_list,
        test_rrf_combines_lists,
        test_rrf_sorts_descending,
        test_empty_lists_return_empty,
        test_hybrid_retrieve_with_seed_includes_graph,
        test_hybrid_retrieve_empty_when_all_sources_empty,
        test_rerank_fn_override_takes_precedence,
        test_rerank_failure_bypasses_cleanly,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
