"""Smoke test for core/semantic_cache.py.

Tests the get_or_compute API surface, role-discipline safety rails,
similarity thresholding, and the cache_stats telemetry path. Uses a
deterministic stub embed function so we don't depend on a running
Ollama server.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import semantic_cache as sc  # noqa: E402


def _isolate() -> None:
    """Each test gets a fresh DB."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_sc_")) / "cache.db"
    sc.DB_PATH = tmp


def _stub_embed(text: str) -> list[float]:
    """Deterministic embedding stub.

    Maps the first 8 characters of the input to a fixed 16-dim vector
    so two prompts with identical first 8 chars get identical
    embeddings (similarity = 1.0). Two prompts with different first
    chars get orthogonal embeddings (similarity ~ 0.0).
    """
    head = (text + " " * 8)[:8]
    return [float(ord(c) % 7) / 7.0 for c in head] + [0.0] * 8


def test_cacheable_role_miss_then_hit() -> None:
    _isolate()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"text": f"answer-{calls['n']}", "tokens": 100}

    # Use a role in CACHEABLE_ROLES (post-calibration: threshing /
    # implementer only). Anchor-free prompts so the guard doesn't
    # fire between calls.
    # First call — miss + dispatch
    response_a, hit_a = sc.get_or_compute(
        "threshing", "should this go forward",
        compute, embed_fn=_stub_embed,
    )
    assert hit_a is False
    assert response_a["text"] == "answer-1"
    assert calls["n"] == 1

    # Second call — same first-8-chars + same (empty) anchor set → hit
    response_b, hit_b = sc.get_or_compute(
        "threshing", "should this go ahead now",
        compute, embed_fn=_stub_embed,
    )
    assert hit_b is True
    assert response_b["text"] == "answer-1"  # cached
    assert response_b["semantic_cache_hit"] is True
    assert response_b["semantic_cache_similarity"] >= 0.95
    assert calls["n"] == 1  # compute still only called once


def test_uncacheable_role_always_dispatches() -> None:
    """LOAD-BEARING: verdict + exploration roles must NEVER cache.
    Stale verdicts would corrupt the Quaker discernment discipline."""
    _isolate()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"text": f"verdict-{calls['n']}"}

    # Post-calibration CACHEABLE_ROLES excludes all of these:
    for role in ("evaluator", "clearness_phase2", "clearness_phase1",
                  "director", "researcher", "strategist"):
        a, hit_a = sc.get_or_compute(role, "x", compute, embed_fn=_stub_embed)
        b, hit_b = sc.get_or_compute(role, "x", compute, embed_fn=_stub_embed)
        assert hit_a is False, f"{role} unexpected hit"
        assert hit_b is False, f"{role} unexpected hit on repeat"
    assert calls["n"] == 12  # 6 roles × 2 calls each


def test_anchor_guard_blocks_topic_collision() -> None:
    """LOAD-BEARING: when embed model can't distinguish topic words,
    the anchor-term guard must prevent wrong-answer cache hits.

    Two prompts that ALMOST cosine-match (≥0.95) but differ in an
    anchor term (CamelCase identifier) must NOT collide."""
    _isolate()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"text": f"answer-{calls['n']}"}

    # Stub embed produces identical vectors when first 8 chars match.
    # The anchor guard should catch the LatentMAS vs KVComm difference.
    sc.get_or_compute(
        "threshing", "evaluate the KVComm Phase B target",
        compute, embed_fn=_stub_embed,
    )
    response, hit = sc.get_or_compute(
        "threshing", "evaluate the LatentMAS Phase B target",
        compute, embed_fn=_stub_embed,
    )
    # Cosine would say "hit" but anchor guard blocks it
    assert hit is False, "anchor guard failed to block topic collision"
    assert calls["n"] == 2  # both compute calls fired


def test_anchor_guard_allows_anchor_free_match() -> None:
    """Prompts with NO anchor terms (generic prose) should still match
    on cosine alone — the guard only kicks in when anchors exist."""
    _isolate()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"text": f"answer-{calls['n']}"}

    sc.get_or_compute("threshing", "should this proceed",
                       compute, embed_fn=_stub_embed)
    _, hit = sc.get_or_compute("threshing", "should this go on",
                                compute, embed_fn=_stub_embed)
    assert hit is True
    assert calls["n"] == 1


def test_anchor_terms_extraction() -> None:
    """Direct test of the anchor-term extractor."""
    assert sc._anchor_terms("plain prose with no anchors") == frozenset()
    assert "kvcomm" in sc._anchor_terms("evaluate KVComm Phase B")
    assert "latentmas" in sc._anchor_terms("LatentMAS comparison")
    assert "p-vs-02" in sc._anchor_terms("apply P-VS-02 cross-family review")
    assert "ollama_keep_alive" in sc._anchor_terms("OLLAMA_KEEP_ALIVE=24h")
    assert "100" in sc._anchor_terms("cycle 100 results")
    assert "100" not in sc._anchor_terms("section 12")  # 2 digits only


def test_dissimilar_prompts_miss() -> None:
    _isolate()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"text": f"answer-{calls['n']}"}

    sc.get_or_compute("threshing", "first dispatch",
                       compute, embed_fn=_stub_embed)
    # Different first char → different embedding → miss
    sc.get_or_compute("threshing", "zXY8 second dispatch",
                       compute, embed_fn=_stub_embed)
    assert calls["n"] == 2  # both missed


def test_threshold_controls_hit_strictness() -> None:
    _isolate()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"text": f"answer-{calls['n']}"}

    sc.get_or_compute(
        "threshing", "the quick brown fox",
        compute, embed_fn=_stub_embed,
    )
    # Same first 8 chars → similarity ≈ 1.0 → hit even at threshold 0.99
    _, hit = sc.get_or_compute(
        "threshing", "the quick brown owl",
        compute, embed_fn=_stub_embed,
        similarity_threshold=0.99,
    )
    assert hit is True
    assert calls["n"] == 1


def test_embed_failure_bypasses_cache() -> None:
    """If the embedding service is down, dispatch must still work."""
    _isolate()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"text": "ok"}

    def failing_embed(text: str) -> list[float]:
        raise RuntimeError("ollama unreachable")

    response, hit = sc.get_or_compute(
        "threshing", "x", compute, embed_fn=failing_embed,
    )
    assert hit is False
    assert response["text"] == "ok"
    assert calls["n"] == 1


def test_empty_response_not_cached() -> None:
    """compute_fn returning empty text should NOT pollute the cache."""
    _isolate()

    def compute():
        return {"text": "", "error": "provider error"}

    a, hit_a = sc.get_or_compute(
        "threshing", "test", compute, embed_fn=_stub_embed,
    )
    assert hit_a is False
    # Second call also misses because nothing was cached
    b, hit_b = sc.get_or_compute(
        "threshing", "test", compute, embed_fn=_stub_embed,
    )
    assert hit_b is False


def test_cache_stats_records_hits_and_misses() -> None:
    _isolate()

    def compute():
        return {"text": "x"}

    # 1 miss, then 3 hits
    sc.get_or_compute("threshing", "aaaaaaaa qry",
                       compute, embed_fn=_stub_embed)
    for _ in range(3):
        sc.get_or_compute("threshing", "aaaaaaaa qry",
                           compute, embed_fn=_stub_embed)

    stats = sc.cache_stats("threshing")
    assert len(stats) == 1
    s = stats[0]
    assert s.role == "threshing"
    assert s.rows == 1
    assert s.hits_24h == 3
    assert s.misses_24h == 1
    assert s.hit_rate == 0.75
    assert s.avg_similarity_on_hit >= 0.95


def test_clear_drops_rows() -> None:
    _isolate()

    def compute():
        return {"text": "x"}

    sc.get_or_compute("threshing", "a", compute, embed_fn=_stub_embed)
    sc.get_or_compute("implementer", "b", compute, embed_fn=_stub_embed)
    # Clear one role only
    removed = sc.clear("threshing")
    assert removed == 1
    stats = sc.cache_stats()
    assert {s.role: s.rows for s in stats}.get("implementer") == 1


def test_prune_expired() -> None:
    _isolate()

    def compute():
        return {"text": "x"}

    # TTL=1s; sleep + prune
    import time
    sc.get_or_compute("threshing", "soon-expired", compute,
                       embed_fn=_stub_embed, ttl_secs=1)
    time.sleep(1.5)
    removed = sc.prune_expired()
    assert removed >= 1


def test_cosine_basics() -> None:
    assert abs(sc.cosine([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-6
    assert abs(sc.cosine([1, 0, 0], [0, 1, 0]) - 0.0) < 1e-6
    assert sc.cosine([], []) == 0.0
    assert sc.cosine([1], [0]) == 0.0


def main() -> int:
    tests = [
        test_cacheable_role_miss_then_hit,
        test_uncacheable_role_always_dispatches,
        test_anchor_guard_blocks_topic_collision,
        test_anchor_guard_allows_anchor_free_match,
        test_anchor_terms_extraction,
        test_dissimilar_prompts_miss,
        test_threshold_controls_hit_strictness,
        test_embed_failure_bypasses_cache,
        test_empty_response_not_cached,
        test_cache_stats_records_hits_and_misses,
        test_clear_drops_rows,
        test_prune_expired,
        test_cosine_basics,
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
