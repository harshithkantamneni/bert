"""TDD for benchmarks/b9_rag_stats.py — retrieval-quality metrics for the
long-context RAG benchmark. These score the RETRIEVER independently of the
reader (recall@k / nDCG@k vs gold chunk ids) so we can tell "retriever missed
the chunk" from "reader had the chunk but answered wrong". Hand-checkable."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks import b9_rag_stats as S  # noqa: E402


def test_recall_at_k():
    # retrieved top-3 = [g1, x, g2]; gold = {g1, g2} -> recall = 2/2 = 1.0
    assert S.recall_at_k(["g1", "x", "g2", "y"], ["g1", "g2"], k=3) == 1.0
    # only g1 in top-2 -> 1/2
    assert S.recall_at_k(["g1", "x", "g2"], ["g1", "g2"], k=2) == 0.5
    # empty gold -> 0.0 (avoid div by zero), not a crash
    assert S.recall_at_k(["a"], [], k=3) == 0.0


def test_hit_rate_at_k():
    assert S.hit_rate_at_k(["x", "g1", "y"], ["g1", "g2"], k=3) == 1.0
    assert S.hit_rate_at_k(["x", "y", "z"], ["g1"], k=3) == 0.0
    assert S.hit_rate_at_k(["x", "g1"], ["g1"], k=1) == 0.0   # g1 at rank 2, k=1


def test_ndcg_at_k_hand_value():
    # retrieved=[g1, x, g2], gold={g1,g2}, k=3
    # DCG  = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)     = 1.0 + 0.63092975 = 1.63092975
    nd = S.ndcg_at_k(["g1", "x", "g2"], ["g1", "g2"], k=3)
    assert abs(nd - (1.5 / (1.0 + 1.0 / math.log2(3)))) < 1e-9
    # perfect ranking -> 1.0
    assert abs(S.ndcg_at_k(["g1", "g2", "x"], ["g1", "g2"], k=3) - 1.0) < 1e-9
    # nothing relevant -> 0.0
    assert S.ndcg_at_k(["x", "y"], ["g1"], k=2) == 0.0


def test_recall_spans_content_match():
    # Gold is verbatim substrings; recall = fraction found in the top-k texts.
    texts = ["the AsyncConnectionPool is built here", "unrelated chunk"]
    assert S.recall_spans(texts, ["AsyncConnectionPool"], k=2) == 1.0
    assert S.recall_spans(texts, ["AsyncConnectionPool", "TimeoutTypes"], k=2) == 0.5
    assert S.hit_spans(texts, ["AsyncConnectionPool"], k=2) == 1.0
    assert S.hit_spans(texts, ["NOPE"], k=2) == 0.0
    # k limits how many retrieved texts are searched.
    assert S.recall_spans(["a", "needleX"], ["needleX"], k=1) == 0.0
    assert S.recall_spans([], ["x"], k=3) == 0.0


def test_corpus_token_estimate():
    # ~4 chars/token heuristic; used for the cost-vs-corpus-size curve.
    assert S.est_tokens("a" * 400) == 100
    assert S.est_tokens("") == 0


def test_per_arm_aggregate():
    rows = [
        {"arm": "A3", "tier": "needle", "correct": 1, "input_tokens": 8000,
         "recall_at_10": 1.0, "ndcg_at_10": 0.9},
        {"arm": "A3", "tier": "needle", "correct": 0, "input_tokens": 9000,
         "recall_at_10": 0.0, "ndcg_at_10": 0.0},
        {"arm": "A1", "tier": "needle", "correct": 0, "input_tokens": 120000,
         "recall_at_10": None, "ndcg_at_10": None},
    ]
    agg = S.aggregate_by_arm(rows)
    assert agg["A3"]["n"] == 2
    assert abs(agg["A3"]["accuracy"] - 0.5) < 1e-9
    assert agg["A3"]["mean_input_tokens"] == 8500
    assert abs(agg["A3"]["mean_recall_at_10"] - 0.5) < 1e-9   # Nones excluded
    assert agg["A1"]["accuracy"] == 0.0
    assert agg["A1"]["mean_recall_at_10"] is None             # retrieval n/a for non-RAG


def main() -> int:
    tests = [test_recall_at_k, test_hit_rate_at_k, test_ndcg_at_k_hand_value,
             test_corpus_token_estimate, test_per_arm_aggregate]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
