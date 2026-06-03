"""TDD for benchmarks/b9_rag_runner.run_one_query — the per-(question,arm) logic
that builds context, calls the reader, grades vs gold, and scores retrieval. All
deps (retrieve/reader/grader) are injected so it runs offline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks import b9_rag_runner as RR  # noqa: E402

CORPUS = [{"path": "x.py", "content": "ALPHA " * 500},
          {"path": "y.py", "content": "OMEGA needle lives here " * 50}]
# retrieved = list of (chunk_id, content); the gold chunk is c_y0.
RETRIEVED = [("c_y0", "OMEGA needle lives here"), ("c_x3", "ALPHA noise")]


def test_rag_arm_scores_retrieval_and_grades():
    row = RR.run_one_query(
        "where does the needle live?", gold_answer="in y.py",
        gold_chunk_ids=["c_y0"], arm="A3", corpus_files=CORPUS,
        retrieved=RETRIEVED, budget_tokens=20,
        reader=lambda prompt: "The needle lives in y.py.",
        grader=lambda q, gold, ans: 1, k=10)
    assert row["arm"] == "A3" and row["correct"] == 1
    assert row["recall_at_10"] == 1.0          # gold chunk retrieved at rank 1
    assert row["ndcg_at_10"] == 1.0
    # RAG context = retrieved chunks only -> small input
    assert row["input_tokens"] < 50


def test_non_rag_arm_has_no_retrieval_metrics_and_bigger_input():
    row = RR.run_one_query(
        "where does the needle live?", gold_answer="in y.py",
        gold_chunk_ids=["c_y0"], arm="A0", corpus_files=CORPUS,
        retrieved=[], budget_tokens=20,
        reader=lambda prompt: "I don't see it.",
        grader=lambda q, gold, ans: 0, k=10)
    assert row["arm"] == "A0" and row["correct"] == 0
    assert row["recall_at_10"] is None and row["ndcg_at_10"] is None
    assert row["input_tokens"] > 100           # full corpus stuffed


def test_reader_sees_only_arm_context():
    seen = {}

    def capture_reader(prompt):
        seen["prompt"] = prompt
        return "ok"

    RR.run_one_query("q", "gold", ["c_y0"], arm="A3", corpus_files=CORPUS,
                     retrieved=RETRIEVED, budget_tokens=20,
                     reader=capture_reader, grader=lambda *a: 1, k=10)
    # A3 reader must see the retrieved needle but NOT the bulk ALPHA corpus.
    assert "OMEGA needle" in seen["prompt"]
    assert "ALPHA ALPHA ALPHA" not in seen["prompt"]


def main() -> int:
    tests = [test_rag_arm_scores_retrieval_and_grades,
             test_non_rag_arm_has_no_retrieval_metrics_and_bigger_input,
             test_reader_sees_only_arm_context]
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
