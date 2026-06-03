"""TDD for benchmarks/b9_rag.py — the RAG benchmark arm context-builders. Pure
(no corpus indexing / no model calls): given corpus files + retrieved chunks +
a token budget, each arm assembles exactly the context its definition allows.
This is where the fairness lives — A1/A2 are steelman truncations, A3 sees only
retrieved chunks. See benchmarks/B8 plan WS1c."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks import b9_rag as R  # noqa: E402

FILES = [
    {"path": "a.py", "content": "ALPHA " * 200},     # ~1000 chars
    {"path": "b.py", "content": "BETA " * 200},
    {"path": "c.py", "content": "GAMMA " * 200},
]
CHUNKS = ["retrieved chunk one about ALPHA", "retrieved chunk two about GAMMA"]


def test_a0_full_includes_every_file():
    ctx = R.build_context("A0", corpus_files=FILES, retrieved_chunks=CHUNKS,
                          budget_tokens=10_000)
    assert "ALPHA" in ctx and "BETA" in ctx and "GAMMA" in ctx


def test_a1_naive_truncates_to_budget_from_the_head():
    ctx = R.build_context("A1", corpus_files=FILES, retrieved_chunks=CHUNKS,
                          budget_tokens=50)        # ~200 chars
    from benchmarks import b9_rag_stats as ST
    assert ST.est_tokens(ctx) <= 60               # within budget (+ small slack)
    assert "ALPHA" in ctx                          # head kept
    assert "GAMMA" not in ctx                       # tail (c.py) dropped


def test_a2_smart_includes_a_file_manifest():
    ctx = R.build_context("A2", corpus_files=FILES, retrieved_chunks=CHUNKS,
                          budget_tokens=200)
    # steelman: a manifest of all files + heads, not random bytes
    assert "a.py" in ctx and "b.py" in ctx and "c.py" in ctx


def test_a3_rag_uses_only_retrieved_chunks():
    ctx = R.build_context("A3", corpus_files=FILES, retrieved_chunks=CHUNKS,
                          budget_tokens=10_000)
    assert "retrieved chunk one" in ctx and "retrieved chunk two" in ctx
    # It must NOT contain the bulk corpus (that's the whole point — cheap context).
    assert "BETA" not in ctx


def test_a2prime_budget_matched_to_retrieved_token_count():
    from benchmarks import b9_rag_stats as ST
    retrieved_budget = ST.est_tokens("".join(CHUNKS))
    ctx = R.build_context("A2prime", corpus_files=FILES, retrieved_chunks=CHUNKS,
                          budget_tokens=None)       # derives budget from chunks
    # A2' must use ~the same token budget the RAG arm actually used.
    assert ST.est_tokens(ctx) <= retrieved_budget + 40


def test_unknown_arm_raises():
    try:
        R.build_context("ZZZ", corpus_files=FILES, retrieved_chunks=CHUNKS,
                        budget_tokens=100)
        raise AssertionError("should reject unknown arm")
    except ValueError:
        pass


def main() -> int:
    tests = [test_a0_full_includes_every_file,
             test_a1_naive_truncates_to_budget_from_the_head,
             test_a2_smart_includes_a_file_manifest,
             test_a3_rag_uses_only_retrieved_chunks,
             test_a2prime_budget_matched_to_retrieved_token_count,
             test_unknown_arm_raises]
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
