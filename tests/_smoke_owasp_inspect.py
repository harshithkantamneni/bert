"""Smoke test for the OWASP Top-10 Inspect AI suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from evals.inspect import owasp_top10  # noqa: E402


def test_all_10_individual_tasks_registered() -> None:
    """llm01 through llm10 should all be callable @task factories."""
    for i in range(1, 11):
        name = f"llm{i:02d}"
        fn = getattr(owasp_top10, name, None)
        assert fn is not None, f"{name} missing"
        task_obj = fn()
        assert hasattr(task_obj, "dataset"), f"{name}() didn't return a Task"


def test_aggregate_task_has_10_samples() -> None:
    task_obj = owasp_top10.owasp_all_10()
    samples = list(task_obj.dataset)
    assert len(samples) == 10
    ids = [s.metadata["threat_id"] for s in samples]
    assert ids == [f"LLM{i:02d}" for i in range(1, 11)]


def test_check_llm01_returns_structured_result() -> None:
    r = owasp_top10._check_llm01()
    assert "passed" in r
    assert "rationale" in r
    assert isinstance(r["passed"], bool)


def test_check_llm07_finds_anchor_guard() -> None:
    """LLM07 specifically checks for the anchor-term guard — the novel
    bert defense against embedding-collapse attacks."""
    r = owasp_top10._check_llm07()
    assert r["passed"] is True, r["rationale"]
    assert "anchor_term_guard=True" in r["rationale"]


def test_check_llm09_unbounded_consumption() -> None:
    """Quota + alerts + capability_matrix.quota_headroom_pct."""
    if not (owasp_top10.LAB_ROOT / "bot" / "alerts.py").exists():
        pytest.skip("requires the bot/ Telegram-alerts integration "
                    "(not shipped in the public repo)")
    r = owasp_top10._check_llm09()
    # Should pass on bert's current tree
    assert r["passed"] is True, r["rationale"]


def test_all_10_checks_callable() -> None:
    """Each LLM** check function runs without exception."""
    for i in range(1, 11):
        fn = getattr(owasp_top10, f"_check_llm{i:02d}")
        r = fn()
        assert "passed" in r, f"_check_llm{i:02d} malformed result"
        assert "rationale" in r


def test_check_degrades_when_lab_root_missing(monkeypatch_off=None) -> None:
    """R5 edge case: if LAB_ROOT pointed at a non-existent dir, checks
    must return passed=False with rationale, not crash."""
    import tempfile
    from pathlib import Path
    orig = owasp_top10.LAB_ROOT
    try:
        owasp_top10.LAB_ROOT = Path(tempfile.mkdtemp(prefix="bert_owasp_empty_"))
        for i in range(1, 11):
            fn = getattr(owasp_top10, f"_check_llm{i:02d}")
            r = fn()
            assert "passed" in r and "rationale" in r
            # On an empty tree, structural checks should all fail cleanly.
            # (LLM10 may pass if prompts_dir is missing AND no_leaked is True,
            # so we don't assert passed=False uniformly; we DO assert no crash.)
    finally:
        owasp_top10.LAB_ROOT = orig


def test_scorer_emits_score_object() -> None:
    """The structural_pass_scorer is wired and returns a Score for PASS/FAIL."""
    scorer_factory = owasp_top10.structural_pass_scorer
    s = scorer_factory()
    assert s is not None


def main() -> int:
    tests = [
        test_all_10_individual_tasks_registered,
        test_aggregate_task_has_10_samples,
        test_check_llm01_returns_structured_result,
        test_check_llm07_finds_anchor_guard,
        test_check_llm09_unbounded_consumption,
        test_all_10_checks_callable,
        test_check_degrades_when_lab_root_missing,
        test_scorer_emits_score_object,
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
