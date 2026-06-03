"""Smoke test for core/capability_matrix.py (L-24).

Per FINAL_implementation_plan_amendment_2026-05-13.md §A4 + FALS-L24-03.

Tests:
  1. load_rows on empty/missing file returns []
  2. append_row writes a valid JSONL line; load_rows reads it back
  3. rows_by_role groups + sorts by score descending
  4. best_for_role returns highest-score row with headroom filter
  5. best_for_role excludes by family
  6. best_for_role returns None when no row satisfies constraints
  7. FALS-L24-03: pick_evaluator_model body references capability_matrix

Run: `.venv/bin/python tests/_smoke_capability_matrix.py`
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import capability_matrix as cm  # noqa: E402


def _tmpfile() -> Path:
    return Path(tempfile.mkdtemp(prefix="bert_capmatrix_")) / "matrix.jsonl"


def test_load_rows_missing_file_returns_empty() -> None:
    p = _tmpfile()
    assert not p.exists()
    assert cm.load_rows(p) == []


def test_append_then_load() -> None:
    p = _tmpfile()
    row = cm.CapabilityRow(
        ts="2026-05-13T00:00:00Z",
        role="evaluator", provider="nvidia",
        model="qwen/qwen3-next-80b-a3b-thinking",
        score=0.88, cost_per_task_usd=0.0,
        latency_p50_ms=2000, latency_p95_ms=4000,
        quota_headroom_pct=85, task_count=30,
        reference_set="test",
    )
    cm.append_row(row, p)
    rows = cm.load_rows(p)
    assert len(rows) == 1
    assert rows[0].role == "evaluator"
    assert rows[0].score == 0.88


def test_rows_by_role_groups_and_sorts() -> None:
    p = _tmpfile()
    for r in [
        cm.CapabilityRow("2026-05-13T00:00:00Z", "evaluator", "a", "m1", 0.5, 0, 0, 0, 100, 30, ""),
        cm.CapabilityRow("2026-05-13T00:00:00Z", "evaluator", "b", "m2", 0.9, 0, 0, 0, 100, 30, ""),
        cm.CapabilityRow("2026-05-13T00:00:00Z", "researcher", "c", "m3", 0.7, 0, 0, 0, 100, 30, ""),
    ]:
        cm.append_row(r, p)
    grouped = cm.rows_by_role(cm.load_rows(p))
    assert set(grouped.keys()) == {"evaluator", "researcher"}
    assert [r.score for r in grouped["evaluator"]] == [0.9, 0.5]


def test_rows_by_role_keeps_latest_per_provider_model() -> None:
    p = _tmpfile()
    for r in [
        cm.CapabilityRow("2026-05-13T00:00:00Z", "evaluator", "a", "m1", 0.5, 0, 0, 0, 100, 30, ""),
        cm.CapabilityRow("2026-05-13T01:00:00Z", "evaluator", "a", "m1", 0.7, 0, 0, 0, 100, 30, ""),
    ]:
        cm.append_row(r, p)
    grouped = cm.rows_by_role(cm.load_rows(p))
    assert len(grouped["evaluator"]) == 1
    assert grouped["evaluator"][0].score == 0.7


def test_best_for_role_with_headroom() -> None:
    p = _tmpfile()
    cm.MATRIX_PATH = p  # redirect module-level default
    for r in [
        cm.CapabilityRow("2026-05-13T00:00:00Z", "evaluator", "a", "m1", 0.9, 0, 0, 0, 5, 30, ""),  # no headroom
        cm.CapabilityRow("2026-05-13T00:00:00Z", "evaluator", "b", "m2", 0.8, 0, 0, 0, 95, 30, ""),  # has headroom
    ]:
        cm.append_row(r, p)
    best = cm.best_for_role("evaluator")
    assert best is not None
    assert best.score == 0.8
    assert best.provider == "b"


def test_best_for_role_excludes_family() -> None:
    p = _tmpfile()
    cm.MATRIX_PATH = p
    for r in [
        cm.CapabilityRow("2026-05-13T00:00:00Z", "evaluator", "nvidia", "llama-3.3", 0.9, 0, 0, 0, 90, 30, ""),
        cm.CapabilityRow("2026-05-13T00:00:00Z", "evaluator", "mistral", "mistral-small", 0.7, 0, 0, 0, 90, 30, ""),
    ]:
        cm.append_row(r, p)
    family_of = lambda prov: {"nvidia": "llama", "mistral": "mistral"}.get(prov, "?")
    best = cm.best_for_role(
        "evaluator",
        exclude_family="llama",
        family_of_fn=family_of,
    )
    assert best is not None
    assert best.provider == "mistral"


def test_best_for_role_returns_none_when_no_match() -> None:
    p = _tmpfile()
    cm.MATRIX_PATH = p
    cm.append_row(
        cm.CapabilityRow("2026-05-13T00:00:00Z", "evaluator", "x", "m", 0.9, 0, 0, 0, 5, 30, ""),
        p,
    )
    # All rows below the headroom threshold → None
    best = cm.best_for_role("evaluator", min_headroom_pct=50)
    assert best is None


def test_fals_l24_03_pick_evaluator_model_references_matrix() -> None:
    """FALS-L24-03: pick_evaluator_model body must reference capability_matrix."""
    src = (LAB_ROOT / "core" / "subagent.py").read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "pick_evaluator_model"
    )
    body_src = ast.unparse(fn)
    assert "capability_matrix" in body_src


def main() -> int:
    tests = [
        test_load_rows_missing_file_returns_empty,
        test_append_then_load,
        test_rows_by_role_groups_and_sorts,
        test_rows_by_role_keeps_latest_per_provider_model,
        test_best_for_role_with_headroom,
        test_best_for_role_excludes_family,
        test_best_for_role_returns_none_when_no_match,
        test_fals_l24_03_pick_evaluator_model_references_matrix,
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
