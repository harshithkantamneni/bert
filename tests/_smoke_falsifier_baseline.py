"""Smoke test for tools/falsifier_baseline.py — measurement framework.

Tests:
  1. Empty data → 14/14 INSUFFICIENT_DATA
  2. Seeded threshing verdict events → t2 fires PASS at ≥70% SCOPE_STOP
  3. Seeded stand-aside verdict events → t7 fires
  4. Seeded seasoning entries with valid summaries → t14 PASS, t11 PASS
  5. render_markdown produces a well-formed table with all 14 rows
  6. JSON payload is round-trippable

Run: `.venv/bin/python tests/_smoke_falsifier_baseline.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

# Use a temp tree
TMP = Path(tempfile.mkdtemp(prefix="bert_falsifier_smoke_"))
OBS_DIR = TMP / "state" / "observability"
RESULTS_DIR = TMP / "state" / "results"
SEASONING_PATH = TMP / "lab" / "sod" / "seasoning.jsonl"
FINDINGS_DIR = TMP / "findings"
for d in [OBS_DIR, RESULTS_DIR, SEASONING_PATH.parent, FINDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import tools.falsifier_baseline as fb  # noqa: E402

fb.LAB_ROOT = TMP
fb.OBS_DIR = OBS_DIR
fb.RESULTS_DIR = RESULTS_DIR
fb.FINDINGS_DIR = FINDINGS_DIR
fb.SEASONING_PATH = SEASONING_PATH


def _clear() -> None:
    for d in [OBS_DIR, RESULTS_DIR, SEASONING_PATH.parent]:
        for p in d.glob("*"):
            if p.is_file():
                p.unlink()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_empty_data_all_insufficient() -> None:
    _clear()
    results = fb.run_all(window=30)
    assert all(r.status == fb.Status.INSUFFICIENT for r in results)


def test_threshing_verdict_pass() -> None:
    _clear()
    # 28 SCOPE_STOP / 2 OTHER → 93% ≥ 70%
    rows = [{"role": "threshing_pass", "verdict": "SCOPE_STOP"} for _ in range(28)]
    rows += [{"role": "threshing_pass", "verdict": "OTHER"} for _ in range(2)]
    _write_jsonl(OBS_DIR / "verdict.jsonl", rows)
    results = fb.run_all(window=30)
    t2 = next(r for r in results if r.target_id == 2)
    assert t2.status == fb.Status.PASS, f"expected PASS; got {t2.status} ({t2.current_value})"


def test_threshing_verdict_fail() -> None:
    _clear()
    # 5 SCOPE_STOP / 25 OTHER → 17% < 70%
    rows = [{"role": "threshing_pass", "verdict": "SCOPE_STOP"} for _ in range(5)]
    rows += [{"role": "threshing_pass", "verdict": "OTHER"} for _ in range(25)]
    _write_jsonl(OBS_DIR / "verdict.jsonl", rows)
    results = fb.run_all(window=30)
    t2 = next(r for r in results if r.target_id == 2)
    assert t2.status == fb.Status.FAIL, f"expected FAIL; got {t2.status}"


def test_stand_aside_concerns_populated_pass() -> None:
    _clear()
    rows = [{"concern_count": 2} for _ in range(28)]
    rows += [{"concern_count": 0} for _ in range(2)]
    _write_jsonl(OBS_DIR / "stand_aside_verdict.jsonl", rows)
    results = fb.run_all(window=30)
    t7 = next(r for r in results if r.target_id == 7)
    assert t7.status == fb.Status.PASS


def test_seasoning_well_formed_pass() -> None:
    _clear()
    rows = [
        {"id": f"s-{i}", "ts": "2026-05-07T00:00:00Z",
         "source_dispatch_id": "x", "verdict": "REJECT",
         "summary": "x" * 60, "revival_conditions": ["if PI revisits"], "cycle": 1}
        for i in range(5)
    ]
    _write_jsonl(SEASONING_PATH, rows)
    results = fb.run_all(window=30)
    t14 = next(r for r in results if r.target_id == 14)
    assert t14.status == fb.Status.PASS
    assert t14.sample_size == 5


def test_seasoning_queue_size_bounded_pass() -> None:
    _clear()
    rows = [
        {"id": f"s-{i}", "ts": "2026-05-07T00:00:00Z",
         "source_dispatch_id": "x", "verdict": "REJECT",
         "summary": "x" * 60, "revival_conditions": ["x" * 25],
         "cycle": 1}
        for i in range(20)  # 20 ≤ 25 cap
    ]
    _write_jsonl(SEASONING_PATH, rows)
    results = fb.run_all(window=30)
    t11 = next(r for r in results if r.target_id == 11)
    assert t11.status == fb.Status.PASS


def test_seasoning_queue_size_fail_when_over() -> None:
    _clear()
    rows = [
        {"id": f"s-{i}", "ts": "2026-05-07T00:00:00Z",
         "source_dispatch_id": "x", "verdict": "REJECT",
         "summary": "x" * 60, "revival_conditions": ["x" * 25],
         "cycle": 1}
        for i in range(30)
    ]
    _write_jsonl(SEASONING_PATH, rows)
    results = fb.run_all(window=30)
    t11 = next(r for r in results if r.target_id == 11)
    assert t11.status == fb.Status.FAIL


def test_render_markdown_well_formed() -> None:
    _clear()
    results = fb.run_all(window=30)
    md = fb.render_markdown(results, cycle=42)
    assert "# Falsifier baseline — cycle 42" in md
    # Each of 15 targets gets a row (t1-t14 engine discipline
    # + t15 supervisor_pattern_evidence).
    for tid in range(1, 16):
        assert f"| {tid} |" in md, f"target {tid} missing from table"


def test_run_all_returns_15() -> None:
    """The registry has 15 targets (t1-t14 engine-discipline
    + t15 supervisor_pattern_evidence)."""
    _clear()
    results = fb.run_all()
    assert len(results) == 15
    ids = sorted(r.target_id for r in results)
    assert ids == list(range(1, 16))


def main() -> int:
    tests = [
        test_empty_data_all_insufficient,
        test_threshing_verdict_pass,
        test_threshing_verdict_fail,
        test_stand_aside_concerns_populated_pass,
        test_seasoning_well_formed_pass,
        test_seasoning_queue_size_bounded_pass,
        test_seasoning_queue_size_fail_when_over,
        test_render_markdown_well_formed,
        test_run_all_returns_15,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
