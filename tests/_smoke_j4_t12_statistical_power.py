"""Smoke test for J.4: T12 seasoning_revival_rate statistical-power gate."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.falsifier_baseline as fb


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_t12_insufficient_when_under_20_entries() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_t12_"))
    orig = fb.OBS_DIR
    fb.OBS_DIR = tmp
    try:
        _write_events(tmp / "seasoning_entry.jsonl", [
            {"id": f"s{i}", "cycle": 100} for i in range(7)
        ])
        _write_events(tmp / "seasoning_revive.jsonl", [
            {"id": f"s{i}"} for i in range(2)
        ])
        r = fb.t12_seasoning_revival_rate(window=30)
        assert r.status.value == "INSUFFICIENT_DATA", (
            f"expected INSUFFICIENT_DATA, got {r.status.value}"
        )
        assert "statistical power" in r.notes
    finally:
        fb.OBS_DIR = orig
        import shutil
        shutil.rmtree(tmp)


def test_t12_insufficient_when_under_5_distinct_cycles() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_t12_"))
    orig = fb.OBS_DIR
    fb.OBS_DIR = tmp
    try:
        # 25 entries but all from 2 cycles → still insufficient
        _write_events(tmp / "seasoning_entry.jsonl", [
            {"id": f"s{i}", "cycle": 100 if i < 12 else 200}
            for i in range(25)
        ])
        _write_events(tmp / "seasoning_revive.jsonl", [
            {"id": "s1"}, {"id": "s2"}
        ])
        r = fb.t12_seasoning_revival_rate(window=30)
        assert r.status.value == "INSUFFICIENT_DATA"
        assert "cycles" in r.notes.lower()
    finally:
        fb.OBS_DIR = orig
        import shutil
        shutil.rmtree(tmp)


def test_t12_passes_with_sufficient_data_and_low_rate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_t12_"))
    orig = fb.OBS_DIR
    fb.OBS_DIR = tmp
    try:
        # 25 entries across 5 cycles, 2 revives → 8% rate → PASS
        _write_events(tmp / "seasoning_entry.jsonl", [
            {"id": f"s{i}", "cycle": 100 + (i % 5) * 10}
            for i in range(25)
        ])
        _write_events(tmp / "seasoning_revive.jsonl", [
            {"id": "s1"}, {"id": "s2"}
        ])
        r = fb.t12_seasoning_revival_rate(window=30)
        assert r.status.value == "PASS", (
            f"expected PASS at 8% rate, got {r.status.value} {r.current_value}"
        )
    finally:
        fb.OBS_DIR = orig
        import shutil
        shutil.rmtree(tmp)


def test_t12_fails_with_sufficient_data_and_high_rate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_t12_"))
    orig = fb.OBS_DIR
    fb.OBS_DIR = tmp
    try:
        # 25 entries across 5 cycles, 10 revives → 40% rate → FAIL
        _write_events(tmp / "seasoning_entry.jsonl", [
            {"id": f"s{i}", "cycle": 100 + (i % 5) * 10}
            for i in range(25)
        ])
        _write_events(tmp / "seasoning_revive.jsonl", [
            {"id": f"s{i}"} for i in range(10)
        ])
        r = fb.t12_seasoning_revival_rate(window=30)
        assert r.status.value == "FAIL"
    finally:
        fb.OBS_DIR = orig
        import shutil
        shutil.rmtree(tmp)


def test_t12_insufficient_when_no_entries_at_all() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_t12_"))
    orig = fb.OBS_DIR
    fb.OBS_DIR = tmp
    try:
        r = fb.t12_seasoning_revival_rate(window=30)
        assert r.status.value == "INSUFFICIENT_DATA"
    finally:
        fb.OBS_DIR = orig
        import shutil
        shutil.rmtree(tmp)


def main() -> int:
    tests = [
        test_t12_insufficient_when_under_20_entries,
        test_t12_insufficient_when_under_5_distinct_cycles,
        test_t12_passes_with_sufficient_data_and_low_rate,
        test_t12_fails_with_sufficient_data_and_high_rate,
        test_t12_insufficient_when_no_entries_at_all,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
