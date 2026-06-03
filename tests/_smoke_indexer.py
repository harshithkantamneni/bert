"""Smoke test for core/indexer.py — fs-watcher + re-embed daemon.

Per FINAL_implementation_plan_2026-05-07.md §5.4 H4.

Tests:
  1. _is_indexable: only .md files in non-hidden dirs
  2. _DebouncedHandler delays reindex until burst settles
  3. _DebouncedHandler coalesces multiple events into one reindex
  4. flush_now triggers immediate reindex
  5. Stats persisted to JSON file
  6. Reindex exception caught — handler doesn't crash
  7. Non-indexable events ignored (don't bump files_seen)

Run: `.venv/bin/python tests/_smoke_indexer.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import indexer  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="bert_indexer_smoke_"))


def test_is_indexable() -> None:
    assert indexer._is_indexable("memories/log.md")
    assert not indexer._is_indexable("memories/.hidden.md")
    assert not indexer._is_indexable("logs/cycle_1.jsonl")
    assert not indexer._is_indexable("memories/config.txt")
    assert not indexer._is_indexable(".venv/lib/python3.13/site-packages/foo.md")
    assert not indexer._is_indexable("__pycache__/bar.md")


def test_debouncer_delays_reindex() -> None:
    calls = {"n": 0}
    def reindex():
        calls["n"] += 1
        return 0
    h = indexer._DebouncedHandler(
        reindex_fn=reindex, debounce_secs=0.3,
        stats_path=TMP / "stats1.json",
    )
    h.on_event(src_path="memories/x.md")
    # Immediately after — reindex should NOT have fired yet
    assert calls["n"] == 0
    time.sleep(0.5)
    assert calls["n"] == 1


def test_debouncer_coalesces_burst() -> None:
    calls = {"n": 0}
    def reindex():
        calls["n"] += 1
        return 3
    h = indexer._DebouncedHandler(
        reindex_fn=reindex, debounce_secs=0.3,
        stats_path=TMP / "stats2.json",
    )
    for i in range(10):
        h.on_event(src_path=f"memories/x{i}.md")
        time.sleep(0.02)
    # 10 events × 20ms = 200ms < 300ms debounce — single fire
    time.sleep(0.5)
    assert calls["n"] == 1, f"expected 1 fire after burst; got {calls['n']}"
    assert h.stats.files_seen == 10
    assert h.stats.chunks_reindexed == 3


def test_flush_now_fires_immediately() -> None:
    calls = {"n": 0}
    def reindex():
        calls["n"] += 1
        return 0
    h = indexer._DebouncedHandler(
        reindex_fn=reindex, debounce_secs=10.0,
        stats_path=TMP / "stats3.json",
    )
    h.on_event(src_path="memories/x.md")
    assert calls["n"] == 0
    h.flush_now()
    assert calls["n"] == 1


def test_stats_persisted() -> None:
    sp = TMP / "stats4.json"
    h = indexer._DebouncedHandler(
        reindex_fn=lambda: 7, debounce_secs=0.1, stats_path=sp,
    )
    h.on_event(src_path="memories/y.md")
    time.sleep(0.3)
    assert sp.exists()
    payload = json.loads(sp.read_text())
    assert payload["chunks_reindexed"] == 7
    assert payload["files_seen"] == 1
    assert payload["last_run_elapsed_ms"] >= 0


def test_reindex_exception_swallowed() -> None:
    def boom():
        raise RuntimeError("simulated reindex failure")
    h = indexer._DebouncedHandler(
        reindex_fn=boom, debounce_secs=0.1,
        stats_path=TMP / "stats5.json",
    )
    h.on_event(src_path="memories/y.md")
    time.sleep(0.3)
    # Did not crash; chunks_reindexed stays 0; files_seen still bumped
    assert h.stats.files_seen == 1
    assert h.stats.chunks_reindexed == 0


def test_non_indexable_events_ignored() -> None:
    h = indexer._DebouncedHandler(
        reindex_fn=lambda: 0, debounce_secs=0.1,
        stats_path=TMP / "stats6.json",
    )
    h.on_event(src_path="logs/cycle_1.jsonl")  # not .md
    h.on_event(src_path="memories/.hidden.md")  # hidden
    h.on_event(src_path=".venv/lib/foo.md")  # excluded path
    assert h.stats.files_seen == 0


def main() -> int:
    tests = [
        test_is_indexable,
        test_debouncer_delays_reindex,
        test_debouncer_coalesces_burst,
        test_flush_now_fires_immediately,
        test_stats_persisted,
        test_reindex_exception_swallowed,
        test_non_indexable_events_ignored,
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
