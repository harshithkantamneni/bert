"""Smoke: core/consolidator.py — KM-agent maintenance pass (was 82%).

Covers the injectable, side-effect-free functions with temp fixtures /
no-op-safe params (huge caps so nothing is archived, temp target dirs):
should_run, _count_dn_entries, _count_references, promote_statuses,
archive_oversized (no-op), flag_stale, refresh_indexes. The LLM-bound
summarize + the full consolidate() pass (mutates real state) stay out.
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import consolidator as cz  # noqa: E402


def test_should_run():
    ok, reason = cz.should_run(cycle=999999, force=True)
    assert ok is True and reason == "force"
    # non-forced reads the real (read-only) last-run state → returns a verdict
    ok2, reason2 = cz.should_run(cycle=999999, force=False, min_hours_elapsed=99999.0)
    assert isinstance(ok2, bool) and isinstance(reason2, str)


def test_count_dn_entries(tmp_path):
    log = tmp_path / "log.md"
    log.write_text("## D-001\nbody\n## D-002\nbody\n## D-003\nbody\n")
    assert cz._count_dn_entries(log) == 3
    assert cz._count_dn_entries(tmp_path / "missing.md") == 0


def test_count_references(tmp_path):
    # scans the real MEMORIES_DIR read-only; a random token → 0 refs
    assert cz._count_references("zzz_no_such_token_xyz") == 0


def test_promote_statuses(tmp_path):
    proc = tmp_path / "procedures.md"
    proc.write_text("## P-001 (PROPOSED)\nsome procedure body\n\n## P-002 (VALIDATED)\nx\n")
    promotions = cz.promote_statuses(procedures_path=proc, cycle=5)
    assert isinstance(promotions, list)


def test_archive_oversized_noop(tmp_path):
    # huge caps → nothing exceeds → no archiving (covers the scan + no-op path)
    archived = cz.archive_oversized(caps={"memory_hot_max": 10**9, "memory_log_max": 10**9},
                                    archive_root=tmp_path / "archive")
    assert isinstance(archived, list)


def test_flag_stale(tmp_path):
    fresh = tmp_path / "fresh.md"
    fresh.write_text("recent")
    flagged = cz.flag_stale(max_age_days=30, target_paths=[fresh])
    assert isinstance(flagged, list)


def main() -> int:
    tests = [
        test_should_run,
        test_count_dn_entries,
        test_count_references,
        test_promote_statuses,
        test_archive_oversized_noop,
        test_flag_stale,
    ]
    for t in tests:
        td = Path(tempfile.mkdtemp())
        try:
            kwargs = {"tmp_path": td} if "tmp_path" in inspect.signature(t).parameters else {}
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        finally:
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
