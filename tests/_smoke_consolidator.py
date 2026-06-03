"""Smoke test for core/consolidator.py — async memory KM agent.

Per FINAL_implementation_plan_2026-05-07.md §5.4 H4.

Tests:
  1. should_run respects force, hours, new-DN-count thresholds
  2. promote_statuses promotes PROPOSED entries with cross-refs
  3. archive_oversized archives oldest third of an over-cap file
  4. refresh_indexes writes INDEX.md per directory
  5. flag_stale annotates files with old mtimes
  6. consolidate skipped when triggers not met
  7. consolidate honors force=True
  8. last-run state file roundtrips
  9. summarize_log_head returns False when under cap (no LLM call)

Run: `.venv/bin/python tests/_smoke_consolidator.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

TMP = Path(tempfile.mkdtemp(prefix="bert_consolidator_smoke_"))
MEMORIES_DIR = TMP / "memories"
ARCHIVE_DIR = MEMORIES_DIR / "archive"
STATE_DIR = TMP / "lab" / "state"
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

from core import consolidator as cm  # noqa: E402

cm.LAB_ROOT = TMP
cm.MEMORIES_DIR = MEMORIES_DIR
cm.ARCHIVE_DIR = ARCHIVE_DIR
cm.LAST_RUN_PATH = STATE_DIR / "consolidator.last_run.json"


def _reset() -> None:
    for p in MEMORIES_DIR.rglob("*"):
        if p.is_file():
            p.unlink()
    if cm.LAST_RUN_PATH.exists():
        cm.LAST_RUN_PATH.unlink()


def test_should_run_force() -> None:
    _reset()
    ok, reason = cm.should_run(cycle=1, force=True)
    assert ok and reason == "force"


def test_should_run_hours_threshold() -> None:
    _reset()
    cm.LAST_RUN_PATH.write_text(json.dumps({"last_ts": time.time(), "last_cycle": 0, "log_dn_count": 0}))
    ok, _ = cm.should_run(cycle=1, min_hours_elapsed=6.0)
    assert not ok, "fresh last-run should suppress trigger"
    cm.LAST_RUN_PATH.write_text(json.dumps({
        "last_ts": time.time() - 7 * 3600, "last_cycle": 0, "log_dn_count": 0,
    }))
    ok, reason = cm.should_run(cycle=1, min_hours_elapsed=6.0)
    assert ok
    assert "since last run" in reason


def test_should_run_new_dn_count() -> None:
    _reset()
    log_md = MEMORIES_DIR / "log.md"
    log_md.write_text("\n".join(f"## D-{i:03d}\nReason X\n" for i in range(20)))
    cm.LAST_RUN_PATH.write_text(json.dumps({
        "last_ts": time.time(), "last_cycle": 0, "log_dn_count": 5,
    }))
    ok, reason = cm.should_run(cycle=1, min_new_dn=10)
    assert ok
    assert "D-N entries" in reason


def test_promote_statuses() -> None:
    _reset()
    pp = MEMORIES_DIR / "procedures.md"
    pp.write_text(
        "## P-100 — example\n"
        "**STATUS:** PROPOSED on 2026-05-07\n"
        "Body of P-100\n"
    )
    # Cross-ref from another file
    (MEMORIES_DIR / "log.md").write_text("D-101 references P-100 in passing")
    promotions = cm.promote_statuses(cycle=2)
    assert len(promotions) == 1
    assert "P-100" in promotions[0]
    txt = pp.read_text()
    assert "VALIDATED" in txt
    assert "PROPOSED" not in txt


def test_archive_oversized() -> None:
    _reset()
    big = MEMORIES_DIR / "log.md"
    big.write_text("X" * 60_000)  # over 30k cap
    caps = {"memories/log.md": 30_000}
    archived = cm.archive_oversized(caps=caps, archive_root=ARCHIVE_DIR)
    assert archived == ["memories/log.md"]
    new_size = big.stat().st_size
    assert new_size < 60_000
    # Archive file exists
    today = time.strftime("%Y-%m-%d", time.gmtime())
    assert (ARCHIVE_DIR / today / "log.md").exists()


def test_refresh_indexes() -> None:
    _reset()
    sub = MEMORIES_DIR / "programs" / "p001"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "alpha.md").write_text("alpha doc")
    (sub / "beta.md").write_text("beta doc")
    refreshed = cm.refresh_indexes()
    # Find programs/INDEX.md
    idx = MEMORIES_DIR / "programs" / "INDEX.md"
    assert idx.exists()
    txt = idx.read_text()
    assert "alpha" in txt and "beta" in txt


def test_flag_stale() -> None:
    _reset()
    pp = MEMORIES_DIR / "procedures.md"
    pp.write_text("## P-1\nbody")
    # Backdate mtime to 60 days ago
    old = time.time() - 60 * 86400
    os.utime(pp, (old, old))
    flagged = cm.flag_stale(max_age_days=30, target_paths=[pp])
    assert "memories/procedures.md" in flagged
    assert "STALE" in pp.read_text()


def test_consolidate_skipped_when_triggers_not_met() -> None:
    _reset()
    cm.LAST_RUN_PATH.write_text(json.dumps({
        "last_ts": time.time(), "last_cycle": 5, "log_dn_count": 0,
    }))
    rep = cm.consolidate(cycle=5)
    assert rep.skipped
    assert rep.skip_reason


def test_consolidate_force_runs() -> None:
    _reset()
    rep = cm.consolidate(cycle=1, force=True, summarize=False)
    assert not rep.skipped
    assert rep.finished_ts > rep.started_ts


def test_summarize_log_head_no_op_under_cap() -> None:
    _reset()
    log_md = MEMORIES_DIR / "log.md"
    log_md.write_text("small log under cap")
    fired = cm.summarize_log_head(log_path=log_md, cap_bytes=30_000)
    assert not fired


def test_last_run_roundtrip() -> None:
    _reset()
    rep = cm.consolidate(cycle=42, force=True)
    assert cm.LAST_RUN_PATH.exists()
    payload = json.loads(cm.LAST_RUN_PATH.read_text())
    assert payload["last_cycle"] == 42


def main() -> int:
    tests = [
        test_should_run_force,
        test_should_run_hours_threshold,
        test_should_run_new_dn_count,
        test_promote_statuses,
        test_archive_oversized,
        test_refresh_indexes,
        test_flag_stale,
        test_consolidate_skipped_when_triggers_not_met,
        test_consolidate_force_runs,
        test_summarize_log_head_no_op_under_cap,
        test_last_run_roundtrip,
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
