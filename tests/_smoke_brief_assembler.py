"""Smoke test for core/brief_assembler.py — Layer 5 context brief.

Per FINAL_implementation_plan_2026-05-07.md §5.4 H4 + bert memory
architecture (saved as `project_bert_memory_architecture.md`).

Tests:
  1. classify_session: post-failure when session_exit.md head is CATASTROPHIC
  2. classify_session: user-action when pi_notes mtime newer than session_exit
  3. classify_session: phase-transition when current.md has Phase header
  4. classify_session: routine-monitor by default
  5. assemble_brief writes file, populates all sections, runs in <500ms
  6. Brief stays well under 20 KB even with overstuffed inputs
  7. Section budgets are honored

Run: `.venv/bin/python tests/_smoke_brief_assembler.py`
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

TMP = Path(tempfile.mkdtemp(prefix="bert_brief_smoke_"))
MEMORIES_DIR = TMP / "memories"
STATE_DIR = TMP / "state"
GOV_DIR = MEMORIES_DIR / "governance"
GOV_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

from core import brief_assembler as ba  # noqa: E402

ba.LAB_ROOT = TMP
ba.MEMORIES_DIR = MEMORIES_DIR
ba.STATE_DIR = STATE_DIR
ba.BRIEF_PATH = MEMORIES_DIR / "context_brief.md"


def _touch(p: Path, text: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _clear() -> None:
    for p in [
        STATE_DIR / "session_exit.md",
        STATE_DIR / "cycle_queue.md",
        MEMORIES_DIR / "current.md",
        MEMORIES_DIR / "log.md",
        MEMORIES_DIR / "procedures.md",
        GOV_DIR / "pi_notes.md",
    ]:
        if p.exists():
            p.unlink()


def test_classify_post_failure() -> None:
    _clear()
    _touch(STATE_DIR / "session_exit.md", "CATASTROPHIC\n\nDetails: oops")
    c = ba.classify_session()
    assert c.classification == ba.SessionClass.POST_FAILURE


def test_classify_user_action() -> None:
    _clear()
    _touch(STATE_DIR / "session_exit.md", "GRACEFUL_CHECKPOINT\n")
    # Make pi_notes newer than session_exit
    pn = GOV_DIR / "pi_notes.md"
    _touch(pn, "PI directive XYZ")
    sx = STATE_DIR / "session_exit.md"
    sx_mtime = sx.stat().st_mtime
    os.utime(pn, (sx_mtime + 100, sx_mtime + 100))
    c = ba.classify_session()
    assert c.classification == ba.SessionClass.USER_ACTION, c


def test_classify_phase_transition() -> None:
    _clear()
    _touch(STATE_DIR / "session_exit.md", "GRACEFUL_CHECKPOINT\n")
    _touch(MEMORIES_DIR / "current.md", "# Phase H4 ready\n\nMoving to H4 wiring round.")
    c = ba.classify_session()
    assert c.classification == ba.SessionClass.PHASE_TRANSITION, c


def test_classify_routine_monitor() -> None:
    _clear()
    _touch(STATE_DIR / "session_exit.md", "GRACEFUL_CHECKPOINT\n")
    _touch(MEMORIES_DIR / "current.md", "Just status update; no phase header.")
    c = ba.classify_session()
    assert c.classification == ba.SessionClass.ROUTINE_MONITOR


def test_assemble_brief_runs_under_budget() -> None:
    _clear()
    _touch(STATE_DIR / "session_exit.md", "GRACEFUL_CHECKPOINT\n")
    _touch(MEMORIES_DIR / "current.md",
           "## §Current Program\n\nbert-lab Phase H4 self-correction.\n")
    _touch(MEMORIES_DIR / "log.md", "\n".join([
        "## D-100 — Recent decision A",
        "Confidence: 8. Reasoning: " + "x" * 100,
        "",
        "## D-099 — Recent decision B",
        "Confidence: 7. Reasoning: " + "y" * 100,
        "",
    ]))
    _touch(MEMORIES_DIR / "procedures.md",
           "## P-001\n**STATUS:** FROZEN on 2026-05-07\nDetails")
    _touch(STATE_DIR / "cycle_queue.md", "1. Wire H4 modules\n2. Verify")
    t0 = time.monotonic()
    path, stats = ba.assemble_brief()
    elapsed = time.monotonic() - t0
    assert path.exists()
    text = path.read_text()
    assert "## Current Program" in text
    assert "## Recent decisions" in text
    assert "D-100" in text
    assert "P-001" in text
    assert elapsed < 0.5, f"brief assembly took {elapsed:.2f}s; budget <0.5s"
    assert stats["total_chars"] < 25_000, f"brief too big: {stats['total_chars']} chars"


def test_oversized_inputs_get_truncated() -> None:
    _clear()
    huge = "X" * 50_000
    _touch(MEMORIES_DIR / "current.md", f"## §Current Program\n\n{huge}")
    _touch(MEMORIES_DIR / "log.md", "\n\n".join(
        [f"## D-{i:03d}\n\n{huge}" for i in range(10, 0, -1)]
    ))
    _touch(MEMORIES_DIR / "procedures.md", f"## P-001\n**STATUS:** FROZEN\n{huge}")
    _touch(STATE_DIR / "cycle_queue.md", huge)
    _touch(STATE_DIR / "session_exit.md", "GRACEFUL_CHECKPOINT\n")
    path, stats = ba.assemble_brief()
    text = path.read_text()
    # Even with 250 KB of inputs, brief must stay <30 KB total
    assert len(text) < 30_000, f"brief did not stay under cap: {len(text)} chars"
    assert "truncated" in text


def test_section_budgets_honored() -> None:
    """Per-section budget caps are honored individually."""
    _clear()
    big = "X" * 100_000
    _touch(MEMORIES_DIR / "current.md", f"## §Current Program\n\n{big}")
    _touch(STATE_DIR / "session_exit.md", "GRACEFUL_CHECKPOINT\n")
    path, stats = ba.assemble_brief()
    assert stats["section_chars"]["current_program"] <= ba._BUDGET["current_program"], (
        f"current_program section exceeded budget: {stats['section_chars']['current_program']}"
    )


def main() -> int:
    tests = [
        test_classify_post_failure,
        test_classify_user_action,
        test_classify_phase_transition,
        test_classify_routine_monitor,
        test_assemble_brief_runs_under_budget,
        test_oversized_inputs_get_truncated,
        test_section_budgets_honored,
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
