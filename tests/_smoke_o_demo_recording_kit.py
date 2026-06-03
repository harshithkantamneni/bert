"""Smoke test for O.1-O.4: demo recording kit."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
KIT_DIR = LAB_ROOT / "findings" / "investor" / "demo_recording"


def test_kit_dir_exists() -> None:
    assert KIT_DIR.exists()
    assert KIT_DIR.is_dir()


def test_orchestrator_exists_and_is_executable() -> None:
    p = KIT_DIR / "demo_run.sh"
    assert p.exists()
    mode = p.stat().st_mode
    assert mode & stat.S_IXUSR, "demo_run.sh must be executable"


def test_orchestrator_has_safety_traps() -> None:
    """The orchestrator must clean up uvicorn + temp dir on exit AND
    propagate the exit code (V.3 split single EXIT trap into three so
    a preflight ABORT rc=2 isn't masked by cleanup returning 0)."""
    text = (KIT_DIR / "demo_run.sh").read_text()
    # Three independent traps preserve the original exit code
    assert "trap 'cleanup $?' EXIT" in text, \
        "EXIT trap must capture original rc via $?"
    assert "trap 'cleanup 130' INT" in text, \
        "INT trap must use canonical 130 (Ctrl-C)"
    assert "trap 'cleanup 143' TERM" in text, \
        "TERM trap must use canonical 143 (SIGTERM)"
    assert "BERT_DEMO_MODE=on" in text
    assert "BERT_DISABLE_IDLE_COMPUTE=1" in text


def test_orchestrator_invokes_bert_init_with_template() -> None:
    text = (KIT_DIR / "demo_run.sh").read_text()
    assert "tools/bert_init.py" in text
    assert "--from-template demo_note_cli" in text
    assert "--non-interactive" in text


def test_orchestrator_runs_bert_verify() -> None:
    text = (KIT_DIR / "demo_run.sh").read_text()
    assert "tools/bert_verify.py" in text
    assert "cycle-0400.tar.gz" in text


def test_storyboard_exists_with_both_cuts() -> None:
    p = KIT_DIR / "storyboard.md"
    assert p.exists()
    text = p.read_text()
    assert "60-second async cut" in text
    assert "5-minute live-pitch flow" in text
    # Each cut has a timestamped table
    assert "| Time |" in text


def test_narration_exists_with_both_cuts() -> None:
    p = KIT_DIR / "narration.md"
    assert p.exists()
    text = p.read_text()
    # Both 60-second async + 5-minute live versions named
    assert "60-second async cut" in text
    assert "5-minute live-pitch flow" in text
    # Tagline appears
    assert "Build privately. Prove publicly." in text


def test_readme_pre_flight_checklist() -> None:
    """README must include the pre-flight quality bar."""
    p = KIT_DIR / "README.md"
    assert p.exists()
    text = p.read_text()
    for needle in (
        "Pre-flight",
        "Cmd+Shift+5",
        "Do Not Disturb",
        "Quality checklist",
        "Common failure modes",
    ):
        assert needle in text, f"README missing {needle!r}"


def test_orchestrator_runs_clean_short_form() -> None:
    """Smoke: run a non-interactive variant of the orchestrator with
    auto-enter and verify it sets up + tears down without errors."""
    sh = KIT_DIR / "demo_run.sh"
    # Drive it with `yes ""` to auto-press Enter at each pause; cap to
    # 25s and rely on the trap to clean up.
    # V.3 — bert doctor preflight runs by default and FAILs without
    # API keys. This test exercises the orchestrator structure, not the
    # doctor gate (which has its own coverage in _smoke_v_phase), so
    # bypass with --skip-doctor.
    proc = subprocess.Popen(
        ["bash", "-c", f"DEMO_PORT=5189 yes '' | timeout 25 {sh} --skip-doctor"],
        cwd=LAB_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        out, _ = proc.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    # Don't require success exit (timeout will kill it); just verify
    # we got far enough that uvicorn booted.
    assert "uvicorn pid=" in out, f"orchestrator didn't boot uvicorn:\n{out[:1500]}"
    # And that the cleanup trap fired
    assert "cleanup" in out.lower(), f"cleanup didn't run:\n{out[-800:]}"


def main() -> int:
    tests = [
        test_kit_dir_exists,
        test_orchestrator_exists_and_is_executable,
        test_orchestrator_has_safety_traps,
        test_orchestrator_invokes_bert_init_with_template,
        test_orchestrator_runs_bert_verify,
        test_storyboard_exists_with_both_cuts,
        test_narration_exists_with_both_cuts,
        test_readme_pre_flight_checklist,
        test_orchestrator_runs_clean_short_form,
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
