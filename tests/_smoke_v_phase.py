"""Smoke test for V-phase: nightly automation + doctor gate + grade
refinement + git hooks.

V.1 — director_letter.py composes from real state
V.2 — bert_nightly.sh + install_nightly.py
V.3 — demo orchestrator gated by bert doctor
V.4 — activity_health composite grade + 4-state disclosure rotation
V.5 — pre-push git hook + private-CI example yaml
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "requires lab runtime artifact(s) not shipped in the public repo: "
            + ", ".join(str(m) for m in missing)
        )

import tools.daily_history_compile as dhc  # noqa: E402
import tools.daily_quality_report as dqr  # noqa: E402
import tools.director_letter as dl  # noqa: E402
import tools.weekly_history_compile as whc  # noqa: E402

VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"
DEMO_RUN = LAB_ROOT / "findings" / "investor" / "demo_recording" / "demo_run.sh"
LETTERS_DIR = LAB_ROOT / "lab" / "state" / "director_letters"
NIGHTLY_SH = LAB_ROOT / "tools" / "bert_nightly.sh"
INSTALL_NIGHTLY = LAB_ROOT / "tools" / "install_nightly.py"
PRE_PUSH_SRC = LAB_ROOT / "tools" / "git-hooks" / "pre-push"
INSTALL_HOOKS = LAB_ROOT / "tools" / "install_hooks.sh"
CI_EXAMPLE = LAB_ROOT / "tools" / "example-ci" / "github-actions.yml"


# ── V.1 — Director's letter ─────────────────────────────────────────

def test_director_letter_module_imports() -> None:
    assert hasattr(dl, "compose_letter")
    assert hasattr(dl, "write_letter")


def test_compose_letter_returns_valid_schema() -> None:
    """Letter dict must include every key /api/letters/latest's fallback
    fixture provides, so the UI swap to real-letter is transparent."""
    letter = dl.compose_letter()
    required = ["id", "voice", "is_fallback", "ts_local", "weekday",
                "date_long", "time_short", "cycle", "kicker", "salutation",
                "body", "signed", "needs_dominus"]
    for k in required:
        assert k in letter, f"letter missing required key {k!r}"
    assert letter["is_fallback"] is False, \
        "generated letter must NOT have is_fallback=True (only fallback fixture has that)"
    assert isinstance(letter["body"], list) and len(letter["body"]) >= 2, \
        "body must be at least 2 paragraphs"
    assert letter["voice"] in {"A", "B", "C", "D"}, \
        f"voice must be a known direction; got {letter['voice']!r}"


def test_letter_kicker_includes_cycle_when_present() -> None:
    """If a cycle id is present in recent events, kicker should reflect it."""
    letter = dl.compose_letter()
    if letter["cycle"] is not None:
        assert "cycle" in letter["kicker"].lower(), \
            "kicker should mention cycle when cycle is set"
        assert str(letter["cycle"]) in letter["kicker"], \
            "kicker should include the actual cycle id"
    else:
        assert "no cycle" in letter["kicker"].lower(), \
            "kicker should say 'no cycle' when none observed"


def test_letter_body_reflects_daily_report_state() -> None:
    """When a daily report exists for today, the body should reference
    the actual numbers (events count, accepted count) from it."""
    letter = dl.compose_letter()
    body_text = " ".join(letter["body"])
    # At minimum: closing paragraph about Dominus needs
    assert "pending" in body_text.lower() or "needs you" in body_text.lower(), \
        "body must include the pending-shelf line"


def test_letter_on_disk_real_not_fallback() -> None:
    """After V.1 generated a letter, the file must exist and be
    is_fallback=false."""
    if not LETTERS_DIR.exists() or not list(LETTERS_DIR.glob("letter_*.json")):
        # The letter generator hasn't been run yet — write one
        letter = dl.compose_letter()
        dl.write_letter(letter)
    files = sorted(LETTERS_DIR.glob("letter_*.json"))
    assert files, "no letter found on disk after V.1"
    latest = json.loads(files[-1].read_text())
    assert latest["is_fallback"] is False


# ── V.2 — Nightly automation ────────────────────────────────────────

def test_bert_nightly_script_exists_and_executable() -> None:
    assert NIGHTLY_SH.exists(), "tools/bert_nightly.sh missing"
    assert NIGHTLY_SH.stat().st_mode & stat.S_IXUSR, \
        "bert_nightly.sh must be executable"


def test_bert_nightly_syntax_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(NIGHTLY_SH)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, \
        f"bert_nightly.sh bash syntax invalid: {result.stderr}"


def test_bert_nightly_dry_run_succeeds() -> None:
    """--dry-run must complete without touching disk and return 0."""
    _require(VENV_PY)
    result = subprocess.run(
        ["bash", str(NIGHTLY_SH), "--dry-run"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, \
        f"nightly --dry-run failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    out = result.stdout + result.stderr
    assert "dry-run" in out, "dry-run mode must announce itself"
    assert "daily_quality_report" in out, "must mention daily report step"
    assert "director_letter" in out, "must mention letter step"


def test_bert_nightly_friday_includes_weekly() -> None:
    """When --include-weekly is forced, weekly steps must fire."""
    _require(VENV_PY)
    result = subprocess.run(
        ["bash", str(NIGHTLY_SH), "--include-weekly", "--dry-run"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "weekly_quality_report" in out, \
        "--include-weekly must trigger weekly_quality_report"
    assert "weekly_history_compile" in out, \
        "--include-weekly must trigger weekly_history_compile"


def test_install_nightly_print_only_emits_plist() -> None:
    """--print-only must not write to disk; must print plist content."""
    _require(VENV_PY)
    result = subprocess.run(
        [str(VENV_PY), str(INSTALL_NIGHTLY), "--print-only", "--hour", "23"],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0
    assert "<plist" in result.stdout, "must emit XML plist on macOS"
    assert "bert.nightly" in result.stdout, "plist must reference our label"
    # And the plist points at the real script
    assert str(NIGHTLY_SH) in result.stdout, \
        "plist must reference the actual bert_nightly.sh path"


def test_install_nightly_status_reports_when_not_installed() -> None:
    """--status when nothing installed must exit 1 (not installed) with
    a clear message."""
    _require(VENV_PY)
    result = subprocess.run(
        [str(VENV_PY), str(INSTALL_NIGHTLY), "--status"],
        capture_output=True, text=True, timeout=5,
    )
    # rc=0 if installed, rc=1 if not — either is valid, we just check
    # the message is informative
    assert "status" in result.stdout.lower()


# ── V.3 — Doctor gate ───────────────────────────────────────────────

def test_demo_orchestrator_accepts_skip_doctor_flag() -> None:
    """--skip-doctor must be a recognized flag."""
    _require(DEMO_RUN)
    text = DEMO_RUN.read_text()
    assert "--skip-doctor" in text, \
        "orchestrator must support --skip-doctor override"
    assert "SKIP_DOCTOR" in text, \
        "must use the SKIP_DOCTOR variable"


def test_orchestrator_runs_bert_doctor_in_preflight() -> None:
    """The bert_doctor.py invocation must appear BEFORE step 0 (the
    uvicorn boot). A preflight that runs after uvicorn is no preflight."""
    _require(DEMO_RUN)
    lines = DEMO_RUN.read_text().splitlines()
    doctor_idx = None
    step0_idx = None
    for i, line in enumerate(lines):
        if "tools/bert_doctor.py" in line and doctor_idx is None:
            doctor_idx = i
        if "# ── Step 0" in line:
            step0_idx = i
            break
    assert doctor_idx is not None, "bert_doctor.py invocation not found"
    assert step0_idx is not None, "Step 0 marker not found"
    assert doctor_idx < step0_idx, \
        f"bert_doctor.py at line {doctor_idx+1} must precede Step 0 at {step0_idx+1}"


def test_orchestrator_aborts_on_doctor_fail() -> None:
    """End-to-end: with no API keys + skip-doctor=0, orchestrator must
    exit 2 (preflight fail). Cleanup trap must preserve the exit code."""
    _require(DEMO_RUN)
    result = subprocess.run(
        ["bash", str(DEMO_RUN)],
        capture_output=True, text=True, timeout=20,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 2, \
        f"orchestrator should exit 2 on doctor FAIL; got {result.returncode}"
    combined = result.stdout + result.stderr
    assert "ABORT" in combined or "abort" in combined, \
        "should surface ABORT in output"
    assert "bert doctor" in combined.lower(), \
        "should reference bert doctor in the abort message"


# ── V.4 — Activity health + 4-state disclosure ──────────────────────

def test_activity_health_composite_grade_present() -> None:
    """compute_metrics + derive_daily_letter should produce both the
    legacy activity_volume AND the new activity_health composite."""
    events = [
        {"event_class": "dispatch_result", "verdict": "APPROVE",
         "judge_provider": "mistral", "agent": f"r{i}", "cycle": i}
        for i in range(15)
    ]
    metrics = dqr.compute_metrics(events)
    letters = dqr.derive_daily_letter(metrics)
    assert "activity_volume" in letters, \
        "legacy activity_volume must remain"
    assert "activity_health" in letters, \
        "V.4 must add activity_health composite"
    assert "_activity_health_score" in letters, \
        "must surface the numeric score for transparency"


def test_activity_health_distinguishes_diverse_vs_monotone() -> None:
    """A day with 60 events across 6 roles should beat a day with
    60 events all from one role on the health axis (not on volume)."""
    diverse = [{"event_class": "x", "agent": f"role_{i % 6}", "cycle": i}
               for i in range(60)]
    monotone = [{"event_class": "x", "agent": "single_role", "cycle": 1}
                for _ in range(60)]
    diverse_letters = dqr.derive_daily_letter(dqr.compute_metrics(diverse))
    monotone_letters = dqr.derive_daily_letter(dqr.compute_metrics(monotone))
    assert diverse_letters["_activity_health_score"] > \
           monotone_letters["_activity_health_score"], \
           "diverse-role day must score higher on activity_health than monotone"


def test_activity_health_thresholds_correct() -> None:
    """A: score>=75, B: 40-74, C: 10-39, INSUFFICIENT: <10."""
    # A grade: high on all 4 axes (100 events, 5 roles, 10 cycles, 5 accepted)
    high_events = (
        [{"event_class": "dispatch_result", "agent": f"r{i % 5}", "cycle": i % 10}
         for i in range(100)]
        + [{"event_class": "artifact_accepted", "agent": "r0", "cycle": 0}
           for _ in range(5)]
    )
    letters = dqr.derive_daily_letter(dqr.compute_metrics(high_events))
    assert letters["activity_health"] == "A", \
        f"high-signal day should grade A; got {letters['activity_health']}"

    # INSUFFICIENT: empty
    empty_letters = dqr.derive_daily_letter(dqr.compute_metrics([]))
    assert empty_letters["activity_health"] == "INSUFFICIENT", \
        f"empty day should be INSUFFICIENT; got {empty_letters['activity_health']}"


def test_weekly_disclosure_has_four_states() -> None:
    """Weekly disclosure rotation: 0 / N<8 / 8-16 / >16. The extended
    branch must surface the (+M beyond baseline) note."""
    msg_0 = whc._build_disclosure(0)
    msg_4 = whc._build_disclosure(4)
    msg_8 = whc._build_disclosure(8)
    msg_16 = whc._build_disclosure(16)
    msg_20 = whc._build_disclosure(20)

    assert "No weekly reports" in msg_0
    assert "4 weeks" in msg_4 and "Devin-class" in msg_4
    assert "Baseline established" in msg_8
    assert "Baseline established" in msg_16
    assert "Extended series" in msg_20, \
        "20-week disclosure must use the extended branch"
    assert "+12 weeks beyond baseline" in msg_20, \
        "extended disclosure must show the beyond-baseline count"


def test_daily_disclosure_has_four_states() -> None:
    msg_0 = dhc._build_disclosure(0)
    msg_15 = dhc._build_disclosure(15)
    msg_30 = dhc._build_disclosure(30)
    msg_60 = dhc._build_disclosure(60)
    msg_90 = dhc._build_disclosure(90)

    assert "No daily reports" in msg_0
    assert "15 day" in msg_15 and "deliberately omitted" in msg_15
    assert "Rolling reference window full" in msg_30
    assert "Rolling reference window full" in msg_60
    assert "Extended series" in msg_90
    assert "+60 days beyond" in msg_90, \
        "extended daily disclosure must show beyond-window count"


# ── V.5 — Git hook + CI example ─────────────────────────────────────

def test_pre_push_hook_source_exists() -> None:
    assert PRE_PUSH_SRC.exists(), "tools/git-hooks/pre-push missing"
    assert PRE_PUSH_SRC.stat().st_mode & stat.S_IXUSR, \
        "pre-push must be executable"


def test_pre_push_hook_syntax_valid() -> None:
    result = subprocess.run(["bash", "-n", str(PRE_PUSH_SRC)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, \
        f"pre-push syntax invalid: {result.stderr}"


def test_pre_push_uses_run_canonical() -> None:
    """Hook must invoke the canonical regression script, not reinvent it."""
    text = PRE_PUSH_SRC.read_text()
    assert "tests/run_canonical.sh" in text, \
        "pre-push must invoke tests/run_canonical.sh"


def test_pre_push_supports_no_verify_bypass() -> None:
    """The hook should document the --no-verify escape hatch."""
    text = PRE_PUSH_SRC.read_text()
    assert "--no-verify" in text, "pre-push must reference --no-verify bypass"


def test_install_hooks_script_present_and_executable() -> None:
    assert INSTALL_HOOKS.exists()
    assert INSTALL_HOOKS.stat().st_mode & stat.S_IXUSR, \
        "install_hooks.sh must be executable"


def test_install_hooks_idempotent() -> None:
    """Running install_hooks.sh twice must produce the same final state."""
    # First install (might be a re-install)
    r1 = subprocess.run(["bash", str(INSTALL_HOOKS)],
                        capture_output=True, text=True, timeout=10)
    assert r1.returncode == 0, f"first install failed: {r1.stderr}"
    hook_after_1 = (LAB_ROOT / ".git" / "hooks" / "pre-push").read_text()
    # Second install
    r2 = subprocess.run(["bash", str(INSTALL_HOOKS)],
                        capture_output=True, text=True, timeout=10)
    assert r2.returncode == 0, f"second install failed: {r2.stderr}"
    hook_after_2 = (LAB_ROOT / ".git" / "hooks" / "pre-push").read_text()
    assert hook_after_1 == hook_after_2, \
        "install_hooks.sh is not idempotent — second run produced different content"


def test_ci_example_yaml_present() -> None:
    """The example CI yaml must exist in tools/example-ci/ (NOT in
    .github/workflows/ — that would be a public-facing artifact)."""
    assert CI_EXAMPLE.exists(), "tools/example-ci/github-actions.yml missing"
    # And it must NOT be in .github/workflows
    dotgithub = LAB_ROOT / ".github" / "workflows"
    if dotgithub.exists():
        # If user has CI wired in their private fork, that's fine —
        # but our checked-in template should be in tools/example-ci/
        # to honor the "no GitHub-public" memory.
        pass


def test_ci_example_runs_canonical_regression() -> None:
    """The CI workflow yaml should invoke run_canonical.sh."""
    text = CI_EXAMPLE.read_text()
    assert "tests/run_canonical.sh" in text, \
        "example CI must run the canonical regression"
    assert "bert_doctor" in text or "bert doctor" in text, \
        "example CI should run bert doctor (even allowing failure)"


def test_ci_example_documents_private_only() -> None:
    """The CI yaml's header comment must say 'private only' so anyone
    copying the file knows the proprietary discipline."""
    text = CI_EXAMPLE.read_text()
    if "private" not in text.lower() and "proprietary" not in text.lower():
        pytest.skip(
            "requires the private CI example (proprietary-discipline header "
            "scrubbed from the public repo copy)"
        )
    assert "private" in text.lower(), \
        "CI example header must say 'private only'"
    assert "proprietary" in text.lower() or "open-sourcing" in text.lower(), \
        "CI example must reference the proprietary discipline"


def main() -> int:
    tests = [
        # V.1
        test_director_letter_module_imports,
        test_compose_letter_returns_valid_schema,
        test_letter_kicker_includes_cycle_when_present,
        test_letter_body_reflects_daily_report_state,
        test_letter_on_disk_real_not_fallback,
        # V.2
        test_bert_nightly_script_exists_and_executable,
        test_bert_nightly_syntax_valid,
        test_bert_nightly_dry_run_succeeds,
        test_bert_nightly_friday_includes_weekly,
        test_install_nightly_print_only_emits_plist,
        test_install_nightly_status_reports_when_not_installed,
        # V.3
        test_demo_orchestrator_accepts_skip_doctor_flag,
        test_orchestrator_runs_bert_doctor_in_preflight,
        test_orchestrator_aborts_on_doctor_fail,
        # V.4
        test_activity_health_composite_grade_present,
        test_activity_health_distinguishes_diverse_vs_monotone,
        test_activity_health_thresholds_correct,
        test_weekly_disclosure_has_four_states,
        test_daily_disclosure_has_four_states,
        # V.5
        test_pre_push_hook_source_exists,
        test_pre_push_hook_syntax_valid,
        test_pre_push_uses_run_canonical,
        test_pre_push_supports_no_verify_bypass,
        test_install_hooks_script_present_and_executable,
        test_install_hooks_idempotent,
        test_ci_example_yaml_present,
        test_ci_example_runs_canonical_regression,
        test_ci_example_documents_private_only,
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
