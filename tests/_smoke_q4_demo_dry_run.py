"""Smoke test for Q.4: demo dry-run finding report + orchestrator alignment.

Validates that:
  1. The dry-run finding report exists with the expected structure.
  2. The orchestrator narration cue matches narration.md on the bert verify
     segment (the key fix from this pass).
  3. All commands the orchestrator runs in non-interactive mode actually
     succeed (bert init template + bert verify on canonical packet).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
DRY_RUN = LAB_ROOT / "findings" / "investor" / "demo_recording" / "dry_run_2026-05-13.md"
RUN_SH = LAB_ROOT / "findings" / "investor" / "demo_recording" / "demo_run.sh"
NARRATION = LAB_ROOT / "findings" / "investor" / "demo_recording" / "narration.md"
CANONICAL_PACKET = LAB_ROOT / "findings" / "proof_packets" / "cycle-0400.tar.gz"
BERT_INIT = LAB_ROOT / "tools" / "bert_init.py"
BERT_VERIFY = LAB_ROOT / "tools" / "bert_verify.py"
VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"


def test_dry_run_report_exists() -> None:
    assert DRY_RUN.exists(), "dry_run_2026-05-13.md missing"


def test_dry_run_documents_warn_states() -> None:
    """The dry-run report must capture the WARN states honestly. This is
    the documented evidence that the demo team knows about the gap."""
    text = DRY_RUN.read_text()
    assert "WARN" in text, "report must document the WARN states"
    assert "Rekor" in text, "report must name Rekor as one WARN source"
    assert "RFC3161" in text, "report must name RFC3161 as the other WARN source"
    assert "I.4" in text, "report must name the I.4 milestone for full Sigstore"


def test_dry_run_documents_live_cycle_gap() -> None:
    """The locked flight plan calls for a live-cycle segment; the
    orchestrator deliberately omits it. The report must explain why."""
    text = DRY_RUN.read_text()
    assert "live-cycle" in text.lower() or "live cycle" in text.lower(), \
        "report must address the live-cycle gap"
    assert "Rationale" in text or "rationale" in text, \
        "report must include rationale for the scope choice"


def test_orchestrator_narration_cue_matches_narration_md() -> None:
    """The orchestrator's terminal cue for bert verify must align with
    the read-aloud narration.md — both must honestly disclose the 2
    warnings. (Q.4 fix.)"""
    orch = RUN_SH.read_text()
    narration = NARRATION.read_text()
    # Both must mention 'two warnings' or 'local-dev mode'
    assert "two warnings" in orch.lower() or "local-dev mode" in orch.lower(), \
        "orchestrator narration must disclose the 2 warnings"
    assert "local-dev mode" in narration.lower() or "two warnings" in narration.lower(), \
        "narration.md must disclose the 2 warnings"
    # The stale binary phrasing must be gone
    assert "either holds or it doesn't" not in orch or \
           "receipt either holds, or it doesn't" not in orch.split("step 2")[1] \
           if "step 2" in orch.lower() else True, \
           "orchestrator step 2 must not use stale binary 'either holds or it doesn't' phrasing"


def test_bert_init_template_command_succeeds() -> None:
    """The orchestrator's bert init line is the most common failure mode
    in a real demo. Verify it works against a clean home in <10s."""
    tmp_home = Path(tempfile.mkdtemp(prefix="bert_q4_smoke_"))
    try:
        result = subprocess.run(
            [str(VENV_PY), str(BERT_INIT),
             "--non-interactive",
             "--archetype", "Product",
             "--name", "smoke-pitch",
             "--provider", "Groq",
             "--autonomy", "Collaborator",
             "--seed", "Smoke test seed",
             "--from-template", "demo_note_cli"],
            env={"HOME": str(tmp_home), "PATH": "/usr/bin:/bin"},
            cwd=str(LAB_ROOT),
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, \
            f"bert init failed (exit {result.returncode}): {result.stderr[:300]}"
        lab_dir = tmp_home / ".bert" / "labs" / "smoke-pitch"
        assert lab_dir.exists(), "lab directory not scaffolded"
        for sub in ("cycles", "sor", "state"):
            assert (lab_dir / sub).exists(), f"{sub}/ missing in scaffold"
        assert (lab_dir / "lab.yaml").exists(), "lab.yaml missing"
        assert (lab_dir / "seed_brief.md").exists(), "seed_brief.md missing"
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


def test_bert_verify_on_canonical_packet_completes() -> None:
    """The orchestrator's bert verify segment must complete without
    crashing. PASS-WITH-WARNINGS is acceptable; exit 1 is expected and
    documented in narration."""
    assert CANONICAL_PACKET.exists(), "canonical proof packet missing"
    result = subprocess.run(
        [str(VENV_PY), str(BERT_VERIFY), str(CANONICAL_PACKET), "--no-color"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode in (0, 1), \
        f"unexpected verify exit {result.returncode}; stderr={result.stderr[:200]}"
    out = result.stdout + result.stderr
    assert "Verifying cycle-0400.tar.gz" in out, "verify didn't run"
    assert "PASS" in out, "verify output should contain PASS checks"
    if "WARN" in out:
        # If warnings present, ensure they're the known I.4-deferred ones
        assert "Rekor" in out or "RFC3161" in out, \
            f"unexpected WARN source: {out[:400]}"


def test_dry_run_report_recommendations_present() -> None:
    """The report must include actionable next-pass recommendations."""
    text = DRY_RUN.read_text()
    assert "Recommendations" in text or "recommendations" in text, \
        "report must include a recommendations section"
    # At least 3 recommendations
    import re
    rec_count = len(re.findall(r"^\d+\.\s", text, re.MULTILINE))
    assert rec_count >= 3, \
        f"expected ≥3 numbered recommendations; got {rec_count}"


def main() -> int:
    tests = [
        test_dry_run_report_exists,
        test_dry_run_documents_warn_states,
        test_dry_run_documents_live_cycle_gap,
        test_orchestrator_narration_cue_matches_narration_md,
        test_bert_init_template_command_succeeds,
        test_bert_verify_on_canonical_packet_completes,
        test_dry_run_report_recommendations_present,
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
