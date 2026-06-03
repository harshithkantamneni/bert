"""Smoke test for H3 — L-04 VSM tags + L-05 OODA markers + L-06 /now page.

Per FINAL_implementation_plan_2026-05-07.md §5.3.
"""

import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent


def test_oods_markers_appended_to_4_role_prompts() -> None:
    """L-05: 4 of 5 role prompts gain OODA marker section (strategist
    deactivated per H-BUILD-01)."""
    expected = ("director", "researcher", "implementer", "evaluator")
    for role in expected:
        text = (LAB_ROOT / "prompts" / f"{role}.md").read_text()
        assert "OODA: observe" in text and "OODA: orient" in text, (
            f"OODA markers missing from {role}.md"
        )
    # Strategist should NOT have OODA additions (deactivated)
    strategist = (LAB_ROOT / "prompts" / "strategist.md").read_text()
    assert "OODA: observe" not in strategist, (
        "Strategist should not have OODA section (deactivated per H-BUILD-01)"
    )


def test_vsm_system_tags_in_role_prompts() -> None:
    """L-04: each active role declares its VSM system tag."""
    expected_systems = {
        "director": ("S3", "System 3"),
        "researcher": ("S4", "System 4", "S4 (intelligence"),
        "implementer": ("S1", "System 1", "S1 (operations"),
        "evaluator": ("S2", "System 2", "S2 (coordination"),
    }
    for role, candidates in expected_systems.items():
        text = (LAB_ROOT / "prompts" / f"{role}.md").read_text()
        if not any(c in text for c in candidates):
            raise AssertionError(
                f"VSM system tag missing from {role}.md (looked for {candidates})"
            )


def test_now_page_generation() -> None:
    """L-06: tools/generate_now_page.py produces lab/state/now.md."""
    result = subprocess.run(
        [str(LAB_ROOT / ".venv" / "bin" / "python"),
         str(LAB_ROOT / "tools" / "generate_now_page.py")],
        capture_output=True, text=True, timeout=30, cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, f"now-page generation failed: {result.stderr}"
    now_md = LAB_ROOT / "lab" / "state" / "now.md"
    assert now_md.exists()
    text = now_md.read_text()
    # Verify Mandala-format 5-ring VSM structure
    for ring in ("S5", "S4", "S3", "S2", "S1"):
        assert ring in text, f"VSM ring {ring} missing from now.md"
    assert "Pattern catalogue" in text
    assert "Seasoning queue" in text


def test_cycle_recognition_addition_in_researcher_prompt() -> None:
    """Day 7 work — verify cycle-recognition section in researcher.md
    is APPENDED (near end of file)."""
    text = (LAB_ROOT / "prompts" / "researcher.md").read_text()
    pos = text.find("## Cycle-recognition revival path")
    assert pos > 0
    # Verify it's near the END of the file (last 50%, not middle)
    relative = pos / len(text)
    assert relative > 0.4, f"cycle-recognition at {relative:.0%}; should be near end"


def main() -> int:
    tests = [
        test_oods_markers_appended_to_4_role_prompts,
        test_vsm_system_tags_in_role_prompts,
        test_now_page_generation,
        test_cycle_recognition_addition_in_researcher_prompt,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
