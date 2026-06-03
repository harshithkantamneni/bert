"""Smoke test for N.3: Lab Picker UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent


def test_active_lab_module_exists() -> None:
    p = LAB_ROOT / "bert" / "v4" / "src" / "api" / "activeLab.ts"
    assert p.exists()
    text = p.read_text()
    assert "useActiveLab" in text
    assert "bert:active-lab" in text  # localStorage key
    assert "withLab" in text
    assert "labQuery" in text


def test_active_lab_setter_invalidates_queries() -> None:
    """Switching lab must invalidate React Query cache."""
    text = (LAB_ROOT / "bert" / "v4" / "src" / "api" / "activeLab.ts").read_text()
    assert "qc.invalidateQueries()" in text or "invalidateQueries" in text


def test_hooks_thread_lab_query_param() -> None:
    text = (LAB_ROOT / "bert" / "v4" / "src" / "api" / "hooks.ts").read_text()
    # useLabStatus, useEvents, useAgents, useFindings all must include
    # _activeLab() in queryKey and _withLab in queryFn URL
    for needle in ("_activeLab()", "_withLab("):
        assert needle in text, f"hooks.ts missing {needle}"
    # All 4 routable hooks present
    for hook in ("useLabStatus", "useEvents", "useAgents", "useFindings"):
        assert hook in text


def test_lab_picker_renders_popover() -> None:
    text = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx").read_text()
    # Picker is now a button + listbox, not a static div
    assert 'role="listbox"' in text
    assert 'aria-haspopup="listbox"' in text
    assert "LabOption" in text
    assert "default (bert-lab)" in text


def test_lab_picker_imports_use_active_lab() -> None:
    text = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx").read_text()
    assert 'import { useActiveLab } from "../api/activeLab";' in text


def test_typescript_compiles_clean() -> None:
    result = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=LAB_ROOT / "bert" / "v4",
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"tsc failed:\n{result.stdout}\n{result.stderr}"
    )


def main() -> int:
    tests = [
        test_active_lab_module_exists,
        test_active_lab_setter_invalidates_queries,
        test_hooks_thread_lab_query_param,
        test_lab_picker_renders_popover,
        test_lab_picker_imports_use_active_lab,
        test_typescript_compiles_clean,
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
