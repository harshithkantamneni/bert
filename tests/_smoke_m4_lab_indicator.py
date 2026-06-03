"""Smoke test for M.4: Lab Indicator (UX layer on L.4)."""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
FIRSTLIGHT = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx"


def test_lab_indicator_renders_in_firstlight() -> None:
    text = FIRSTLIGHT.read_text()
    assert "<LabIndicator />" in text
    assert "function LabIndicator" in text


def test_lab_indicator_uses_useLabs_hook() -> None:
    text = FIRSTLIGHT.read_text()
    assert "useLabs" in text


def test_lab_indicator_distinguishes_default_vs_routed() -> None:
    text = FIRSTLIGHT.read_text()
    assert "viewing: bert-lab (default)" in text
    assert "routed" in text


def test_lab_indicator_is_clickable_picker() -> None:
    """Post-N.3: picker is a button with aria-haspopup, not a hover tooltip.
    M.4's original 'tooltip' semantic was replaced by clickable popover."""
    text = FIRSTLIGHT.read_text()
    fn_start = text.index("function LabIndicator")
    fn_end = text.index("function LabOption", fn_start)
    fn_text = text[fn_start:fn_end]
    assert "title=" in fn_text  # button still has 'Click to switch labs' tooltip
    assert 'aria-haspopup="listbox"' in fn_text
    assert "Click to switch labs" in fn_text


def test_labs_response_type_exists() -> None:
    client = LAB_ROOT / "bert" / "v4" / "src" / "api" / "client.ts"
    text = client.read_text()
    assert "export interface LabsResponse" in text
    assert "export interface ScaffoldedLab" in text


def main() -> int:
    tests = [
        test_lab_indicator_renders_in_firstlight,
        test_lab_indicator_uses_useLabs_hook,
        test_lab_indicator_distinguishes_default_vs_routed,
        test_lab_indicator_is_clickable_picker,
        test_labs_response_type_exists,
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
