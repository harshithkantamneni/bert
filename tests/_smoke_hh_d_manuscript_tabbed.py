"""Smoke test for HH-D — Tabbed Manuscript with virtualized stream.

Collapses three reading surfaces — findings (/book), stream (/tide),
loom (/loom) — into one /manuscript surface with ?tab= routing. The
stream tab is virtualized via @tanstack/react-virtual so 300+ events
don't blow the DOM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


SHELL    = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "ManuscriptTabbed.tsx"
MANU     = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Manuscript.tsx"
TIDE     = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Tide.tsx"
LOOM     = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Loom.tsx"
APP      = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"
PKG      = LAB_ROOT / "bert" / "v4" / "package.json"


# ─── ManuscriptTabbed shell ───────────────────────────────────────


def test_shell_file_exists() -> None:
    assert SHELL.exists()


def test_shell_renders_three_tabs() -> None:
    """Tab bar has findings, stream, loom — the three reading
    surfaces being unified."""
    text = SHELL.read_text()
    for label in ("findings", "stream", "loom"):
        assert label in text


def test_shell_imports_three_bodies() -> None:
    """Each tab body is imported from its origin surface — no
    duplicated rendering logic."""
    text = SHELL.read_text()
    assert "import { FindingsBody }" in text
    assert "import { StreamBody }" in text
    assert "import { LoomBody }" in text


def test_shell_reads_tab_from_query_param() -> None:
    """Tab choice round-trips via ?tab=… so deep links work."""
    text = SHELL.read_text()
    assert "useSearchParams" in text
    assert 'params.get("tab")' in text


def test_shell_default_tab_findings() -> None:
    text = SHELL.read_text()
    assert "parseTab" in text
    assert 'return "findings"' in text


def test_shell_preserves_other_query_params_on_tab_switch() -> None:
    """Per the v3.5 quality answer — switching tabs must not drop
    ?id= or other params."""
    text = SHELL.read_text()
    assert "new URLSearchParams(params)" in text


def test_shell_aria_tablist_and_tabs() -> None:
    text = SHELL.read_text()
    assert 'role="tablist"' in text
    assert 'role="tab"' in text
    assert 'role="tabpanel"' in text
    assert "aria-selected" in text
    assert "aria-controls" in text


def test_shell_lab_indicator_in_header() -> None:
    """The PI should always know which lab they're reading."""
    text = SHELL.read_text()
    assert "useActiveLab" in text
    assert "labLabel" in text


def test_shell_tab_button_touch_targets() -> None:
    text = SHELL.read_text()
    assert "minHeight: 44" in text


# ─── Tide.tsx virtualization ──────────────────────────────────────


def test_tide_uses_window_virtualizer() -> None:
    text = TIDE.read_text()
    assert "useWindowVirtualizer" in text
    assert '@tanstack/react-virtual' in text


def test_tide_drops_animate_presence_for_virtualization() -> None:
    """AnimatePresence is incompatible with virtualization (rows
    unmount on scroll, triggering spurious exit anims). The
    fade-in for fresh rows is preserved via motion.li initial."""
    text = TIDE.read_text()
    assert "AnimatePresence" not in text


def test_tide_river_row_is_absolutely_positioned() -> None:
    """Virtualized rows position with translateY based on the
    virtualizer's start offset."""
    text = TIDE.read_text()
    assert 'position: "absolute"' in text
    assert "translateY" in text


def test_tide_uses_overscan() -> None:
    """A small overscan window keeps rows mounted just outside the
    viewport so quick scrolling stays smooth."""
    text = TIDE.read_text()
    assert "overscan:" in text


def test_tide_uses_measure_element() -> None:
    """Variable row heights need per-row measurement."""
    text = TIDE.read_text()
    assert "measureElement" in text


def test_tide_uses_get_item_key_by_id() -> None:
    """Stable keys via event.id so re-renders don't shuffle rows."""
    text = TIDE.read_text()
    assert "getItemKey" in text
    assert "visible[i]?.id" in text


def test_tide_exports_stream_body() -> None:
    text = TIDE.read_text()
    assert "export function StreamBody" in text


# ─── Manuscript.tsx FindingsBody ──────────────────────────────────


def test_manuscript_exports_findings_body() -> None:
    text = MANU.read_text()
    assert "export function FindingsBody" in text


def test_manuscript_tabbed_owns_outer_article() -> None:
    """ManuscriptTabbed wraps tab bodies in a single outer <article>;
    the legacy Manuscript wrapper is retired in HH-E. Verified by
    finding the <article> tag inside the shell rather than in the
    body file."""
    shell_text = SHELL.read_text()
    assert "<article" in shell_text


# ─── Loom.tsx LoomBody ────────────────────────────────────────────


def test_loom_exports_loom_body() -> None:
    text = LOOM.read_text()
    assert "export function LoomBody" in text


def test_loom_body_exported_without_default_wrapper() -> None:
    """Per HH-E retirement, Loom.tsx exposes only LoomBody (the
    outer article moved to ManuscriptTabbed)."""
    text = LOOM.read_text()
    assert "export function LoomBody" in text


# ─── App.tsx routing ──────────────────────────────────────────────


def test_app_mounts_manuscript_tabbed_at_slash_manuscript() -> None:
    text = APP.read_text()
    assert "ManuscriptTabbed" in text
    assert 'path="/manuscript"' in text


def test_app_keeps_book_route_for_backward_compat() -> None:
    """HH-E retires /book; until then it keeps working."""
    text = APP.read_text()
    assert 'path="/book"' in text


# ─── package.json ─────────────────────────────────────────────────


def test_react_virtual_in_deps() -> None:
    pkg = json.loads(PKG.read_text())
    deps = pkg.get("dependencies", {})
    assert "@tanstack/react-virtual" in deps, deps


def main() -> int:
    tests = [
        test_shell_file_exists,
        test_shell_renders_three_tabs,
        test_shell_imports_three_bodies,
        test_shell_reads_tab_from_query_param,
        test_shell_default_tab_findings,
        test_shell_preserves_other_query_params_on_tab_switch,
        test_shell_aria_tablist_and_tabs,
        test_shell_lab_indicator_in_header,
        test_shell_tab_button_touch_targets,
        test_tide_uses_window_virtualizer,
        test_tide_drops_animate_presence_for_virtualization,
        test_tide_river_row_is_absolutely_positioned,
        test_tide_uses_overscan,
        test_tide_uses_measure_element,
        test_tide_uses_get_item_key_by_id,
        test_tide_exports_stream_body,
        test_manuscript_exports_findings_body,
        test_manuscript_tabbed_owns_outer_article,
        test_loom_exports_loom_body,
        test_loom_body_exported_without_default_wrapper,
        test_app_mounts_manuscript_tabbed_at_slash_manuscript,
        test_app_keeps_book_route_for_backward_compat,
        test_react_virtual_in_deps,
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
