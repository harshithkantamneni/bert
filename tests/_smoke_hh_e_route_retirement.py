"""Smoke test for HH-E — QueryPreservingRedirect + route retirement.

Retires /mission, /meeting, /tide, /loom, /book, /book/:id by
redirecting each to its new home. Mission.tsx + Meeting.tsx files
are removed. Tide/Loom/Manuscript files keep their body exports
(consumed by ManuscriptTabbed) but lose the unused outer wrappers.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


QPR     = LAB_ROOT / "bert" / "v4" / "src" / "components" / "QueryPreservingRedirect.tsx"
APP     = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"
MISSION = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Mission.tsx"
MEETING = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Meeting.tsx"
TIDE    = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Tide.tsx"
LOOM    = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Loom.tsx"
MANU    = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Manuscript.tsx"
KBD     = LAB_ROOT / "bert" / "v4" / "src" / "hooks" / "useKeyboardNav.ts"
HELP    = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "KeyboardHelp.tsx"


# ─── QueryPreservingRedirect ──────────────────────────────────────


def test_qpr_file_exists() -> None:
    assert QPR.exists()


def test_qpr_uses_navigate_with_replace() -> None:
    """replace=true so browser back doesn't bounce the user back
    onto a retired URL."""
    text = QPR.read_text()
    assert "<Navigate to={target} replace />" in text


def test_qpr_resolves_param_tokens() -> None:
    """`:id` tokens in target template fill from useParams() so
    deep links (e.g. /book/abc → /manuscript/abc) survive."""
    text = QPR.read_text()
    assert "useParams" in text
    assert ":${key}" in text
    assert "encodeURIComponent" in text


def test_qpr_preserves_search_by_default() -> None:
    text = QPR.read_text()
    assert "preserveSearch = true" in text
    assert "URLSearchParams(preserveSearch ? location.search" in text


def test_qpr_append_search_merges_without_clobbering() -> None:
    """appendSearch only adds keys that don't already exist."""
    text = QPR.read_text()
    assert "appendSearch" in text
    assert "if (!search.has(k))" in text


# ─── App.tsx redirect routes ──────────────────────────────────────


def test_app_imports_qpr() -> None:
    text = APP.read_text()
    assert "QueryPreservingRedirect" in text


def test_app_redirects_mission_to_home() -> None:
    text = APP.read_text()
    assert 'path="/mission"' in text
    # Look for the redirect block
    assert "<QueryPreservingRedirect to=\"/\" />" in text


def test_app_redirects_meeting_to_home() -> None:
    text = APP.read_text()
    assert 'path="/meeting"' in text


def test_app_redirects_tide_to_manuscript_stream() -> None:
    text = APP.read_text()
    assert 'path="/tide"' in text
    assert 'tab: "stream"' in text


def test_app_redirects_loom_to_manuscript_loom() -> None:
    text = APP.read_text()
    assert 'path="/loom"' in text
    assert 'tab: "loom"' in text


def test_app_redirects_book_to_manuscript() -> None:
    text = APP.read_text()
    assert 'path="/book"' in text
    assert 'path="/book/:id"' in text
    assert 'to="/manuscript/:id"' in text


def test_app_no_longer_imports_retired_surfaces() -> None:
    text = APP.read_text()
    # These were the route-level component imports; redirect-only
    # routes shouldn't need them anymore
    assert 'from "./surfaces/Mission"' not in text
    assert 'from "./surfaces/Meeting"' not in text
    assert 'from "./surfaces/Tide"' not in text
    assert 'from "./surfaces/Loom"' not in text
    # Manuscript file still exists (FindingsBody export) but the
    # outer wrapper component should no longer be imported here.
    # Only ManuscriptTabbed should be imported.
    assert "import { Manuscript }" not in text


# ─── Deleted surface files ────────────────────────────────────────


def test_mission_file_deleted() -> None:
    assert not MISSION.exists()


def test_meeting_file_deleted() -> None:
    assert not MEETING.exists()


# ─── Body-only surface files ──────────────────────────────────────


def test_tide_keeps_stream_body_export() -> None:
    text = TIDE.read_text()
    assert "export function StreamBody" in text


def test_tide_no_longer_exports_default_wrapper() -> None:
    """The standalone <Tide /> wrapper is removed; only StreamBody
    remains, consumed by ManuscriptTabbed."""
    text = TIDE.read_text()
    assert "export function Tide(" not in text


def test_loom_keeps_loom_body_export() -> None:
    text = LOOM.read_text()
    assert "export function LoomBody" in text


def test_loom_no_longer_exports_default_wrapper() -> None:
    text = LOOM.read_text()
    assert "export function Loom(" not in text


def test_manuscript_keeps_findings_body_export() -> None:
    text = MANU.read_text()
    assert "export function FindingsBody" in text


def test_manuscript_no_longer_exports_default_wrapper() -> None:
    text = MANU.read_text()
    assert "export function Manuscript(" not in text


# ─── Keyboard navigation updates ──────────────────────────────────


def test_keyboard_nav_targets_updated() -> None:
    """Shortcut targets point at the new routes."""
    text = KBD.read_text()
    # No retired paths
    assert '"/meeting"' not in text
    assert '"/tide"' not in text
    assert '"/loom"' not in text
    assert '"/book"' not in text
    assert '"/mission"' not in text
    # New paths
    assert '"/manuscript"' in text
    assert '"/manuscript?tab=stream"' in text
    assert '"/manuscript?tab=loom"' in text


def test_keyboard_help_labels_updated() -> None:
    text = HELP.read_text()
    assert "manuscript · stream" in text
    assert "manuscript · findings" in text
    assert "manuscript · loom" in text
    # No retired labels
    assert "the meeting" not in text
    assert "the tide" not in text


def main() -> int:
    tests = [
        test_qpr_file_exists,
        test_qpr_uses_navigate_with_replace,
        test_qpr_resolves_param_tokens,
        test_qpr_preserves_search_by_default,
        test_qpr_append_search_merges_without_clobbering,
        test_app_imports_qpr,
        test_app_redirects_mission_to_home,
        test_app_redirects_meeting_to_home,
        test_app_redirects_tide_to_manuscript_stream,
        test_app_redirects_loom_to_manuscript_loom,
        test_app_redirects_book_to_manuscript,
        test_app_no_longer_imports_retired_surfaces,
        test_mission_file_deleted,
        test_meeting_file_deleted,
        test_tide_keeps_stream_body_export,
        test_tide_no_longer_exports_default_wrapper,
        test_loom_keeps_loom_body_export,
        test_loom_no_longer_exports_default_wrapper,
        test_manuscript_keeps_findings_body_export,
        test_manuscript_no_longer_exports_default_wrapper,
        test_keyboard_nav_targets_updated,
        test_keyboard_help_labels_updated,
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
