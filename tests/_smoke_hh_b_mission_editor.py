"""Smoke test for HH-B — MissionEditor inline on Home.

Replaces the standalone /mission route. The PI edits seed_brief.md
in-place on Home between the director's letter and the pulse strip.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


EDITOR = LAB_ROOT / "bert" / "v4" / "src" / "components" / "MissionEditor.tsx"
FIRST_LIGHT = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx"


def test_editor_file_exists() -> None:
    assert EDITOR.exists()


def test_editor_has_edit_and_preview_tabs() -> None:
    text = EDITOR.read_text()
    assert '"edit"' in text
    assert '"preview"' in text
    assert "TabButton" in text


def test_editor_renders_markdown_preview() -> None:
    """Per v3.5 quality answer: markdown preview, not 'plain textarea
    for prototype'."""
    text = EDITOR.read_text()
    assert "ReactMarkdown" in text
    assert "remarkGfm" in text


def test_editor_uses_seed_brief_endpoint() -> None:
    text = EDITOR.read_text()
    assert "/api/seed-brief" in text
    assert "SeedBriefRead" in text
    assert "SeedBriefWriteResp" in text


def test_editor_routes_via_lab_query() -> None:
    text = EDITOR.read_text()
    assert "labQuery(activeLab)" in text
    assert "useActiveLab" in text


def test_editor_handles_409_conflict() -> None:
    text = EDITOR.read_text()
    assert "409" in text
    assert "conflict" in text.lower()
    # Refresh / override CTAs
    assert "refresh" in text.lower()
    assert "override" in text.lower()


def test_editor_optimistic_concurrency_via_expected_mtime() -> None:
    text = EDITOR.read_text()
    assert "expected_mtime" in text
    assert "serverMtime" in text


def test_editor_default_expanded_when_brief_empty() -> None:
    text = EDITOR.read_text()
    # Heuristic: setExpanded based on content trim length
    assert "trim().length === 0" in text or \
           "content.trim().length === 0" in text


def test_editor_unsaved_changes_beforeunload_guard() -> None:
    """Browser back/refresh/close prompts when draft is dirty."""
    text = EDITOR.read_text()
    assert "beforeunload" in text
    assert "dirty" in text


def test_editor_touch_targets_44px() -> None:
    text = EDITOR.read_text()
    # Tab buttons + save buttons + collapse header all ≥44
    assert text.count("minHeight: 44") >= 2
    # Collapse header is 56 (bigger row); accept either
    assert "minHeight: 56" in text or text.count("minHeight: 44") >= 3


def test_editor_shows_dirty_indicator() -> None:
    text = EDITOR.read_text()
    assert "dirty" in text
    # "unsaved" label appears in the collapsed header
    assert "unsaved" in text.lower()


def test_editor_soft_warns_on_long_missions() -> None:
    text = EDITOR.read_text()
    assert "8000" in text  # soft limit
    assert "overSoftLimit" in text


def test_editor_lab_indicator_in_header() -> None:
    """The header shows 'mission · <lab>' so the user always knows
    which lab's mission they're editing."""
    text = EDITOR.read_text()
    assert "mission · ${labLabel}" in text or "mission ·" in text
    assert "labLabel" in text


def test_editor_aria_region_label() -> None:
    text = EDITOR.read_text()
    assert 'role="region"' in text
    assert 'aria-label="mission editor"' in text


# ─── FirstLight integration ───────────────────────────────────────


def test_first_light_imports_mission_editor() -> None:
    text = FIRST_LIGHT.read_text()
    assert "import { MissionEditor }" in text


def test_first_light_renders_mission_editor() -> None:
    text = FIRST_LIGHT.read_text()
    assert "<MissionEditor />" in text


def test_mission_editor_above_pulse_strip() -> None:
    """Positional check: editor renders between Letter and PulseStrip."""
    text = FIRST_LIGHT.read_text()
    editor_idx = text.find("<MissionEditor />")
    pulse_idx = text.find("<PulseStrip")
    assert editor_idx >= 0
    assert pulse_idx >= 0
    assert editor_idx < pulse_idx


def main() -> int:
    tests = [
        test_editor_file_exists,
        test_editor_has_edit_and_preview_tabs,
        test_editor_renders_markdown_preview,
        test_editor_uses_seed_brief_endpoint,
        test_editor_routes_via_lab_query,
        test_editor_handles_409_conflict,
        test_editor_optimistic_concurrency_via_expected_mtime,
        test_editor_default_expanded_when_brief_empty,
        test_editor_unsaved_changes_beforeunload_guard,
        test_editor_touch_targets_44px,
        test_editor_shows_dirty_indicator,
        test_editor_soft_warns_on_long_missions,
        test_editor_lab_indicator_in_header,
        test_editor_aria_region_label,
        test_first_light_imports_mission_editor,
        test_first_light_renders_mission_editor,
        test_mission_editor_above_pulse_strip,
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
