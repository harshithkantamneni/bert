"""Smoke test for HH-A — Masthead + LabPicker + TalkFold.

The structural rehaul foundation. Replaces FirstLight's bottom
SurfaceShelf with a persistent sticky masthead containing:
  - brand "BERT" + today's date
  - lab picker dropdown (viewing: <lab> ▾)
  - surface nav (home / fleet / proofs / atlas / manuscript / diagnostics)
  - talk affordance (folded in from the bottom-right chip)

The TalkDrawer state is now a singleton via React Context so the
masthead's talk link and the drawer itself share state.

Covers:
  Singleton context (B10 fix):
    - api/talkDrawer.tsx exports TalkDrawerProvider + useTalkDrawer
    - useTalkDrawer returns { open, setOpen, toggle, pendingCount }
    - Defensive no-op shape when consumer outside provider
    - Background poll updates pendingCount every 5s
    - Escape closes drawer globally

  Masthead component:
    - File exists; sticky position; backdrop-filter blur
    - Hidden on /onboard + /dev/* (returns null)
    - Surface nav includes all 6 user surfaces
    - Active-link logic uses pathname.startsWith for prefix match
    - All clickable elements ≥44px touch target
    - Mobile breakpoint at ≤640px (two-line layout)
    - Brand link navigates to /
    - Talk link consumes useTalkDrawer().toggle
    - Talk link shows "talk · N" badge when pendingCount > 0

  LabPicker:
    - Renders trigger "viewing: <lab> ▾"
    - Dropdown lists labs from useLabs()
    - Click option calls setActiveLab + invalidateQueries + navigate('/')
    - Outside click closes
    - Escape closes
    - Arrow nav supported
    - Two variants: desktop (anchored) + mobile (bottom sheet)
    - Active-lab marked with candle dot

  MobileMastheadSheet:
    - Slides from top
    - Lists all surface links
    - Tap navigates + closes
    - Backdrop dismisses

  TalkToLab refactor:
    - Chip removed; only drawer remains
    - Drawer subscribes to useTalkDrawer
    - Close button inside drawer
    - Preserved existing send/draft functionality
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


TALK_CTX = LAB_ROOT / "bert" / "v4" / "src" / "api" / "talkDrawer.tsx"
MASTHEAD = LAB_ROOT / "bert" / "v4" / "src" / "components" / "Masthead.tsx"
LAB_PICKER = LAB_ROOT / "bert" / "v4" / "src" / "components" / "LabPicker.tsx"
MOBILE_SHEET = LAB_ROOT / "bert" / "v4" / "src" / "components" / "MobileMastheadSheet.tsx"
TALK = LAB_ROOT / "bert" / "v4" / "src" / "components" / "TalkToLab.tsx"
APP_TSX = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"


# ─── Singleton context ────────────────────────────────────────────


def test_talk_drawer_context_module_exists() -> None:
    assert TALK_CTX.exists()


def test_talk_drawer_exports_provider_and_hook() -> None:
    text = TALK_CTX.read_text()
    assert "export function TalkDrawerProvider" in text
    assert "export function useTalkDrawer" in text


def test_talk_drawer_returns_expected_shape() -> None:
    """B10 fix: singleton via Context. Hook returns { open, setOpen,
    toggle, pendingCount }."""
    text = TALK_CTX.read_text()
    for field in ("open", "setOpen", "toggle", "pendingCount"):
        assert field in text, f"context shape missing {field!r}"


def test_talk_drawer_has_defensive_noop_when_outside_provider() -> None:
    """If a consumer renders outside the provider, return a no-op
    instead of crashing the app."""
    text = TALK_CTX.read_text()
    assert "if (!ctx)" in text
    assert "() => undefined" in text or "noop" in text.lower()


def test_talk_drawer_polls_pending_count() -> None:
    text = TALK_CTX.read_text()
    assert "setInterval" in text
    assert "pi_message" in text
    assert "consumed_at_cycle" in text


def test_talk_drawer_global_escape_closes() -> None:
    text = TALK_CTX.read_text()
    assert "Escape" in text
    assert "keydown" in text


# ─── Masthead ─────────────────────────────────────────────────────


def test_masthead_component_exists() -> None:
    assert MASTHEAD.exists()


def test_masthead_uses_sticky_position_with_blur() -> None:
    text = MASTHEAD.read_text()
    assert "position: \"sticky\"" in text
    assert "backdropFilter" in text


def test_masthead_hidden_on_onboard_and_dev_routes() -> None:
    text = MASTHEAD.read_text()
    assert 'path.startsWith("/onboard")' in text
    assert 'path.startsWith("/dev/")' in text
    # Returns null when on those paths
    assert "return null" in text


def test_masthead_lists_all_user_surfaces() -> None:
    text = MASTHEAD.read_text()
    for surface in ("home", "fleet", "proofs", "atlas",
                     "manuscript", "diagnostics"):
        assert f'"{surface}"' in text, (
            f"masthead missing surface link: {surface}"
        )


def test_masthead_active_link_uses_starts_with() -> None:
    """Per Q8 + B23: active match uses pathname.startsWith so
    /manuscript?tab=stream still highlights the manuscript link."""
    text = MASTHEAD.read_text()
    assert "startsWith(s.matchPrefix)" in text


def test_masthead_touch_targets_44px() -> None:
    text = MASTHEAD.read_text()
    # Multiple buttons/links in the masthead; all need minHeight: 44
    assert text.count("minHeight: 44") >= 3


def test_masthead_filters_demo_mode_surfaces() -> None:
    """Demo mode hides admin surfaces (matches existing pattern)."""
    text = MASTHEAD.read_text()
    assert "isDemoMode" in text
    assert "filter" in text or "ALL_SURFACES" in text


def test_masthead_responsive_breakpoint() -> None:
    """Mobile breakpoint at 640px per Q11."""
    text = MASTHEAD.read_text()
    assert "640" in text


def test_masthead_brand_links_to_home() -> None:
    text = MASTHEAD.read_text()
    assert 'to="/"' in text  # brand link
    assert 'BERT' in text


# ─── LabPicker ────────────────────────────────────────────────────


def test_lab_picker_component_exists() -> None:
    assert LAB_PICKER.exists()


def test_lab_picker_consumes_use_labs_hook() -> None:
    text = LAB_PICKER.read_text()
    assert "useLabs" in text


def test_lab_picker_invalidates_and_navigates_on_select() -> None:
    """Q10: lab switch lands on Home so user has stable starting point."""
    text = LAB_PICKER.read_text()
    assert "invalidateQueries" in text
    assert 'navigate("/")' in text
    assert "setActiveLab" in text


def test_lab_picker_closes_on_outside_click() -> None:
    text = LAB_PICKER.read_text()
    assert "mousedown" in text


def test_lab_picker_closes_on_escape() -> None:
    text = LAB_PICKER.read_text()
    assert '"Escape"' in text


def test_lab_picker_arrow_nav_supported() -> None:
    text = LAB_PICKER.read_text()
    assert "ArrowDown" in text
    assert "ArrowUp" in text


def test_lab_picker_has_two_variants() -> None:
    text = LAB_PICKER.read_text()
    assert '"desktop"' in text
    assert '"mobile"' in text


def test_lab_picker_uses_aria_listbox() -> None:
    text = LAB_PICKER.read_text()
    assert 'role="listbox"' in text or 'aria-haspopup="listbox"' in text
    assert 'role="option"' in text


# ─── MobileMastheadSheet ──────────────────────────────────────────


def test_mobile_sheet_component_exists() -> None:
    assert MOBILE_SHEET.exists()


def test_mobile_sheet_renders_dialog() -> None:
    text = MOBILE_SHEET.read_text()
    assert 'role="dialog"' in text
    assert 'aria-modal="true"' in text


def test_mobile_sheet_lists_all_surfaces_passed_in() -> None:
    text = MOBILE_SHEET.read_text()
    assert "surfaces.map" in text


def test_mobile_sheet_closes_on_link_click() -> None:
    """Tapping a link closes the sheet + navigates."""
    text = MOBILE_SHEET.read_text()
    assert "onClick={onClose}" in text


def test_mobile_sheet_has_backdrop() -> None:
    text = MOBILE_SHEET.read_text()
    assert "rgba(0,0,0," in text  # backdrop opacity


# ─── TalkToLab refactor ───────────────────────────────────────────


def test_talk_to_lab_consumes_singleton_drawer() -> None:
    text = TALK.read_text()
    assert "useTalkDrawer" in text


def test_talk_to_lab_no_persistent_chip() -> None:
    """The chip moved into the masthead. TalkToLab now renders only
    the drawer."""
    text = TALK.read_text()
    # No fixed bottom-right button block
    assert "Persistent chip in bottom-right" not in text
    # The chip's distinguishing comment block is gone
    decommented = re.sub(r"//[^\n]*", "", text)
    decommented = re.sub(r"/\*.*?\*/", "", decommented, flags=re.DOTALL)
    # No `position: "fixed"` with `bottom: 20, right: 20` (the old chip)
    assert not re.search(r"position:\s*['\"]fixed['\"][^}]*bottom:\s*20[^}]*right:\s*20",
                          decommented)


def test_talk_to_lab_drawer_has_close_button() -> None:
    text = TALK.read_text()
    assert 'aria-label="close talk drawer"' in text


def test_talk_to_lab_still_renders_drawer_on_open() -> None:
    text = TALK.read_text()
    # The slide-in drawer animation primitives are preserved
    assert "x: 480" in text  # slide-from-right
    assert "motion.aside" in text


# ─── App.tsx integration ──────────────────────────────────────────


def test_app_wraps_in_talk_drawer_provider() -> None:
    text = APP_TSX.read_text()
    assert "TalkDrawerProvider" in text
    assert "<TalkDrawerProvider>" in text
    assert "</TalkDrawerProvider>" in text


def test_app_mounts_masthead() -> None:
    text = APP_TSX.read_text()
    assert "import { Masthead }" in text
    assert "<Masthead />" in text


def main() -> int:
    tests = [
        test_talk_drawer_context_module_exists,
        test_talk_drawer_exports_provider_and_hook,
        test_talk_drawer_returns_expected_shape,
        test_talk_drawer_has_defensive_noop_when_outside_provider,
        test_talk_drawer_polls_pending_count,
        test_talk_drawer_global_escape_closes,
        test_masthead_component_exists,
        test_masthead_uses_sticky_position_with_blur,
        test_masthead_hidden_on_onboard_and_dev_routes,
        test_masthead_lists_all_user_surfaces,
        test_masthead_active_link_uses_starts_with,
        test_masthead_touch_targets_44px,
        test_masthead_filters_demo_mode_surfaces,
        test_masthead_responsive_breakpoint,
        test_masthead_brand_links_to_home,
        test_lab_picker_component_exists,
        test_lab_picker_consumes_use_labs_hook,
        test_lab_picker_invalidates_and_navigates_on_select,
        test_lab_picker_closes_on_outside_click,
        test_lab_picker_closes_on_escape,
        test_lab_picker_arrow_nav_supported,
        test_lab_picker_has_two_variants,
        test_lab_picker_uses_aria_listbox,
        test_mobile_sheet_component_exists,
        test_mobile_sheet_renders_dialog,
        test_mobile_sheet_lists_all_surfaces_passed_in,
        test_mobile_sheet_closes_on_link_click,
        test_mobile_sheet_has_backdrop,
        test_talk_to_lab_consumes_singleton_drawer,
        test_talk_to_lab_no_persistent_chip,
        test_talk_to_lab_drawer_has_close_button,
        test_talk_to_lab_still_renders_drawer_on_open,
        test_app_wraps_in_talk_drawer_provider,
        test_app_mounts_masthead,
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
