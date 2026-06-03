"""Smoke test for HH-C — NeedsYou inline expansion on Home.

Replaces the bare "N decisions waiting" link on Home with full inline
matter cards (decision blessing + approval choices). Equivalent to the
soon-retired /meeting surface — keep affordances, lose the navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


NE = LAB_ROOT / "bert" / "v4" / "src" / "components" / "NeedsExpanded.tsx"
FIRST_LIGHT = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx"


# ─── NeedsExpanded.tsx ────────────────────────────────────────────


def test_needs_expanded_file_exists() -> None:
    assert NE.exists()


def test_needs_expanded_uses_pending_and_approvals_hooks() -> None:
    text = NE.read_text()
    assert "usePending" in text
    assert "usePendingApprovals" in text


def test_needs_expanded_invokes_bless_and_veto_mutations() -> None:
    """Bless / veto are the two PI gestures on a decision matter —
    HH-C must surface them inline, not link out."""
    text = NE.read_text()
    assert "bless" in text
    assert "veto" in text
    assert 'queryKey: ["pending-blessings"]' in text


def test_needs_expanded_invokes_approve_mutation() -> None:
    text = NE.read_text()
    assert "approve(approval.id" in text
    assert 'queryKey: ["approvals"]' in text


def test_needs_expanded_renders_voice_strip() -> None:
    """Voices summary (concur/object/stand_aside) appears inline so
    the PI sees council weight without a separate page."""
    text = NE.read_text()
    assert "VoiceStrip" in text
    assert "stanceColor" in text
    assert "concur" in text
    assert "object" in text
    assert "stand_aside" in text


def test_needs_expanded_renders_resolved_banner() -> None:
    """After bless/veto/approve, the card flips to a receipt so the
    PI knows the action landed."""
    text = NE.read_text()
    assert "ResolvedBanner" in text
    assert "blessed" in text
    assert "vetoed" in text


def test_needs_expanded_caps_visible_then_show_all() -> None:
    """First 3 matters render; rest hide behind 'show all N matters'.
    Prevents Home from becoming a wall of cards when bert raises a
    burst of matters."""
    text = NE.read_text()
    assert "MAX_VISIBLE" in text
    assert "showAll" in text
    assert "show all" in text


def test_needs_expanded_veto_reason_field() -> None:
    """Set-aside collects an optional reason so bert knows what to
    season further. Hidden by default behind 'set aside ▾'."""
    text = NE.read_text()
    assert "vetoOpen" in text
    assert "reason" in text
    assert "textarea" in text


def test_needs_expanded_approval_options_rendered_as_buttons() -> None:
    text = NE.read_text()
    assert "approvalOptions" in text
    assert "options.map" in text


def test_needs_expanded_touch_targets_44px() -> None:
    text = NE.read_text()
    # Primary / secondary btn helpers + showAll btn
    assert text.count("minHeight: 44") >= 3


def test_needs_expanded_aria_labels() -> None:
    """Accessibility — region label + per-card labels + alert role
    on error banner."""
    text = NE.read_text()
    assert 'role="region"' in text
    assert 'aria-label="needs you"' in text
    assert 'role="alert"' in text
    assert 'role="status"' in text  # ResolvedBanner


def test_needs_expanded_hides_when_no_matters() -> None:
    """The card disappears entirely when nothing pends. Quiet days
    stay quiet (Tufte)."""
    text = NE.read_text()
    assert "if (total === 0) return null" in text


def test_needs_expanded_handles_mutation_errors() -> None:
    text = NE.read_text()
    assert "humanize" in text
    assert "ErrorLine" in text
    assert "is unreachable" in text


def test_needs_expanded_truncates_long_decision_body() -> None:
    """Decision body excerpt caps at ~280 chars so each card stays
    scannable on Home; full body is in the matter cycle output."""
    text = NE.read_text()
    assert "truncate" in text
    assert "280" in text


# ─── FirstLight integration ───────────────────────────────────────


def test_first_light_imports_needs_expanded() -> None:
    text = FIRST_LIGHT.read_text()
    assert "import { NeedsExpanded }" in text


def test_first_light_renders_needs_expanded() -> None:
    text = FIRST_LIGHT.read_text()
    assert "<NeedsExpanded />" in text


def test_first_light_drops_legacy_needsblock() -> None:
    """The old NeedsBlock + inlineLinkStyle + supervisor counters
    are gone — NeedsExpanded owns the rendering now."""
    text = FIRST_LIGHT.read_text()
    assert "function NeedsBlock" not in text
    assert "needsCount" not in text
    assert "inlineLinkStyle" not in text


def test_first_light_no_legacy_link_to_meeting() -> None:
    """The previous NeedsBlock linked to /meeting; HH-C makes that
    inline, and HH-E will retire the route entirely."""
    text = FIRST_LIGHT.read_text()
    assert '<Link to="/meeting"' not in text


def main() -> int:
    tests = [
        test_needs_expanded_file_exists,
        test_needs_expanded_uses_pending_and_approvals_hooks,
        test_needs_expanded_invokes_bless_and_veto_mutations,
        test_needs_expanded_invokes_approve_mutation,
        test_needs_expanded_renders_voice_strip,
        test_needs_expanded_renders_resolved_banner,
        test_needs_expanded_caps_visible_then_show_all,
        test_needs_expanded_veto_reason_field,
        test_needs_expanded_approval_options_rendered_as_buttons,
        test_needs_expanded_touch_targets_44px,
        test_needs_expanded_aria_labels,
        test_needs_expanded_hides_when_no_matters,
        test_needs_expanded_handles_mutation_errors,
        test_needs_expanded_truncates_long_decision_body,
        test_first_light_imports_needs_expanded,
        test_first_light_renders_needs_expanded,
        test_first_light_drops_legacy_needsblock,
        test_first_light_no_legacy_link_to_meeting,
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
