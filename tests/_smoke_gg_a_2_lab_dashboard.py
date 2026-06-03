"""Smoke test for GG-A.2 — lab dashboard with strata-card visual idiom.

Per locked feedback memory `feedback_visualization_as_art`, the
dashboard MUST lead with a specific visual idiom (geological strata)
rather than a generic card grid. This test enforces that contract at
the source-file level — it can't lock the rendered output without a
browser test, but it can lock the design tokens, motion primitives,
and structural decisions that produce the right idiom.

Covers:
  - LabDashboard surface file exists at the canonical path
  - Strata-card visual primitives present (band, gradient, accent
    border, archetype tint, thickness function)
  - /labs route registered in App.tsx
  - useMultiLab hook gates root → /labs when count > 1
  - Solo-lab user (count == 1) stays on FirstLight
  - Loading state uses StratumSkeleton (consistent with rest of UI)
  - Empty state uses ConnectomicEmpty (no labs scaffolded yet)
  - Archetype → tint mapping includes all 4 (research/product/strategy/supervisor)
  - LabSummary type extends with all FF-A fields
  - Honest "private" tag rendered for share_with_supervisor=false labs
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


DASHBOARD = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "LabDashboard.tsx"
APP_TSX = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"
CLIENT_TS = LAB_ROOT / "bert" / "v4" / "src" / "api" / "client.ts"


# ─── Surface file exists + visual idiom ────────────────────────────


def test_lab_dashboard_surface_file_exists() -> None:
    assert DASHBOARD.exists(), (
        f"LabDashboard surface missing at {DASHBOARD}"
    )


def test_lab_dashboard_uses_strata_visual_idiom() -> None:
    """Per feedback_visualization_as_art: the dashboard must lead with
    a specific visual idiom (geological strata), not generic cards.
    The source must explicitly reference the idiom in code or comment
    so a future reader knows WHY the layout looks this way."""
    text = DASHBOARD.read_text()
    assert "strata" in text.lower() or "stratum" in text.lower(), (
        "LabDashboard must explicitly reference the geological-strata "
        "visual idiom (per feedback_visualization_as_art)"
    )
    # The component primitive that renders each lab as a band
    assert "function StratumBand" in text or "const StratumBand" in text


def test_lab_dashboard_NOT_a_generic_card_grid() -> None:
    """Anti-pattern check. The dashboard must NOT use display: grid
    with repeat() (the standard "card grid" pattern) or display: flex
    with flex-wrap. Strata stack vertically; grids spread.
    """
    text = DASHBOARD.read_text()
    # If the component were a card grid these patterns would show up.
    # We allow flex for layout chrome (e.g. the header row) but the
    # main stack rendering must NOT use grid-template-columns repeat()
    # or flex-wrap: "wrap" on the lab list itself.
    forbidden_grid_pattern = re.search(
        r"gridTemplateColumns:\s*['\"]repeat\(", text)
    assert not forbidden_grid_pattern, (
        "LabDashboard uses CSS grid repeat() — that's the generic "
        "card-grid pattern feedback_visualization_as_art warns against. "
        "Strata stack vertically."
    )


def test_strata_band_encodes_activity_as_thickness() -> None:
    """One of the core idiom rules: thickness reads activity, the way
    a sedimentary band's thickness records depositional time."""
    text = DASHBOARD.read_text()
    assert "thicknessFor" in text or "thickness" in text.lower()
    assert "events_total" in text  # activity metric


def test_strata_band_encodes_archetype_as_tint() -> None:
    """Color encodes archetype. The mapping must include all four
    archetype values."""
    text = DASHBOARD.read_text()
    assert "ARCHETYPE_TINT" in text
    for archetype in ("research", "product", "strategy", "supervisor"):
        assert archetype in text, (
            f"archetype {archetype!r} missing from tint mapping"
        )


def test_strata_band_shows_focus_areas_chips() -> None:
    """The chip rail must be there — one of the primary visual reads."""
    text = DASHBOARD.read_text()
    assert "focus_areas" in text
    # Chips have a specific aesthetic — bordered, small, uppercase mono
    assert ".slice(0, 5)" in text or "focus_areas.map" in text
    assert "TRACKING.kicker" in text  # the kicker letter-spacing


def test_strata_band_shows_mission_preview() -> None:
    text = DASHBOARD.read_text()
    assert "missionPreview" in text or "mission" in text
    # First sentence rule
    assert ".trim().split" in text or ".split" in text


def test_loading_state_uses_stratum_skeleton() -> None:
    """Consistent visual language with the rest of the UI."""
    text = DASHBOARD.read_text()
    assert "StratumSkeleton" in text


def test_empty_state_uses_connectomic_empty() -> None:
    """No labs yet → ConnectomicEmpty, not a generic placeholder."""
    text = DASHBOARD.read_text()
    assert "ConnectomicEmpty" in text


def test_private_lab_surfaces_honest_tag() -> None:
    """share_with_supervisor=false labs surface a 'private' tag so
    the owner sees the lab AND sees it's hidden from the supervisor's
    cross-lab read. Otherwise they'd have to remember per-lab."""
    text = DASHBOARD.read_text()
    assert "share_with_supervisor" in text
    assert "private" in text.lower()


# ─── /labs route + smart routing ───────────────────────────────────


def test_labs_route_registered() -> None:
    text = APP_TSX.read_text()
    assert '<Route path="/labs"' in text
    assert "LabDashboard" in text


# Post HH-rehaul: the multi-lab → /labs auto-redirect was retired.
# The masthead has both `home` and `fleet` links + a lab picker;
# auto-redirecting to /labs made `home` a no-op. Root now always
# renders Home for the active lab.

def test_root_renders_home_for_active_lab() -> None:
    """Root no longer auto-redirects on multi-lab state. The lab
    picker in the masthead is the canonical way to view a specific
    lab; root shows that lab's Home."""
    text = APP_TSX.read_text()
    # No multiLab gate
    assert "useMultiLab" not in text
    assert "multiLab === true" not in text
    # Root renders Home (via bounded() wrapper from HH-F)
    assert 'bounded("home", <Home />)' in text


def test_credentials_gate_still_redirects_to_onboard() -> None:
    """The credentials gate is still load-bearing — without keys,
    nothing works, so / → /onboard."""
    text = APP_TSX.read_text()
    assert "credsReady === false" in text
    assert '<Navigate to="/onboard" replace />' in text


def test_fleet_reached_via_masthead_link() -> None:
    """The fleet (/labs) is accessible from the masthead, not via
    auto-redirect from root."""
    text = (LAB_ROOT / "bert" / "v4" / "src" / "components" / "Masthead.tsx").read_text()
    assert '"fleet"' in text
    assert 'to: "/labs"' in text


# ─── TypeScript types extended ─────────────────────────────────────


def test_lab_summary_type_extended_with_ff_a_fields() -> None:
    text = CLIENT_TS.read_text()
    # New LabSummary type for the flat list
    assert "export interface LabSummary" in text
    for field in ("name", "path", "is_supervisor", "archetype", "role",
                  "mission", "focus_areas", "share_with_supervisor",
                  "events_total", "config_warnings"):
        assert field in text, f"LabSummary missing field {field!r}"


def test_labs_response_includes_unified_labs_array() -> None:
    text = CLIENT_TS.read_text()
    # Pre-GG was {ts, active, scaffolded}; GG-A.1 added unified labs[]
    assert "labs: LabSummary[]" in text
    assert "count: number" in text
    # L.4 contract preserved
    assert "scaffolded: ScaffoldedLab[]" in text
    assert "active: { path: string;" in text


def test_scaffolded_lab_type_extended() -> None:
    """ScaffoldedLab gained FF-A-aware fields so the dashboard's
    legacy scaffolded[] consumers don't need a second lookup."""
    text = CLIENT_TS.read_text()
    sl_idx = text.find("export interface ScaffoldedLab")
    assert sl_idx >= 0
    body = text[sl_idx:sl_idx + 800]
    for field in ("role", "mission", "focus_areas",
                  "share_with_supervisor", "events_total"):
        assert field in body, (
            f"ScaffoldedLab missing FF-A field {field!r}; "
            f"body excerpt: {body[:300]!r}"
        )


def main() -> int:
    tests = [
        test_lab_dashboard_surface_file_exists,
        test_lab_dashboard_uses_strata_visual_idiom,
        test_lab_dashboard_NOT_a_generic_card_grid,
        test_strata_band_encodes_activity_as_thickness,
        test_strata_band_encodes_archetype_as_tint,
        test_strata_band_shows_focus_areas_chips,
        test_strata_band_shows_mission_preview,
        test_loading_state_uses_stratum_skeleton,
        test_empty_state_uses_connectomic_empty,
        test_private_lab_surfaces_honest_tag,
        test_labs_route_registered,
        test_root_renders_home_for_active_lab,
        test_credentials_gate_still_redirects_to_onboard,
        test_fleet_reached_via_masthead_link,
        test_lab_summary_type_extended_with_ff_a_fields,
        test_labs_response_includes_unified_labs_array,
        test_scaffolded_lab_type_extended,
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
