"""Smoke test for HH-F — SurfaceShelf removal + FirstLight→Home
rename + ErrorBoundary + lazy loading.

Rounds out the rehaul:
- Home.tsx (renamed from FirstLight.tsx) without the bottom
  SurfaceShelf — the masthead now owns navigation.
- ErrorBoundary scopes render errors to a single surface; the
  masthead / drawer / banner chrome stays mounted.
- React.lazy + Suspense splits each surface into its own JS chunk
  so first paint only pays for Home.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


HOME       = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx"
OLD_FL     = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "FirstLight.tsx"
APP        = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"
BOUNDARY   = LAB_ROOT / "bert" / "v4" / "src" / "components" / "ErrorBoundary.tsx"
LOADING    = LAB_ROOT / "bert" / "v4" / "src" / "components" / "RouteLoading.tsx"


# ─── Home rename ──────────────────────────────────────────────────


def test_home_file_exists() -> None:
    assert HOME.exists()


def test_first_light_file_removed() -> None:
    """FirstLight.tsx renamed via git mv to Home.tsx; old path
    must not exist."""
    assert not OLD_FL.exists()


def test_home_exports_home_function() -> None:
    text = HOME.read_text()
    assert "export function Home(" in text


def test_home_drops_first_light_export() -> None:
    text = HOME.read_text()
    assert "export function FirstLight(" not in text


def test_home_kicker_updated() -> None:
    text = HOME.read_text()
    assert "bert · home" in text
    assert "bert · first light" not in text


# ─── SurfaceShelf removal ─────────────────────────────────────────


def test_home_no_surface_shelf() -> None:
    """The bottom shelf of surface links is gone — the masthead
    is the canonical nav surface now. Comments may still mention
    SurfaceShelf when explaining the removal; what's banned is
    the component definition and the rendered element."""
    text = HOME.read_text()
    assert "<SurfaceShelf />" not in text
    assert "function SurfaceShelf(" not in text
    assert "function SurfaceLink(" not in text


def test_home_no_links_to_retired_surfaces() -> None:
    text = HOME.read_text()
    # The bottom shelf used to render these; with shelf gone, no
    # links to retired routes should remain inside Home.
    assert '"/mission"' not in text
    assert '"/meeting"' not in text
    assert '"/tide"' not in text
    assert '"/loom"' not in text


# ─── ErrorBoundary ────────────────────────────────────────────────


def test_error_boundary_file_exists() -> None:
    assert BOUNDARY.exists()


def test_error_boundary_is_class_component() -> None:
    """class component, the only reliable way to catch render
    errors via componentDidCatch."""
    text = BOUNDARY.read_text()
    assert "extends Component" in text


def test_error_boundary_uses_get_derived_state_from_error() -> None:
    text = BOUNDARY.read_text()
    assert "getDerivedStateFromError" in text


def test_error_boundary_uses_component_did_catch() -> None:
    text = BOUNDARY.read_text()
    assert "componentDidCatch" in text


def test_error_boundary_logs_to_console() -> None:
    """Per quality answer — log the stack so the PI can grab it
    from devtools to file an issue."""
    text = BOUNDARY.read_text()
    assert "console.error" in text


def test_error_boundary_has_try_again_button() -> None:
    """Reset path — clear the error state and re-render the
    children. If the failure was transient, retry recovers."""
    text = BOUNDARY.read_text()
    assert "try again" in text
    assert "this.reset" in text or "reset = " in text


def test_error_boundary_has_return_home_link() -> None:
    """Bail path — give the PI a way out of a stuck surface."""
    text = BOUNDARY.read_text()
    assert "return home" in text
    assert 'to="/"' in text


def test_error_boundary_aria_live_alert() -> None:
    text = BOUNDARY.read_text()
    assert 'role="alert"' in text
    assert 'aria-live="assertive"' in text


def test_error_boundary_surfaces_error_message() -> None:
    """The message is shown verbatim so the PI knows what broke
    — not a generic 'something went wrong'."""
    text = BOUNDARY.read_text()
    assert "error.message" in text


def test_error_boundary_44px_touch_targets() -> None:
    text = BOUNDARY.read_text()
    assert "minHeight: 44" in text


# ─── RouteLoading (Suspense fallback) ─────────────────────────────


def test_route_loading_file_exists() -> None:
    assert LOADING.exists()


def test_route_loading_has_aria_busy() -> None:
    """Screen-reader hint that a chunk is loading."""
    text = LOADING.read_text()
    assert 'aria-busy="true"' in text
    assert 'role="status"' in text


# ─── App.tsx lazy loading + boundaries ────────────────────────────


def test_app_imports_home_not_first_light() -> None:
    text = APP.read_text()
    assert "./surfaces/Home" in text
    assert "./surfaces/FirstLight" not in text


def test_app_imports_error_boundary() -> None:
    text = APP.read_text()
    assert "ErrorBoundary" in text


def test_app_imports_route_loading() -> None:
    text = APP.read_text()
    assert "RouteLoading" in text


def test_app_uses_lazy_for_home() -> None:
    """Per HH-F — surfaces are lazy-loaded so first paint only
    pays for the Home chunk."""
    text = APP.read_text()
    assert "lazy(() => import" in text
    assert "Home: lazy" not in text   # not object-bag style
    # Use lazy specifically for Home
    assert 'lazy(() => import("./surfaces/Home")' in text


def test_app_lazy_loads_all_major_surfaces() -> None:
    text = APP.read_text()
    for surf in ("Home", "ManuscriptTabbed", "LabDashboard", "Outputs",
                 "Choreography", "Diagnostics", "Atlas", "Onboarding"):
        # Each major surface is loaded via lazy()
        assert f'import("./surfaces/{surf}")' in text, f"missing lazy import for {surf}"


def test_app_uses_suspense() -> None:
    text = APP.read_text()
    assert "Suspense" in text
    assert "<RouteLoading />" in text


def test_app_bounded_helper_wraps_routes() -> None:
    """bounded(label, node) wraps each route in ErrorBoundary +
    Suspense; render errors stay scoped to the surface."""
    text = APP.read_text()
    assert "function bounded" in text
    assert 'bounded("home"' in text
    assert 'bounded("manuscript"' in text
    assert 'bounded("diagnostics"' in text


def main() -> int:
    tests = [
        test_home_file_exists,
        test_first_light_file_removed,
        test_home_exports_home_function,
        test_home_drops_first_light_export,
        test_home_kicker_updated,
        test_home_no_surface_shelf,
        test_home_no_links_to_retired_surfaces,
        test_error_boundary_file_exists,
        test_error_boundary_is_class_component,
        test_error_boundary_uses_get_derived_state_from_error,
        test_error_boundary_uses_component_did_catch,
        test_error_boundary_logs_to_console,
        test_error_boundary_has_try_again_button,
        test_error_boundary_has_return_home_link,
        test_error_boundary_aria_live_alert,
        test_error_boundary_surfaces_error_message,
        test_error_boundary_44px_touch_targets,
        test_route_loading_file_exists,
        test_route_loading_has_aria_busy,
        test_app_imports_home_not_first_light,
        test_app_imports_error_boundary,
        test_app_imports_route_loading,
        test_app_uses_lazy_for_home,
        test_app_lazy_loads_all_major_surfaces,
        test_app_uses_suspense,
        test_app_bounded_helper_wraps_routes,
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
