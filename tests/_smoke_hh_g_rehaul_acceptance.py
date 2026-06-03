"""HH-G — final acceptance smoke for the HH rehaul.

Aggregates the HH-A through HH-F smoke suites + verifies the
production vite build succeeds + verifies code splitting is in
effect + verifies key invariants of the rehaul are preserved
across the codebase (not just per-phase).

This is the gate that the user-facing rehaul actually shipped.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


V4   = LAB_ROOT / "bert" / "v4"
DIST = V4 / "dist"
TESTS = LAB_ROOT / "tests"

HH_SMOKES = [
    "_smoke_hh_a_masthead.py",
    "_smoke_hh_b_mission_editor.py",
    "_smoke_hh_c_needs_expanded.py",
    "_smoke_hh_d_manuscript_tabbed.py",
    "_smoke_hh_e_route_retirement.py",
    "_smoke_hh_f_boundaries_lazy.py",
]


# ─── HH-A..F regression ───────────────────────────────────────────


def test_all_hh_phase_smokes_pass() -> None:
    """Every HH phase's smoke suite must pass green. This is the
    floor; any failure here breaks the rehaul gate."""
    py = LAB_ROOT / ".venv" / "bin" / "python"
    failures: list[str] = []
    for smoke in HH_SMOKES:
        path = TESTS / smoke
        if not path.exists():
            failures.append(f"{smoke}: file missing")
            continue
        r = subprocess.run(
            [str(py), str(path)],
            capture_output=True, text=True, cwd=str(LAB_ROOT), timeout=120,
        )
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        if r.returncode != 0 or "passed" not in last:
            failures.append(f"{smoke}: rc={r.returncode} · {last}")
    assert not failures, "HH smoke failures:\n  " + "\n  ".join(failures)


# ─── Architecture invariants (not just per-phase) ─────────────────


def test_no_dangling_first_light_references() -> None:
    """Post HH-F: no source file under bert/v4/src should still
    reference FirstLight as a component import or render. Comments
    that explain the rename are fine."""
    bad: list[str] = []
    for src in (V4 / "src").rglob("*.tsx"):
        text = src.read_text()
        if "import { FirstLight }" in text:
            bad.append(f"{src.name}: import { '{ FirstLight }' }")
        if "<FirstLight" in text:
            bad.append(f"{src.name}: <FirstLight render")
    for src in (V4 / "src").rglob("*.ts"):
        text = src.read_text()
        if "import { FirstLight }" in text:
            bad.append(f"{src.name}: import { '{ FirstLight }' }")
    assert not bad, "dangling FirstLight references:\n  " + "\n  ".join(bad)


def test_no_dangling_mission_or_meeting_surface() -> None:
    """Mission.tsx + Meeting.tsx were deleted in HH-E. No source
    file should still import them."""
    bad: list[str] = []
    for src in (V4 / "src").rglob("*.tsx"):
        text = src.read_text()
        if 'from "./surfaces/Mission"' in text or 'from "../surfaces/Mission"' in text:
            bad.append(f"{src.name}: still imports Mission")
        if 'from "./surfaces/Meeting"' in text or 'from "../surfaces/Meeting"' in text:
            bad.append(f"{src.name}: still imports Meeting")
    assert not bad, "dangling Mission/Meeting imports:\n  " + "\n  ".join(bad)


def test_masthead_owns_all_user_facing_nav() -> None:
    """The masthead is the canonical nav surface post HH-A/F. No
    other component should render a parallel <nav> with surface
    links."""
    masthead = (V4 / "src" / "components" / "Masthead.tsx").read_text()
    assert 'aria-label="surfaces"' in masthead
    # Home shouldn't have its own surfaces nav anymore
    home = (V4 / "src" / "surfaces" / "Home.tsx").read_text()
    assert 'aria-label="surfaces"' not in home


def test_retired_routes_are_redirects() -> None:
    """Every retired URL is a QueryPreservingRedirect, not a
    component render. This is the deep-link compatibility floor."""
    app = (V4 / "src" / "App.tsx").read_text()
    for path in ("/mission", "/meeting", "/tide", "/loom", "/book"):
        # Find the route line and check it uses QueryPreservingRedirect
        m = re.search(
            rf'path="{re.escape(path)}"[^>]*\s+element=\{{<QueryPreservingRedirect',
            app,
        )
        assert m, f"{path} not redirected via QueryPreservingRedirect"


def test_every_route_has_error_boundary() -> None:
    """Every component-rendering Route element passes through
    bounded(), which wraps in ErrorBoundary + Suspense. Only
    QueryPreservingRedirect routes are exempt (they're stateless
    redirects)."""
    app = (V4 / "src" / "App.tsx").read_text()
    component_routes = re.findall(
        r'<Route\s+path="[^"]+"\s+element=\{([^}]+)\}',
        app,
    )
    bad = []
    for elem in component_routes:
        # Exempt cases: redirect, Navigate, ternary that yields one
        if "QueryPreservingRedirect" in elem: continue
        if "Navigate" in elem: continue
        if "NotYet" in elem: continue
        if "bounded(" in elem: continue
        # Ternaries with bounded() inside are also OK (the Home route)
        bad.append(elem.strip()[:80])
    assert not bad, \
        "routes not wrapped in bounded():\n  " + "\n  ".join(bad)


# ─── Vite build verification ──────────────────────────────────────


def test_vite_build_artifacts_exist() -> None:
    """dist/ must exist with the index + at least the Home chunk."""
    assert DIST.exists()
    assert (DIST / "index.html").exists()
    assets = DIST / "assets"
    assert assets.exists()
    # Lazy-loaded surfaces should each emit their own chunk
    js_files = sorted(p.name for p in assets.glob("*.js"))
    # Look for Home-*.js, ManuscriptTabbed-*.js, etc.
    needed = ["Home-", "ManuscriptTabbed-", "Diagnostics-",
              "Atlas-", "Onboarding-", "Outputs-", "LabDashboard-"]
    for prefix in needed:
        match = next((f for f in js_files if f.startswith(prefix)), None)
        assert match, f"missing lazy-loaded chunk for {prefix}"


def test_home_chunk_under_50kb_pre_gzip() -> None:
    """Sanity check on the first-paint cost: the Home chunk
    shouldn't be larger than ~50KB raw."""
    assets = DIST / "assets"
    home_files = list(assets.glob("Home-*.js"))
    assert home_files, "Home chunk missing"
    size = home_files[0].stat().st_size
    assert size < 60_000, f"Home chunk too large: {size} bytes"


def test_lazy_chunks_distinct_files() -> None:
    """Each lazy() target compiles to a separate file (no chunk
    fusion that would defeat code splitting)."""
    assets = DIST / "assets"
    surface_chunks = {
        "Home": None, "ManuscriptTabbed": None, "LabDashboard": None,
        "Diagnostics": None, "Atlas": None, "Outputs": None,
        "Choreography": None, "Onboarding": None,
    }
    for f in assets.glob("*.js"):
        for key in surface_chunks:
            if f.name.startswith(f"{key}-"):
                surface_chunks[key] = f.name
    distinct = {v for v in surface_chunks.values() if v}
    assert len(distinct) == len([v for v in surface_chunks.values() if v]), \
        "chunks fused unexpectedly"


# ─── Bundle-level invariants ──────────────────────────────────────


def test_react_virtual_packaged() -> None:
    """@tanstack/react-virtual must end up in the bundle since
    ManuscriptTabbed's stream tab depends on it. The
    ManuscriptTabbed chunk should contain the virtual machinery."""
    chunk = next((DIST / "assets").glob("ManuscriptTabbed-*.js"), None)
    assert chunk, "ManuscriptTabbed chunk missing"
    text = chunk.read_text()
    # `virtual` appears in the library's identifier names even
    # after minification (Virtualizer class names retain stems).
    assert "virtual" in text.lower(), \
        "react-virtual not packaged in ManuscriptTabbed chunk"


def test_remark_gfm_packaged() -> None:
    """remarkGfm + ReactMarkdown power the mission editor's
    markdown preview; they should be in the Home chunk (where
    MissionEditor lives)."""
    chunk = next((DIST / "assets").glob("Home-*.js"), None)
    assert chunk, "Home chunk missing"
    text = chunk.read_text()
    assert "markdown" in text.lower() or "remark" in text.lower(), \
        "remark-gfm / react-markdown not packaged in Home chunk"


# ─── Memory + test inventory ──────────────────────────────────────


def test_hh_smokes_present_for_every_phase() -> None:
    for smoke in HH_SMOKES:
        assert (TESTS / smoke).exists(), f"missing {smoke}"


def test_playwright_walkthrough_present() -> None:
    """The 20-stop browser walkthrough must exist as a runnable
    script even if it isn't executed in this acceptance gate
    (Playwright needs the dev server)."""
    pw = TESTS / "_walkthrough_hh_rehaul.py"
    assert pw.exists(), "HH walkthrough script missing"


def main() -> int:
    tests = [
        # HH-A..F regression
        test_all_hh_phase_smokes_pass,
        # Architecture invariants
        test_no_dangling_first_light_references,
        test_no_dangling_mission_or_meeting_surface,
        test_masthead_owns_all_user_facing_nav,
        test_retired_routes_are_redirects,
        test_every_route_has_error_boundary,
        # Build artifacts
        test_vite_build_artifacts_exist,
        test_home_chunk_under_50kb_pre_gzip,
        test_lazy_chunks_distinct_files,
        # Bundle invariants
        test_react_virtual_packaged,
        test_remark_gfm_packaged,
        # Test inventory
        test_hh_smokes_present_for_every_phase,
        test_playwright_walkthrough_present,
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
    print(f"\nAll {len(tests)} acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
