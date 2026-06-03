"""Smoke test for I.8: 5 polish components + audit doc."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent


def test_five_components_exist() -> None:
    comp_dir = LAB_ROOT / "bert" / "v4" / "src" / "components"
    expected = [
        "StratumSkeleton.tsx",
        "LabIsThinking.tsx",
        "ConnectomicEmpty.tsx",
        "ProviderCooledBadge.tsx",
        "RelativeTime.tsx",
    ]
    for name in expected:
        p = comp_dir / name
        assert p.exists(), f"bert/v4/src/components/{name} missing"


def test_components_export_named_component() -> None:
    """Each component file must export a function with the same name."""
    comp_dir = LAB_ROOT / "bert" / "v4" / "src" / "components"
    for name in ["StratumSkeleton", "LabIsThinking", "ConnectomicEmpty",
                  "ProviderCooledBadge", "RelativeTime"]:
        text = (comp_dir / f"{name}.tsx").read_text()
        assert f"export function {name}" in text, (
            f"{name}.tsx missing `export function {name}`"
        )


def test_typescript_compiles_clean() -> None:
    """bert/v4 must pass tsc --noEmit with no errors."""
    result = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=LAB_ROOT / "bert" / "v4",
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"tsc failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_audit_doc_exists() -> None:
    audit = LAB_ROOT / "findings" / "bert_v4" / "11_surface_audit.md"
    assert audit.exists()
    text = audit.read_text()
    # All 11 surfaces named
    for surface in ("FirstLight", "Atlas", "Choreography", "DevGestures",
                    "Diagnostics", "KeyboardHelp", "Loom", "Manuscript",
                    "Meeting", "Onboarding", "Tide"):
        assert surface in text, f"audit doc missing surface {surface}"
    # All 5 components named
    for comp in ("StratumSkeleton", "LabIsThinking", "ConnectomicEmpty",
                 "ProviderCooledBadge", "RelativeTime"):
        assert comp in text, f"audit doc missing component {comp}"


def test_atlas_uses_connectomic_empty() -> None:
    """Atlas surface must import + use ConnectomicEmpty."""
    atlas = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Atlas.tsx"
    text = atlas.read_text()
    assert 'import { ConnectomicEmpty }' in text
    assert "<ConnectomicEmpty" in text


def test_strata_pulse_keyframe_present() -> None:
    """StratumSkeleton must inline @keyframes stratum-pulse."""
    text = (LAB_ROOT / "bert" / "v4" / "src" / "components"
            / "StratumSkeleton.tsx").read_text()
    assert "@keyframes stratum-pulse" in text


def test_relative_time_self_refresh() -> None:
    """RelativeTime must use setTimeout for self-refresh (not setInterval —
    timeout lets the refresh cadence change as the timestamp ages)."""
    text = (LAB_ROOT / "bert" / "v4" / "src" / "components"
            / "RelativeTime.tsx").read_text()
    assert "setTimeout" in text
    assert "useEffect" in text
    assert "title=" in text  # absolute ts on hover


def main() -> int:
    tests = [
        test_five_components_exist,
        test_components_export_named_component,
        test_typescript_compiles_clean,
        test_audit_doc_exists,
        test_atlas_uses_connectomic_empty,
        test_strata_pulse_keyframe_present,
        test_relative_time_self_refresh,
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
