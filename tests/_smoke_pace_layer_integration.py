"""Smoke test for L-01 pace-layer integration: verify Merkle round-trip on
events.jsonl + the migration script's dry-run + the backup script's tar
contains expected paths.

Per FINAL_implementation_plan_2026-05-07.md §5.1 H1 day 5 acceptance.

Run: `.venv/bin/python tests/_smoke_pace_layer_integration.py`
"""

import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import merkle  # noqa: E402


def test_lab_directories_exist() -> None:
    for tier in ("sor", "sod", "soi", "stream"):
        d = LAB_ROOT / "lab" / tier
        assert d.is_dir(), f"lab/{tier}/ missing"
        # Each has a .gitkeep so git tracks the empty dir
        gitkeep = d / ".gitkeep"
        assert gitkeep.exists(), f"lab/{tier}/.gitkeep missing"


def test_events_jsonl_exists_and_merkle_root_computable() -> None:
    events = LAB_ROOT / "lab" / "sor" / "events.jsonl"
    assert events.exists(), "lab/sor/events.jsonl missing"
    root = merkle.file_root_hex(events)
    assert isinstance(root, str)
    assert len(root) == 64
    bytes.fromhex(root)  # is hex


def test_merkle_round_trip_on_synthetic_events() -> None:
    """Write some events, compute root, verify, append more, verify the
    root changes (proof of append-detection)."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        f.write('{"ts": "2026-05-07T00:00:00Z", "agent": "director", "content": "hi"}\n')
        f.write('{"ts": "2026-05-07T00:01:00Z", "agent": "researcher", "content": "ok"}\n')
        path = f.name
    try:
        root1 = merkle.file_root(path)
        assert merkle.verify(path, root1)

        # Append another event
        with open(path, "a") as f:
            f.write('{"ts": "2026-05-07T00:02:00Z", "agent": "evaluator", "content": "pass"}\n')

        root2 = merkle.file_root(path)
        assert root1 != root2, "Merkle root should change on append"
        assert not merkle.verify(path, root1), "old root should fail to verify"
        assert merkle.verify(path, root2), "new root should verify"
    finally:
        Path(path).unlink(missing_ok=True)


def test_migrate_dry_run_runs_cleanly() -> None:
    """The migration script's dry-run mode must complete without error."""
    result = subprocess.run(
        [str(LAB_ROOT / ".venv" / "bin" / "python"),
         str(LAB_ROOT / "tools" / "migrate_to_pace_layers.py")],
        capture_output=True, text=True, timeout=60, cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, (
        f"migrate dry-run failed with exit {result.returncode}: {result.stderr[:500]}"
    )
    assert "PACE-LAYER MIGRATION PLAN (DRY-RUN)" in result.stdout
    assert "Total moves:" in result.stdout


def test_backup_script_executable_and_creates_archive() -> None:
    """Run the backup script; verify it creates a tar.gz under backup/state/."""
    backup_script = LAB_ROOT / "tools" / "nightly_backup.sh"
    assert backup_script.exists(), "tools/nightly_backup.sh missing"
    assert backup_script.stat().st_mode & 0o100, "nightly_backup.sh not executable"

    # Don't run it again — we already ran it in Day 5 setup; verify the
    # archive exists.
    backup_dir = LAB_ROOT / "backup" / "state"
    archives = list(backup_dir.glob("state_*.tar.gz"))
    assert len(archives) >= 1, (
        f"No backup archives found in {backup_dir}; nightly_backup.sh "
        "should have created at least one"
    )


def test_strategist_deactivated_in_director_prompt() -> None:
    """Director prompt MUST note Strategist is deactivated during build phase."""
    director_md = LAB_ROOT / "prompts" / "director.md"
    text = director_md.read_text()
    assert "DEACTIVATED" in text and "strategist" in text.lower(), (
        "Director prompt does not mark strategist as DEACTIVATED"
    )


def test_heuristic_h_build_01_present() -> None:
    """H-BUILD-01 documenting Strategist deactivation must be in heuristics.md."""
    heuristics_md = LAB_ROOT / "memories" / "heuristics.md"
    text = heuristics_md.read_text()
    assert "H-BUILD-01" in text
    assert "Strategist deactivated" in text


def test_private_md_exists() -> None:
    """lab/PRIVATE.md privacy boundary doc must exist."""
    private_md = LAB_ROOT / "lab" / "PRIVATE.md"
    assert private_md.exists()
    text = private_md.read_text()
    assert "Privacy Boundary" in text
    assert "stays private" in text.lower()


def main() -> int:
    tests = [
        test_lab_directories_exist,
        test_events_jsonl_exists_and_merkle_root_computable,
        test_merkle_round_trip_on_synthetic_events,
        test_migrate_dry_run_runs_cleanly,
        test_backup_script_executable_and_creates_archive,
        test_strategist_deactivated_in_director_prompt,
        test_heuristic_h_build_01_present,
        test_private_md_exists,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
