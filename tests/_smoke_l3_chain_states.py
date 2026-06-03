"""Smoke test for L.3: chain-state distinction (linked / unlinked / mismatch).

L.3 introduced 3-state chain semantics:
  linked    — curr.parentCycleId == prev.cycle_id (real lineage)
  unlinked  — curr declared NO parent (valid for first cycle)
  mismatch  — curr declared a parent but it doesn't match (integrity failure)

Exit codes from `bert verify --chain`:
  0  Chain OK (every link is "linked")
  1  Chain unlinked (no mismatch, but at least one link is "unlinked")
  2  Chain BROKEN (at least one "mismatch")
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import proof_packet, verify_packet

# These tests build proof packets for synthetic cycles 99 and 400, which
# requires the active lab's events.jsonl to contain events for those cycles.
# That events.jsonl is a live-lab runtime artifact and is NOT shipped/tracked
# in the public repo, so the packet build raises "cycle N has no events".
_REQUIRED_CYCLES = (99, 400)


def _require_cycle_events(*cycle_ids: int) -> None:
    events_path, _, _ = proof_packet._active_lab_paths()
    if not events_path.exists():
        pytest.skip("requires lab runtime artifact not shipped in the public repo")
    present: set[int] = set()
    with events_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if "cycle" in ev:
                present.add(ev["cycle"])
    missing = [c for c in cycle_ids if c not in present]
    if missing:
        pytest.skip("requires lab runtime artifact not shipped in the public repo")


def _build_two_packets(tmp: Path, link_state: str) -> tuple[Path, Path]:
    """Build two packets with a chain relationship per link_state.

    link_state ∈ {"linked", "unlinked", "mismatch"}.
    """
    # Packet A (cycle 99)
    pa = proof_packet.build_packet(cycle_id=99, output_dir=tmp)
    # Compute A's cycle.json digest for the "linked" case
    with tarfile.open(pa, "r:gz") as t:
        t.extractall(tmp, filter="data")
    cj_a_bytes = (tmp / "cycle-0099" / "cycle.json").read_bytes()
    a_digest = "sha256:" + hashlib.sha256(cj_a_bytes).hexdigest()

    if link_state == "linked":
        pb = proof_packet.build_packet(
            cycle_id=400, output_dir=tmp,
            parent_cycle_id=99, parent_digest=a_digest,
        )
    elif link_state == "unlinked":
        pb = proof_packet.build_packet(
            cycle_id=400, output_dir=tmp,
            # No parent declared
        )
    elif link_state == "mismatch":
        pb = proof_packet.build_packet(
            cycle_id=400, output_dir=tmp,
            parent_cycle_id=999, parent_digest="sha256:WRONG_PARENT",
        )
    else:
        raise ValueError(link_state)
    return pa, pb


def test_linked_chain_returns_ok() -> None:
    _require_cycle_events(*_REQUIRED_CYCLES)
    tmp = Path(tempfile.mkdtemp(prefix="bert_l3_linked_"))
    try:
        pa, pb = _build_two_packets(tmp, "linked")
        result = verify_packet.verify_chain([pa, pb])
        assert result["chain_ok"] is True, result
        assert result["has_mismatch"] is False
        link = result["chain_links"][0]
        assert link["state"] == "linked"
        assert link["linked"] is True
    finally:
        shutil.rmtree(tmp)


def test_unlinked_chain_is_not_broken() -> None:
    """No declared parent → state=unlinked, NOT mismatch. exit 1, not 2."""
    _require_cycle_events(*_REQUIRED_CYCLES)
    tmp = Path(tempfile.mkdtemp(prefix="bert_l3_unlinked_"))
    try:
        pa, pb = _build_two_packets(tmp, "unlinked")
        result = verify_packet.verify_chain([pa, pb])
        assert result["chain_ok"] is False, "unlinked is not OK"
        assert result["has_mismatch"] is False, "unlinked is NOT a mismatch"
        link = result["chain_links"][0]
        assert link["state"] == "unlinked"
        assert link["linked"] is False
    finally:
        shutil.rmtree(tmp)


def test_mismatch_chain_is_broken() -> None:
    """Wrong parent declared → state=mismatch. exit 2 (integrity failure)."""
    _require_cycle_events(*_REQUIRED_CYCLES)
    tmp = Path(tempfile.mkdtemp(prefix="bert_l3_mismatch_"))
    try:
        pa, pb = _build_two_packets(tmp, "mismatch")
        result = verify_packet.verify_chain([pa, pb])
        assert result["chain_ok"] is False
        assert result["has_mismatch"] is True, \
            "mismatch state must set has_mismatch=True"
        link = result["chain_links"][0]
        assert link["state"] == "mismatch"
        assert link["linked"] is False
    finally:
        shutil.rmtree(tmp)


def test_cli_exit_code_0_on_linked() -> None:
    """`bert verify --chain` exits 0 when fully linked."""
    _require_cycle_events(*_REQUIRED_CYCLES)
    tmp = Path(tempfile.mkdtemp(prefix="bert_l3_cli_linked_"))
    try:
        pa, pb = _build_two_packets(tmp, "linked")
        result = subprocess.run(
            [".venv/bin/python", "tools/bert_verify.py",
             str(pa), str(pb), "--chain", "--no-color"],
            cwd=LAB_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"linked chain should exit 0; got {result.returncode}\n{result.stdout}"
        )
        assert "Chain OK" in result.stdout
    finally:
        shutil.rmtree(tmp)


def test_cli_exit_code_1_on_unlinked() -> None:
    """`bert verify --chain` exits 1 (informational) when unlinked, not 2."""
    _require_cycle_events(*_REQUIRED_CYCLES)
    tmp = Path(tempfile.mkdtemp(prefix="bert_l3_cli_unlinked_"))
    try:
        pa, pb = _build_two_packets(tmp, "unlinked")
        result = subprocess.run(
            [".venv/bin/python", "tools/bert_verify.py",
             str(pa), str(pb), "--chain", "--no-color"],
            cwd=LAB_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1, (
            f"unlinked chain should exit 1; got {result.returncode}\n{result.stdout}"
        )
        assert "Chain unlinked" in result.stdout
        assert "BROKEN" not in result.stdout
    finally:
        shutil.rmtree(tmp)


def test_cli_exit_code_2_on_mismatch() -> None:
    """`bert verify --chain` exits 2 when there's a real integrity failure."""
    _require_cycle_events(*_REQUIRED_CYCLES)
    tmp = Path(tempfile.mkdtemp(prefix="bert_l3_cli_mismatch_"))
    try:
        pa, pb = _build_two_packets(tmp, "mismatch")
        result = subprocess.run(
            [".venv/bin/python", "tools/bert_verify.py",
             str(pa), str(pb), "--chain", "--no-color"],
            cwd=LAB_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 2, (
            f"mismatch chain should exit 2; got {result.returncode}\n{result.stdout}"
        )
        assert "Chain BROKEN" in result.stdout
        assert "integrity failure" in result.stdout
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    tests = [
        test_linked_chain_returns_ok,
        test_unlinked_chain_is_not_broken,
        test_mismatch_chain_is_broken,
        test_cli_exit_code_0_on_linked,
        test_cli_exit_code_1_on_unlinked,
        test_cli_exit_code_2_on_mismatch,
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
