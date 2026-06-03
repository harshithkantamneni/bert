"""Smoke test for I.4: bert verify thin wrapper."""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import proof_packet, verify_packet


def _build_packet(cycle_id: int = 400) -> tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="bert_i4_"))
    path = proof_packet.build_packet(cycle_id=cycle_id, output_dir=tmp)
    return tmp, path


def test_module_exports() -> None:
    assert hasattr(verify_packet, "verify_packet")
    assert hasattr(verify_packet, "verify_chain")
    assert hasattr(verify_packet, "format_result")
    assert verify_packet.CheckStatus.PASS.value == "PASS"
    assert verify_packet.CheckStatus.WARN.value == "WARN"
    assert verify_packet.CheckStatus.FAIL.value == "FAIL"


def test_verify_nonexistent_packet_fails() -> None:
    r = verify_packet.verify_packet(Path("/does/not/exist.tar.gz"))
    assert r.overall == "FAIL"


def test_verify_real_packet_returns_pass_with_warnings() -> None:
    """A freshly-built packet should be PASS-WITH-WARNINGS (warnings
    from local-dev mode Rekor/RFC3161 placeholders)."""
    tmp, path = _build_packet(cycle_id=400)
    try:
        r = verify_packet.verify_packet(path)
        assert r.overall in ("PASS", "PASS-WITH-WARNINGS"), (
            f"expected PASS/PASS-WITH-WARNINGS, got {r.overall}: "
            f"{[(c.name, c.status.value, c.detail) for c in r.checks]}"
        )
        # All 8 checks must run
        assert len(r.checks) == 8
        # The local-dev warnings should show up on checks 3 and 4
        warn_names = [c.name for c in r.checks if c.status == verify_packet.CheckStatus.WARN]
        assert any("Rekor" in n for n in warn_names)
        assert any("RFC3161" in n for n in warn_names)
    finally:
        shutil.rmtree(tmp)


def test_verify_pass_specific_checks() -> None:
    """The HASHES, SLSA subject digest, and failures checks must PASS."""
    tmp, path = _build_packet(cycle_id=400)
    try:
        r = verify_packet.verify_packet(path)
        names_pass = [c.name for c in r.checks if c.status == verify_packet.CheckStatus.PASS]
        assert any("HASHES signature" in n for n in names_pass)
        assert any("SLSA subject digest" in n for n in names_pass)
        assert any("failures.md present" in n for n in names_pass)
        assert any("HASHES manifest matches" in n for n in names_pass)
    finally:
        shutil.rmtree(tmp)


def test_verify_detects_tampering() -> None:
    """If we tamper with cycle.json after sealing, verify must FAIL."""
    tmp, path = _build_packet(cycle_id=400)
    extract_tmp = Path(tempfile.mkdtemp(prefix="bert_i4_t_"))
    try:
        # Extract, tamper, re-tar
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(extract_tmp, filter="data")
        pdir = extract_tmp / "cycle-0400"
        cj = json.loads((pdir / "cycle.json").read_text())
        cj["_tampered"] = True
        (pdir / "cycle.json").write_text(json.dumps(cj, indent=2))
        tampered_path = tmp / "tampered.tar.gz"
        with tarfile.open(tampered_path, "w:gz") as tar:
            tar.add(pdir, arcname="cycle-0400")
        r = verify_packet.verify_packet(tampered_path)
        assert r.overall == "FAIL"
        # Specifically: check 5 (subject digest) or check 8 (HASHES) should fail
        failed = [c for c in r.checks if c.status == verify_packet.CheckStatus.FAIL]
        assert len(failed) >= 1, "should have caught at least one failure"
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(extract_tmp)


def test_verify_detects_missing_failures_md() -> None:
    """Removing failures.md should FAIL check [7]."""
    tmp, path = _build_packet(cycle_id=400)
    extract_tmp = Path(tempfile.mkdtemp(prefix="bert_i4_t2_"))
    try:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(extract_tmp, filter="data")
        pdir = extract_tmp / "cycle-0400"
        (pdir / "failures.md").unlink()
        # Note: removing also breaks check 8 (HASHES manifest); that's fine
        bad_path = tmp / "bad.tar.gz"
        with tarfile.open(bad_path, "w:gz") as tar:
            tar.add(pdir, arcname="cycle-0400")
        r = verify_packet.verify_packet(bad_path)
        # Check 7 must report FAIL with the specific reason
        check_7 = next(c for c in r.checks if "[7]" in c.name)
        assert check_7.status == verify_packet.CheckStatus.FAIL
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(extract_tmp)


def test_format_result_includes_cosign_equivalent_on_fail() -> None:
    """CTO-friend test: failure output prints cosign reproduction command."""
    r = verify_packet.VerifyResult(
        packet_path="test.tar.gz", overall="FAIL",
        checks=[verify_packet.CheckResult(
            name="[1] HASHES signature valid",
            status=verify_packet.CheckStatus.FAIL,
            detail="forged signature",
            cosign_equivalent=("cosign verify-blob --key cosign.pub --signature "
                               "HASHES.sig HASHES.txt --insecure-ignore-tlog=true"),
        )],
    )
    out = verify_packet.format_result(r, color=False)
    assert "Reproduce:" in out
    assert "cosign verify-blob --key" in out


def test_verify_chain_links_match() -> None:
    """A 2-packet chain where packet B declares packet A as parent should
    return chain_ok=True."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i4_chain_"))
    try:
        # Build packet A (cycle 99)
        pa = proof_packet.build_packet(cycle_id=99, output_dir=tmp)
        # Compute pa's cycle.json digest
        import hashlib
        with tarfile.open(pa, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        cj_a_bytes = (tmp / "cycle-0099" / "cycle.json").read_bytes()
        a_digest = "sha256:" + hashlib.sha256(cj_a_bytes).hexdigest()
        # Build packet B (cycle 400) declaring A as parent
        pb = proof_packet.build_packet(
            cycle_id=400, output_dir=tmp,
            parent_cycle_id=99, parent_digest=a_digest,
        )
        chain = verify_packet.verify_chain([pa, pb])
        assert chain["chain_ok"] is True, chain
        assert len(chain["chain_links"]) == 1
        assert chain["chain_links"][0]["linked"] is True
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    tests = [
        test_module_exports,
        test_verify_nonexistent_packet_fails,
        test_verify_real_packet_returns_pass_with_warnings,
        test_verify_pass_specific_checks,
        test_verify_detects_tampering,
        test_verify_detects_missing_failures_md,
        test_format_result_includes_cosign_equivalent_on_fail,
        test_verify_chain_links_match,
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
