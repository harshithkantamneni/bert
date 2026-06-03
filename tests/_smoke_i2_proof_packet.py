"""Smoke test for I.2: proof packet exporter v1."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import proof_packet, signing


def test_module_exports() -> None:
    assert hasattr(proof_packet, "build_packet")
    assert hasattr(proof_packet, "build_cycle_json")
    assert proof_packet.SCHEMA_VERSION == "bert.proof.v1"


def test_build_cycle_json_shape() -> None:
    """Shape matches the locked schema."""
    cj = proof_packet.build_cycle_json(
        cycle_id=42,
        parent_cycle_id=41,
        parent_digest="sha256:abc",
        events=[{"ts": "2026-05-13T20:00:00+00:00", "agent": "researcher",
                 "verdict": "APPROVE", "cycle": 42}],
    )
    assert cj["schemaVersion"] == "bert.proof.v1"
    assert cj["cycleId"] == "cycle-0042"
    assert cj["parentCycleId"] == "cycle-0041"
    assert cj["parentDigest"] == "sha256:abc"
    assert cj["predicateType"] == "https://bert.dev/cycle/v1"
    assert cj["subject"] == {"name": "cycle-0042"}
    assert "claims" in cj and "limitations" in cj
    assert cj["eventCount"] == 1


def test_empty_cycle_raises() -> None:
    """Building a packet for a cycle with no events must raise."""
    try:
        proof_packet.build_packet(cycle_id=999_999_999)
        raise AssertionError("expected ValueError for no events")
    except ValueError as e:
        assert "no events" in str(e)


def test_build_packet_produces_tarball_with_required_files() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_i2_"))
    try:
        # Build for a known real cycle (400) — assumes events.jsonl has it
        path = proof_packet.build_packet(cycle_id=400, output_dir=tmp)
        assert path.exists()
        assert path.suffix == ".gz"
        with tarfile.open(path, "r:gz") as tar:
            names = tar.getnames()
        # Required files per schema
        prefix = "cycle-0400/"
        required = [
            prefix + "manifest.json",
            prefix + "README.md",
            prefix + "cycle.json",
            prefix + "HASHES.txt",
            prefix + "HASHES.sigstore",
            prefix + "provenance/slsa.intoto.jsonl",
            prefix + "provenance/slsa.sigstore",
            prefix + "provenance/trusted-root.json",
            prefix + "inputs/prompt.txt",
            prefix + "inputs/seed.json",
            prefix + "outputs/results.md",
            prefix + "outputs/metrics.json",
            prefix + "failures.md",  # I.3 makes it real
            prefix + "reproduce.sh",
            prefix + "eval/self-eval.json",
            prefix + "eval/adversarial.json",  # I.3 makes it real
        ]
        for r in required:
            assert r in names, f"missing required file {r}"
    finally:
        shutil.rmtree(tmp)


def test_slsa_envelope_well_formed() -> None:
    """The slsa.intoto.jsonl line is a valid DSSE envelope with SLSA v1 predicate."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i2_"))
    try:
        path = proof_packet.build_packet(cycle_id=400, output_dir=tmp)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp)
        env_path = tmp / "cycle-0400" / "provenance" / "slsa.intoto.jsonl"
        env = json.loads(env_path.read_text())
        assert env["payloadType"] == "application/vnd.in-toto+json"
        statement = json.loads(base64.b64decode(env["payload"]))
        assert statement["_type"] == "https://in-toto.io/Statement/v1"
        assert statement["predicateType"] == "https://slsa.dev/provenance/v1"
        assert statement["subject"][0]["name"] == "cycle-0400"
        assert "sha256" in statement["subject"][0]["digest"]
        assert "buildDefinition" in statement["predicate"]
        assert "runDetails" in statement["predicate"]
    finally:
        shutil.rmtree(tmp)


def test_signatures_verify() -> None:
    """The Sigstore bundle signature must verify against the signed bytes."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i2_"))
    try:
        path = proof_packet.build_packet(cycle_id=400, output_dir=tmp)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp)
        pdir = tmp / "cycle-0400"
        # HASHES signature — now a cosign-compatible ECDSA-P256 envelope (item 25)
        hashes_text = (pdir / "HASHES.txt").read_bytes()
        hashes_env = json.loads((pdir / "HASHES.sigstore").read_text())
        assert hashes_env["algo"] == "ecdsa-p256-sha256"
        assert signing.verify_cosign_blob(
            hashes_text, hashes_env["signature_b64"], hashes_env["pubkey_pem"]
        ), "HASHES signature should verify (ECDSA, cosign-compatible)"
        # cosign-native siblings present + consistent with the envelope
        assert (pdir / "cosign.pub").read_text() == hashes_env["pubkey_pem"]
        assert (pdir / "HASHES.sig").read_text() == hashes_env["signature_b64"]

        # The spec AC, end-to-end: a fully built packet must verify under
        # VANILLA cosign (the oracle). Skip-with-message when cosign absent.
        cosign = shutil.which("cosign")
        if cosign:
            r = subprocess.run(
                [cosign, "verify-blob", "--key", str(pdir / "cosign.pub"),
                 "--signature", str(pdir / "HASHES.sig"),
                 "--insecure-ignore-tlog=true", str(pdir / "HASHES.txt")],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, f"vanilla cosign rejected the packet: {r.stderr.strip()}"
        else:
            print("    (cosign not on PATH — skipped vanilla-cosign packet roundtrip)")

        # SLSA envelope signature — signed over the canonical statement bytes
        env = json.loads((pdir / "provenance" / "slsa.intoto.jsonl").read_text())
        statement = json.loads(base64.b64decode(env["payload"]))
        # The signed bytes are canonical_json of the statement
        statement_bytes = signing.canonical_json(statement)
        slsa_bundle = json.loads((pdir / "provenance" / "slsa.sigstore").read_text())
        slsa_sig = signing.Signature(
            artifact_kind="proof_packet_slsa",
            artifact_hash=slsa_bundle["messageSignature"]["messageDigest"]["digest"],
            signature_b64=slsa_bundle["messageSignature"]["signature"],
            pubkey_pem=slsa_bundle["verificationMaterial"]["publicKey"]["rawBytes"],
            ts=slsa_bundle["_signedAt"],
            mode="local-dev",
        )
        assert signing.verify_bytes(statement_bytes, slsa_sig), \
            "SLSA envelope signature should verify"
    finally:
        shutil.rmtree(tmp)


def test_hashes_file_matches_actual_files() -> None:
    """Every file in the packet (except HASHES.*) must be in HASHES.txt with correct hash."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i2_"))
    try:
        path = proof_packet.build_packet(cycle_id=400, output_dir=tmp)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp)
        pdir = tmp / "cycle-0400"
        hashes_lines = (pdir / "HASHES.txt").read_text().strip().splitlines()
        declared = {}
        for line in hashes_lines:
            h, p = line.split("  ", 1)
            declared[p] = h
        # Compute actual hashes for files NOT in HASHES list
        for p in pdir.rglob("*"):
            if not p.is_file():
                continue
            if p.name.startswith("HASHES"):
                continue
            # manifest.json is built AFTER hashes, so it's not in the declared set
            if p.name == "manifest.json":
                continue
            rel = p.relative_to(pdir).as_posix()
            assert rel in declared, f"file {rel} missing from HASHES.txt"
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            assert actual == declared[rel], (
                f"hash mismatch for {rel}: actual={actual[:16]} declared={declared[rel][:16]}"
            )
    finally:
        shutil.rmtree(tmp)


def test_cycle_json_lineage_fields() -> None:
    """parentCycleId + parentDigest are present (may be null for first cycle)."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i2_"))
    try:
        path = proof_packet.build_packet(
            cycle_id=400, output_dir=tmp,
            parent_cycle_id=399, parent_digest="sha256:deadbeef",
        )
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp)
        cj = json.loads((tmp / "cycle-0400" / "cycle.json").read_text())
        assert cj["parentCycleId"] == "cycle-0399"
        assert cj["parentDigest"] == "sha256:deadbeef"
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    tests = [
        test_module_exports,
        test_build_cycle_json_shape,
        test_empty_cycle_raises,
        test_build_packet_produces_tarball_with_required_files,
        test_slsa_envelope_well_formed,
        test_signatures_verify,
        test_hashes_file_matches_actual_files,
        test_cycle_json_lineage_fields,
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
