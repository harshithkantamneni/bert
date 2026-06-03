"""Smoke + TDD: core/signing.py cosign-compatible blob signing (Sprint 5 item 25).

The proof packet's external trust surface must be verifiable by VANILLA cosign
(spec AC). cosign requires ECDSA-P256 + SHA-256 (ed25519 is rejected), so bert
signs blobs with a separate ECDSA-P256 key and emits a signature that
`cosign verify-blob --key <pub> --signature <b64> <blob>` accepts offline.

These tests pin BOTH layers:
  - pure-python roundtrip (always runs): sign -> verify True; tamper -> False.
  - the REAL cosign binary (the oracle): when `cosign` is on PATH, shell out to
    `cosign verify-blob` and assert it accepts our signature + rejects a tamper.
    Skips-with-message when cosign is absent (integration-tier), so the suite is
    honest, never hollow.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import signing  # noqa: E402

_BLOB = b"bert proof packet HASHES.txt content\nsha256:deadbeef  cycle.json\n"


def _tmp_key() -> Path:
    return Path(tempfile.mkdtemp(prefix="bert_cosign_")) / "cosign-signing.key"


def test_sign_blob_cosign_envelope_shape():
    kp = _tmp_key()
    env = signing.sign_blob_cosign(_BLOB, key_path=kp)
    assert env["algo"] == "ecdsa-p256-sha256"
    assert env["signature_b64"] and base64.b64decode(env["signature_b64"])
    assert "BEGIN PUBLIC KEY" in env["pubkey_pem"]
    import hashlib
    assert env["digest_sha256"] == hashlib.sha256(_BLOB).hexdigest()


def test_pure_python_roundtrip_and_tamper():
    kp = _tmp_key()
    env = signing.sign_blob_cosign(_BLOB, key_path=kp)
    assert signing.verify_cosign_blob(_BLOB, env["signature_b64"], env["pubkey_pem"]) is True
    # tampered bytes must NOT verify
    assert signing.verify_cosign_blob(_BLOB + b"x", env["signature_b64"],
                                      env["pubkey_pem"]) is False


def test_pubkey_pem_is_ecdsa_p256():
    # cosign requires ECDSA-P256 — assert the key curve, not just "a key".
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    kp = _tmp_key()
    pem = signing.cosign_public_key_pem(key_path=kp)
    pub = serialization.load_pem_public_key(pem.encode())
    assert isinstance(pub.curve, ec.SECP256R1)


def test_real_cosign_accepts_our_signature():
    # The oracle: real `cosign verify-blob --key --signature` must accept our
    # pure-python ECDSA-P256 signature and reject a tampered blob.
    cosign = shutil.which("cosign")
    if not cosign:
        print("    SKIP  cosign not on PATH (integration-tier — install cosign to run)")
        return
    kp = _tmp_key()
    env = signing.sign_blob_cosign(_BLOB, key_path=kp)
    d = Path(tempfile.mkdtemp(prefix="bert_cosign_oracle_"))
    blob = d / "HASHES.txt"
    blob.write_bytes(_BLOB)
    (d / "cosign.pub").write_text(env["pubkey_pem"])
    (d / "HASHES.sig").write_text(env["signature_b64"])
    ok = subprocess.run(
        [cosign, "verify-blob", "--key", str(d / "cosign.pub"),
         "--signature", str(d / "HASHES.sig"),
         "--insecure-ignore-tlog=true", str(blob)],
        capture_output=True, text=True,
    )
    assert ok.returncode == 0, f"cosign rejected our sig: {ok.stderr.strip()}"
    # tamper the blob — cosign must reject
    blob.write_bytes(_BLOB + b"tampered")
    bad = subprocess.run(
        [cosign, "verify-blob", "--key", str(d / "cosign.pub"),
         "--signature", str(d / "HASHES.sig"),
         "--insecure-ignore-tlog=true", str(blob)],
        capture_output=True, text=True,
    )
    assert bad.returncode != 0, "cosign accepted a tampered blob (should reject)"
    print("    (real cosign verify-blob accepted our sig + rejected tamper)")


def main() -> int:
    tests = [
        test_sign_blob_cosign_envelope_shape,
        test_pure_python_roundtrip_and_tamper,
        test_pubkey_pem_is_ecdsa_p256,
        test_real_cosign_accepts_our_signature,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
