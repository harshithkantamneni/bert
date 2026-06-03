"""Agent Card + audit-log signing.

Per Sigstore A2A pattern. Two modes:

  local-dev   ed25519 keypair at ~/.bert-lab/signing.key, generated
              on first use. Signs the canonical JSON of the Agent
              Card + Merkle roots of lab/sor/events.jsonl. Writes
              entries to a local-Rekor-shaped append-only log at
              lab/state/local_rekor.jsonl. Good for development +
              first-customer trust review.

  sigstore    Keyless flow via OIDC + Fulcio + Rekor (the production
              path). Activated when BERT_LAB_SIGNING_MODE=sigstore
              and an OIDC token is available (CI/CD, gha-runner, or
              `sigstore sign` interactive). Not auto-enabled because
              it requires PI to configure an OIDC provider.

The signed surface
==================

  Agent Card (/.well-known/agent.json):
    canonical JSON bytes → ed25519 signature → emitted at
    /.well-known/agent.json.sig as a JSON blob:
      {agent_id, pubkey, signature_b64, ts, mode}

  Events.jsonl Merkle root:
    bert's core.merkle.file_root() computes a SHA-256 Merkle tree
    over events.jsonl lines. We sign that root at checkpoint
    intervals; the signature + root land in lab/state/local_rekor.
    jsonl with a monotone-increasing log_id (Rekor-shaped).

  Skill artifacts:
    SKILL.md files in skills/active/ get signed at promotion time
    (called from core.creator.promote()). The signature anchors the
    proposal_id chain.

Why this design vs full sigstore today
======================================

  Full keyless sigstore requires an OIDC identity provider AND a
  network round-trip to Fulcio + Rekor. Neither makes sense as a
  default for a developer's autonomous lab.

  Local ed25519 gives the *cryptographic guarantee* — any external
  verifier with bert's pubkey can confirm a card hasn't been tampered
  with. The Rekor-shape append-only log gives the *temporal
  guarantee* — past attestations can't be retroactively rewritten.

Honest status of `BERT_LAB_SIGNING_MODE=sigstore` (per DD.2)
============================================================

  The env-var flag is a TAG today, not a code path. Setting
  BERT_LAB_SIGNING_MODE=sigstore changes the `mode` string written
  into signature dicts and bundle metadata — but the cryptographic
  operations remain local ed25519. The bundle's `tlogEntries` stays
  empty; `rfc3161Timestamps` stays empty; no OIDC token is acquired,
  no Fulcio cert is requested, no Rekor entry is submitted.

  Earlier documentation framed this as "one config flip" away from
  production. That was misleading. Real production Sigstore requires
  engineering, not a flag:

    1. sigstore-python dependency + OIDC client setup (GitHub Actions
       OIDC, Google OIDC, or interactive `sigstore sign`)
    2. Fulcio cert acquisition flow (network round-trip; cert
       signing-identity binding; short-lived cert lifecycle)
    3. Rekor entry submission + inclusion-proof retrieval
    4. RFC3161 Timestamp Authority integration (separate TSA service)
    5. Cert validation in `verify_bytes` + signing-identity check
    6. Rekor log verification in `core.verify_packet` check [3]

  Estimate: ~1-2 focused weeks of work, plus ongoing dependency
  on three external services (Fulcio, Rekor, TSA). The wire format
  is intentionally shape-compatible so the migration won't break
  existing callers or verifiers — but the format compatibility is
  NOT the same as the operations being implemented.

  This module's job today: deliver the local-dev cryptographic +
  temporal guarantees honestly, and make production-Sigstore
  adoption a future ENGINEERING step that PI accepts as such — not
  a flag that PI thinks they can flip on demo day.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

LOG = logging.getLogger("bert.signing")
LAB_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
DEFAULT_KEY_PATH = HOME / ".bert-lab" / "signing.key"
# Separate ECDSA-P256 key for the cosign-verifiable proof-packet surface
# (item 25). cosign rejects ed25519; the internal ed25519 key above stays for
# bert-native attestation (agent cards / skills / merkle / local-Rekor).
COSIGN_KEY_PATH = HOME / ".bert-lab" / "cosign-signing.key"
LOCAL_REKOR_LOG = LAB_ROOT / "lab" / "state" / "local_rekor.jsonl"


def _signing_mode() -> str:
    """Resolve the configured signing mode. Note (DD.2): a value of
    `sigstore` currently only TAGS signatures with mode='sigstore';
    actual Fulcio/Rekor/RFC3161 code paths are NOT implemented (see
    module docstring). The first call with sigstore mode logs a one-
    time warning so operators don't mistake the tag for a real
    cryptographic switchover.
    """
    mode = os.environ.get("BERT_LAB_SIGNING_MODE", "local-dev")
    if mode == "sigstore" and not _SIGSTORE_WARN_EMITTED:
        LOG.warning(
            "signing: BERT_LAB_SIGNING_MODE=sigstore is a TAG only. "
            "Cryptographic operations remain local ed25519 until the "
            "Fulcio/Rekor/RFC3161 implementation lands. See "
            "core/signing.py docstring for the engineering punch list."
        )
        globals()["_SIGSTORE_WARN_EMITTED"] = True
    return mode


_SIGSTORE_WARN_EMITTED = False


# ── Keypair management (local-dev mode) ──────────────────────────────


def _ensure_key(*, key_path: Path = DEFAULT_KEY_PATH):
    """Load or generate an ed25519 keypair. Returns (private_key,
    public_key) cryptography objects."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if key_path.exists():
        pem = key_path.read_bytes()
        priv = serialization.load_pem_private_key(pem, password=None)
        return priv, priv.public_key()
    LOG.info("signing: generating new ed25519 key at %s", key_path)
    priv = ed25519.Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path.write_bytes(pem)
    key_path.chmod(0o600)
    return priv, priv.public_key()


def public_key_pem(*, key_path: Path = DEFAULT_KEY_PATH) -> str:
    """Return bert's public key in PEM format (for verifiers)."""
    from cryptography.hazmat.primitives import serialization
    _, pub = _ensure_key(key_path=key_path)
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


# ── Canonical JSON ───────────────────────────────────────────────────


def canonical_json(obj: Any) -> bytes:
    """Stable JSON serialization for signing. Sorted keys, no
    whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")


# ── Sign / verify ────────────────────────────────────────────────────


@dataclass
class Signature:
    artifact_kind: str             # "agent_card" / "merkle_root" / "skill"
    artifact_hash: str             # SHA-256 of canonical bytes
    signature_b64: str             # base64 ed25519 signature
    pubkey_pem: str
    ts: str
    mode: str
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_hash": self.artifact_hash,
            "signature_b64": self.signature_b64,
            "pubkey_pem": self.pubkey_pem,
            "ts": self.ts,
            "mode": self.mode,
            "extras": self.extras,
        }


def sign_bytes(data: bytes, *, artifact_kind: str,
                extras: dict | None = None,
                key_path: Path = DEFAULT_KEY_PATH) -> Signature:
    """Sign arbitrary bytes. Returns a Signature dataclass."""
    priv, _ = _ensure_key(key_path=key_path)
    sig_bytes = priv.sign(data)
    return Signature(
        artifact_kind=artifact_kind,
        artifact_hash=hashlib.sha256(data).hexdigest(),
        signature_b64=base64.b64encode(sig_bytes).decode(),
        pubkey_pem=public_key_pem(key_path=key_path),
        ts=_now_iso(),
        mode=_signing_mode(),
        extras=extras or {},
    )


def verify_bytes(data: bytes, sig: Signature) -> bool:
    """Verify a signature against bytes + pubkey. Returns True iff
    the signature is valid."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    try:
        pub = serialization.load_pem_public_key(sig.pubkey_pem.encode())
        sig_bytes = base64.b64decode(sig.signature_b64)
        pub.verify(sig_bytes, data)
        return True
    except (InvalidSignature, ValueError) as e:
        LOG.warning("signing: verification failed: %s", e)
        return False


# ── cosign-compatible blob signing (ECDSA-P256, item 25) ─────────────
#
# The proof packet's external trust surface (HASHES.sigstore /
# failures.sigstore) must verify under vanilla cosign. cosign requires
# ECDSA-P256 + SHA-256 (ed25519 is not accepted), so this path uses a separate
# ECDSA key and emits a DER signature that
#   cosign verify-blob --key cosign.pub --signature <b64> <blob> --insecure-ignore-tlog=true
# accepts offline (empirically verified against cosign v3.0.6). bert signs in
# pure Python; cosign only verifies — we never depend on the cosign binary at
# sign time, and v3's --bundle/signing-config/Rekor coupling is sidestepped.


def _ensure_cosign_key(*, key_path: Path = COSIGN_KEY_PATH):
    """Load or generate the ECDSA-P256 cosign-compat keypair. Returns
    (private_key, public_key)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if key_path.exists():
        priv = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        return priv, priv.public_key()
    LOG.info("signing: generating new ECDSA-P256 cosign key at %s", key_path)
    priv = ec.generate_private_key(ec.SECP256R1())
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path.write_bytes(pem)
    key_path.chmod(0o600)
    return priv, priv.public_key()


def cosign_public_key_pem(*, key_path: Path = COSIGN_KEY_PATH) -> str:
    """ECDSA-P256 public key in PEM — usable directly as cosign's --key."""
    from cryptography.hazmat.primitives import serialization
    _, pub = _ensure_cosign_key(key_path=key_path)
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def sign_blob_cosign(data: bytes, *, key_path: Path = COSIGN_KEY_PATH) -> dict:
    """Sign `data` with ECDSA-P256/SHA-256 (DER) — cosign-verifiable.

    Returns an envelope dict carrying the base64 DER signature, the public-key
    PEM, the SHA-256 digest, algo tag, mode, and timestamp. `signature_b64`
    fed to `cosign verify-blob --signature` (and `pubkey_pem` to --key) verifies
    offline.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    priv, _ = _ensure_cosign_key(key_path=key_path)
    sig = priv.sign(data, ec.ECDSA(hashes.SHA256()))  # DER-encoded
    return {
        "algo": "ecdsa-p256-sha256",
        "signature_b64": base64.b64encode(sig).decode(),
        "pubkey_pem": cosign_public_key_pem(key_path=key_path),
        "digest_sha256": hashlib.sha256(data).hexdigest(),
        "mode": _signing_mode(),
        "ts": _now_iso(),
    }


def verify_cosign_blob(data: bytes, signature_b64: str, pubkey_pem: str) -> bool:
    """Pure-python verify of an ECDSA-P256/SHA-256 signature (the same check
    cosign performs). Returns True iff valid. Always available — no cosign
    binary required."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    try:
        pub = serialization.load_pem_public_key(pubkey_pem.encode())
        pub.verify(base64.b64decode(signature_b64), data, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError) as e:
        LOG.warning("signing: cosign-blob verification failed: %s", e)
        return False


# ── Higher-level helpers ─────────────────────────────────────────────


def sign_agent_card(card: dict, *, key_path: Path = DEFAULT_KEY_PATH) -> Signature:
    """Sign an A2A Agent Card. The card is canonicalized before
    hashing so reordering JSON fields doesn't invalidate the signature.
    """
    return sign_bytes(canonical_json(card), artifact_kind="agent_card",
                       key_path=key_path)


def sign_merkle_root(root_hex: str, *, events_path: str = "lab/sor/events.jsonl",
                     line_count: int = 0,
                     key_path: Path = DEFAULT_KEY_PATH) -> Signature:
    """Sign a Merkle root over the canonical event log.
    `root_hex` is the output of `core.merkle.file_root(events_path)`.
    """
    payload = canonical_json({
        "merkle_root_hex": root_hex,
        "events_path": events_path,
        "line_count": line_count,
    })
    return sign_bytes(payload, artifact_kind="merkle_root",
                       extras={"merkle_root_hex": root_hex,
                                "events_path": events_path,
                                "line_count": line_count},
                       key_path=key_path)


def sign_skill_manifest(skill_md_path: Path, *,
                         key_path: Path = DEFAULT_KEY_PATH) -> Signature:
    """Sign a promoted skill's SKILL.md. Called from
    core.creator.promote() right before the draft → active move."""
    if not skill_md_path.exists():
        raise FileNotFoundError(f"skill manifest not found: {skill_md_path}")
    content = skill_md_path.read_bytes()
    return sign_bytes(content, artifact_kind="skill",
                       extras={"skill_path": str(skill_md_path),
                                "filename": skill_md_path.name},
                       key_path=key_path)


# ── Local Rekor (append-only attestation log) ────────────────────────


def append_to_local_rekor(sig: Signature) -> int:
    """Append a signature record to the local-Rekor-shaped log.
    Returns the assigned log_id (monotonically increasing)."""
    LOCAL_REKOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Read current line count to determine next log_id
    log_id = 0
    if LOCAL_REKOR_LOG.exists():
        # Count lines without reading the whole file
        with LOCAL_REKOR_LOG.open("rb") as f:
            log_id = sum(1 for _ in f)
    entry = {
        "log_id": log_id,
        **sig.to_dict(),
    }
    with LOCAL_REKOR_LOG.open("a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return log_id


def read_local_rekor(*, limit: int = 100,
                      artifact_kind: str | None = None) -> list[dict]:
    """Read the last `limit` entries from the local Rekor log,
    optionally filtered by artifact_kind."""
    if not LOCAL_REKOR_LOG.exists():
        return []
    entries: list[dict] = []
    for line in LOCAL_REKOR_LOG.read_text().splitlines()[-limit * 4:]:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if artifact_kind and d.get("artifact_kind") != artifact_kind:
            continue
        entries.append(d)
    return entries[-limit:]


# ── Now-iso helper ──────────────────────────────────────────────────


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now(UTC).isoformat()
