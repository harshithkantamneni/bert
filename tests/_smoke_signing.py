"""Smoke test for core/signing.py (G.4) — Agent Card + Merkle + skill
signing with local ed25519, plus the local-Rekor append-only log."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import signing  # noqa: E402


def _isolate() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="bert_sign_"))
    signing.LOCAL_REKOR_LOG = tmp / "local_rekor.jsonl"
    # Always use a per-test keypair so tests don't pollute ~/.bert-lab
    key_path = tmp / "signing.key"
    return key_path


def test_keypair_generated_on_first_use() -> None:
    key_path = _isolate()
    assert not key_path.exists()
    priv, pub = signing._ensure_key(key_path=key_path)
    assert key_path.exists()
    assert priv is not None
    assert pub is not None


def test_keypair_persists_across_calls() -> None:
    key_path = _isolate()
    pem_1 = signing.public_key_pem(key_path=key_path)
    pem_2 = signing.public_key_pem(key_path=key_path)
    assert pem_1 == pem_2  # same key reused, not regenerated


def test_sign_bytes_roundtrip() -> None:
    key_path = _isolate()
    data = b"hello bert"
    sig = signing.sign_bytes(data, artifact_kind="test", key_path=key_path)
    assert sig.artifact_kind == "test"
    assert len(sig.signature_b64) > 20
    assert sig.pubkey_pem.startswith("-----BEGIN PUBLIC KEY-----")
    assert signing.verify_bytes(data, sig) is True


def test_verify_rejects_tampered_bytes() -> None:
    key_path = _isolate()
    sig = signing.sign_bytes(b"original", artifact_kind="test", key_path=key_path)
    assert signing.verify_bytes(b"tampered", sig) is False


def test_canonical_json_stable_across_key_order() -> None:
    a = {"x": 1, "y": 2, "nested": {"b": 1, "a": 2}}
    b = {"y": 2, "x": 1, "nested": {"a": 2, "b": 1}}
    assert signing.canonical_json(a) == signing.canonical_json(b)


def test_sign_agent_card() -> None:
    key_path = _isolate()
    card = {"name": "bert-lab", "version": "0.1", "skills": [{"id": "x"}]}
    sig = signing.sign_agent_card(card, key_path=key_path)
    assert sig.artifact_kind == "agent_card"
    assert signing.verify_bytes(signing.canonical_json(card), sig) is True


def test_sign_merkle_root() -> None:
    key_path = _isolate()
    sig = signing.sign_merkle_root(
        "deadbeef" * 8, events_path="lab/sor/events.jsonl",
        line_count=42, key_path=key_path,
    )
    assert sig.artifact_kind == "merkle_root"
    assert sig.extras["merkle_root_hex"].startswith("dead")
    assert sig.extras["line_count"] == 42


def test_sign_skill_manifest() -> None:
    key_path = _isolate()
    skill_path = Path(tempfile.mkdtemp()) / "SKILL.md"
    skill_path.write_text("---\nname: test-skill\n---\n# body\n")
    sig = signing.sign_skill_manifest(skill_path, key_path=key_path)
    assert sig.artifact_kind == "skill"
    assert sig.extras["filename"] == "SKILL.md"


def test_append_to_local_rekor_assigns_monotone_ids() -> None:
    _isolate()
    key_path = _isolate()
    sig1 = signing.sign_bytes(b"a", artifact_kind="test", key_path=key_path)
    sig2 = signing.sign_bytes(b"b", artifact_kind="test", key_path=key_path)
    id1 = signing.append_to_local_rekor(sig1)
    id2 = signing.append_to_local_rekor(sig2)
    assert id1 < id2
    entries = signing.read_local_rekor(limit=10)
    assert len(entries) == 2
    assert entries[0]["log_id"] == id1


def test_read_local_rekor_filter_by_kind() -> None:
    key_path = _isolate()
    signing.append_to_local_rekor(
        signing.sign_bytes(b"a", artifact_kind="agent_card", key_path=key_path)
    )
    signing.append_to_local_rekor(
        signing.sign_bytes(b"b", artifact_kind="merkle_root", key_path=key_path)
    )
    signing.append_to_local_rekor(
        signing.sign_bytes(b"c", artifact_kind="skill", key_path=key_path)
    )
    cards = signing.read_local_rekor(artifact_kind="agent_card")
    assert len(cards) == 1
    assert cards[0]["artifact_kind"] == "agent_card"


def test_sign_skill_missing_file_raises() -> None:
    key_path = _isolate()
    try:
        signing.sign_skill_manifest(Path("/nope/SKILL.md"), key_path=key_path)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_signing_mode_default_is_local_dev() -> None:
    os.environ.pop("BERT_LAB_SIGNING_MODE", None)
    assert signing._signing_mode() == "local-dev"
    os.environ["BERT_LAB_SIGNING_MODE"] = "sigstore"
    try:
        assert signing._signing_mode() == "sigstore"
    finally:
        del os.environ["BERT_LAB_SIGNING_MODE"]


def main() -> int:
    tests = [
        test_keypair_generated_on_first_use,
        test_keypair_persists_across_calls,
        test_sign_bytes_roundtrip,
        test_verify_rejects_tampered_bytes,
        test_canonical_json_stable_across_key_order,
        test_sign_agent_card,
        test_sign_merkle_root,
        test_sign_skill_manifest,
        test_append_to_local_rekor_assigns_monotone_ids,
        test_read_local_rekor_filter_by_kind,
        test_sign_skill_missing_file_raises,
        test_signing_mode_default_is_local_dev,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
