"""Smoke test for I.3: failures.md separately-signed + adversarial-eval-by-design."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import adversarial_eval, proof_packet, signing


def test_adversarial_module_loads() -> None:
    assert hasattr(adversarial_eval, "run_adversarial_eval")
    assert hasattr(adversarial_eval, "ATTACK_TEMPLATES")
    assert set(adversarial_eval.ATTACK_TEMPLATES.keys()) == {
        "falsifier_probe", "distribution_shift",
        "sample_size", "counterexample",
    }


def test_run_adversarial_eval_shape() -> None:
    claims = [{"id": "C-1", "text": "test claim", "confidence_1to10": 9,
               "limitationRefs": []}]
    result = adversarial_eval.run_adversarial_eval(claims)
    assert result["method"] == "heuristic-v1"
    assert result["total_attempts"] == 4  # 4 attack types × 1 claim
    assert len(result["attempts"]) == 4
    for attempt in result["attempts"]:
        assert "attack_id" in attempt
        assert attempt["claim_id"] == "C-1"
        assert attempt["verdict"] in (
            "claim_defended", "claim_weakened", "claim_falsified"
        )


def test_adversarial_eval_no_claims() -> None:
    """Empty claims list → zero attempts."""
    result = adversarial_eval.run_adversarial_eval([])
    assert result["total_attempts"] == 0
    assert result["attempts"] == []


def test_adversarial_eval_defends_claims_with_limitations() -> None:
    """Claims with limitationRefs → defended on at least some attacks."""
    claims_no_lim = [{"id": "C-1", "text": "x", "confidence_1to10": 9,
                       "limitationRefs": []}]
    claims_with_lim = [{"id": "C-1", "text": "x", "confidence_1to10": 5,
                        "limitationRefs": ["L-1"]}]
    r_no = adversarial_eval.run_adversarial_eval(claims_no_lim)
    r_with = adversarial_eval.run_adversarial_eval(claims_with_lim)
    defended_no = r_no["summary"]["by_verdict"].get("claim_defended", 0)
    defended_with = r_with["summary"]["by_verdict"].get("claim_defended", 0)
    assert defended_with > defended_no, (
        f"limitation-flagged claims should defend more attacks; "
        f"got no-lim={defended_no} with-lim={defended_with}"
    )


def test_extract_claims_from_verdict_events() -> None:
    events = [
        {"event_class": "verdict", "agent": "researcher", "cycle": 1,
         "verdict": "APPROVE", "confidence_1to10": 8},
        {"event_class": "verdict", "agent": "implementer", "cycle": 1,
         "verdict": "APPROVE_WITH_CAVEATS", "confidence_1to10": 5},
        {"event_class": "model_call", "cycle": 1},  # skipped
    ]
    claims = proof_packet.extract_claims_from_events(events)
    assert len(claims) == 2
    assert claims[0]["id"] == "C-1"
    assert claims[0]["role"] == "researcher"
    assert claims[1]["id"] == "C-2"
    assert claims[1]["role"] == "implementer"


def test_extract_limitations_from_low_confidence() -> None:
    """Low-confidence verdict → conservative-judgement limitation."""
    events = [
        {"event_class": "verdict", "agent": "researcher", "cycle": 1,
         "verdict": "APPROVE", "confidence_1to10": 4},
    ]
    claims = proof_packet.extract_claims_from_events(events)
    lims = proof_packet.extract_limitations_from_events(events, claims, cycle_id=1)
    assert len(lims) == 1
    assert lims[0]["category"] == "conservative-judgement"


def test_failures_md_renders_table_when_limitations() -> None:
    claims = [{"id": "C-1", "text": "x", "limitationRefs": ["L-1"]}]
    lims = [{"id": "L-1", "category": "distribution-shift",
              "text": "test limitation", "claimRefs": ["C-1"]}]
    md = proof_packet.render_failures_md(42, claims, lims)
    assert "Failures — cycle-0042" in md
    assert "L-1" in md
    assert "distribution-shift" in md
    assert "POPPER-style" in md


def test_failures_md_yellow_flag_when_empty() -> None:
    """Empty limitations → 'PASS-WITH-WARNINGS' notice in failures.md."""
    md = proof_packet.render_failures_md(42, [], [])
    assert "No structured limitations" in md
    assert "PASS-WITH-WARNINGS" in md or "yellow flag" in md


def test_packet_contains_failures_sigstore() -> None:
    """The packet must include both failures.md and failures.sigstore."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i3_"))
    try:
        path = proof_packet.build_packet(cycle_id=400, output_dir=tmp)
        with tarfile.open(path, "r:gz") as tar:
            names = tar.getnames()
        prefix = "cycle-0400/"
        assert prefix + "failures.md" in names
        assert prefix + "failures.sigstore" in names
        assert prefix + "eval/adversarial.json" in names
    finally:
        shutil.rmtree(tmp)


def test_failures_signature_is_independently_verifiable() -> None:
    """failures.sigstore must verify against failures.md without needing
    any other packet artifact. CTO-friend test for failures alone."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i3_"))
    try:
        path = proof_packet.build_packet(cycle_id=400, output_dir=tmp)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        pdir = tmp / "cycle-0400"
        failures_bytes = (pdir / "failures.md").read_bytes()
        # failures.sigstore is now a cosign-compatible ECDSA-P256 envelope
        # (item 25) — independently verifiable, and via vanilla cosign too.
        env = json.loads((pdir / "failures.sigstore").read_text())
        assert env["algo"] == "ecdsa-p256-sha256"
        assert signing.verify_cosign_blob(
            failures_bytes, env["signature_b64"], env["pubkey_pem"]
        ), "failures.sigstore must verify against failures.md alone"
        cosign = shutil.which("cosign")
        if cosign:
            r = subprocess.run(
                [cosign, "verify-blob", "--key", str(pdir / "cosign.pub"),
                 "--signature", str(pdir / "failures.sig"),
                 "--insecure-ignore-tlog=true", str(pdir / "failures.md")],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, (
                f"vanilla cosign rejected failures.md: {r.stderr.strip()}"
            )
    finally:
        shutil.rmtree(tmp)


def test_adversarial_json_in_packet_is_real_not_placeholder() -> None:
    """eval/adversarial.json must have real attempts, not the I.2 placeholder."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i3_"))
    try:
        path = proof_packet.build_packet(cycle_id=400, output_dir=tmp)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        adv_path = tmp / "cycle-0400" / "eval" / "adversarial.json"
        adv = json.loads(adv_path.read_text())
        assert adv.get("method") == "heuristic-v1"
        assert adv.get("total_attempts", 0) > 0
        assert "attempts" in adv
        assert len(adv["attempts"]) == adv["total_attempts"]
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    tests = [
        test_adversarial_module_loads,
        test_run_adversarial_eval_shape,
        test_adversarial_eval_no_claims,
        test_adversarial_eval_defends_claims_with_limitations,
        test_extract_claims_from_verdict_events,
        test_extract_limitations_from_low_confidence,
        test_failures_md_renders_table_when_limitations,
        test_failures_md_yellow_flag_when_empty,
        test_packet_contains_failures_sigstore,
        test_failures_signature_is_independently_verifiable,
        test_adversarial_json_in_packet_is_real_not_placeholder,
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
