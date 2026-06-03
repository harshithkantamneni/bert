"""Smoke test for DD.1 — LLM-driven adversarial role.

This test never invokes Python's builtin code-execution functions.
All parsing is via json.loads. All cross-module interaction is via
unittest.mock.patch.

Covers:
  - heuristic-v1 mode unchanged (existing I.3 callers don't break)
  - llm mode with no producer_model → all attempts fall back, method="hybrid-v2"
  - _parse_llm_attack_json accepts well-formed, rejects bad shape/verdict
  - _llm_response monkeypatched to simulate provider call → method="llm-driven-v2"
  - _llm_response monkeypatched to fail → fallback_reason set, method="hybrid-v2"
  - proof_packet honors BERT_ADVERSARIAL_MODE env var
  - cross-family attacker family is recorded
  - producer_model derivation from cycle_json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import adversarial_eval as ae  # noqa: E402

SAMPLE_CLAIM = {
    "id": "C-1",
    "text": "bert routes across 8 free-tier providers with cross-family discipline",
    "confidence_1to10": 7,
    "limitationRefs": ["L-22"],
}


def test_module_exports() -> None:
    for name in ("run_adversarial_eval", "ATTACK_TEMPLATES",
                 "VALID_VERDICTS", "_llm_response",
                 "_parse_llm_attack_json", "_LLM_ATTACK_SYSTEM"):
        assert hasattr(ae, name), f"missing export {name!r}"


def test_valid_verdicts_locked() -> None:
    assert set(ae.VALID_VERDICTS) == {
        "claim_defended", "claim_weakened", "claim_falsified"}


def test_heuristic_mode_unchanged() -> None:
    """Existing I.3 callers (no mode= arg) must still work as before."""
    r = ae.run_adversarial_eval([SAMPLE_CLAIM])
    assert r["method"] == "heuristic-v1"
    assert r["mode"] == "heuristic"
    assert r["producer_model"] is None
    assert r["total_attempts"] == 4
    assert r["llm_call_count"] == 0
    assert r["fallback_count"] == 0
    assert all("attacker_model" not in a for a in r["attempts"])


def test_invalid_mode_raises() -> None:
    try:
        ae.run_adversarial_eval([SAMPLE_CLAIM], mode="bogus")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "bogus" in str(e)


def test_parse_well_formed_attack_response() -> None:
    raw = json.dumps({
        "verdict": "claim_weakened",
        "response": "Producer did not enumerate any falsifier for the routing claim.",
        "rationale": "missing falsifier",
    })
    out = ae._parse_llm_attack_json(raw)
    assert out is not None
    assert out["verdict"] == "claim_weakened"
    assert "falsifier" in out["response"]


def test_parse_strips_code_fences() -> None:
    raw = '```json\n' + json.dumps({
        "verdict": "claim_defended", "response": "ok", "rationale": "ok",
    }) + '\n```'
    out = ae._parse_llm_attack_json(raw)
    assert out is not None
    assert out["verdict"] == "claim_defended"


def test_parse_rejects_invalid_verdict() -> None:
    raw = json.dumps({"verdict": "lgtm", "response": "x", "rationale": "x"})
    assert ae._parse_llm_attack_json(raw) is None


def test_parse_rejects_missing_field() -> None:
    raw = json.dumps({"verdict": "claim_defended", "response": "x"})
    assert ae._parse_llm_attack_json(raw) is None


def test_parse_rejects_empty_strings() -> None:
    raw = json.dumps({"verdict": "claim_defended", "response": "", "rationale": "x"})
    assert ae._parse_llm_attack_json(raw) is None


def test_parse_handles_garbage() -> None:
    assert ae._parse_llm_attack_json("not json at all") is None
    assert ae._parse_llm_attack_json("") is None
    assert ae._parse_llm_attack_json("   ") is None


def test_llm_mode_no_producer_model_all_fallback() -> None:
    r = ae.run_adversarial_eval([SAMPLE_CLAIM], mode="llm")
    assert r["mode"] == "llm"
    assert r["llm_call_count"] == 4
    assert r["fallback_count"] == 4
    assert r["method"] == "hybrid-v2"
    for a in r["attempts"]:
        assert a.get("fallback_reason") == "missing_producer_model"


def test_llm_mode_all_succeed_method_is_llm_driven_v2() -> None:
    def fake_llm(claim, atype, *, producer_model):
        return {
            "response": f"adversarial probe for {claim['id']} via {atype}",
            "verdict": "claim_weakened",
            "rationale": "the producer did not declare X",
            "attacker_model": "mistral/mistral-small-latest",
            "attacker_family": "mistral",
        }
    with patch.object(ae, "_llm_response", side_effect=fake_llm):
        r = ae.run_adversarial_eval(
            [SAMPLE_CLAIM], mode="llm",
            producer_model="nvidia/meta/llama-3.3-70b-instruct",
        )
    assert r["method"] == "llm-driven-v2"
    assert r["fallback_count"] == 0
    assert r["llm_call_count"] == 4
    assert r["mode"] == "llm"
    assert r["producer_model"] == "nvidia/meta/llama-3.3-70b-instruct"
    for a in r["attempts"]:
        assert a["attacker_family"] == "mistral"
        assert "fallback_reason" not in a


def test_llm_mode_all_fallback_method_is_hybrid_v2() -> None:
    def fake_llm(claim, atype, *, producer_model):
        return {
            **ae._heuristic_response(claim, atype),
            "fallback_reason": "provider_call_failed",
            "attacker_model": "mistral/mistral-small-latest",
            "attacker_family": "mistral",
        }
    with patch.object(ae, "_llm_response", side_effect=fake_llm):
        r = ae.run_adversarial_eval(
            [SAMPLE_CLAIM], mode="llm",
            producer_model="nvidia/meta/llama-3.3-70b-instruct",
        )
    assert r["method"] == "hybrid-v2"
    assert r["fallback_count"] == 4
    assert r["llm_call_count"] == 4
    for a in r["attempts"]:
        assert a["fallback_reason"] == "provider_call_failed"


def test_llm_mode_mixed_fallback() -> None:
    counter = {"n": 0}

    def fake_llm(claim, atype, *, producer_model):
        counter["n"] += 1
        if counter["n"] % 2 == 0:
            return {
                **ae._heuristic_response(claim, atype),
                "fallback_reason": "unparseable_response",
                "attacker_model": "mistral/mistral-small-latest",
                "attacker_family": "mistral",
            }
        return {
            "response": "live attack",
            "verdict": "claim_falsified",
            "rationale": "we found a counterexample",
            "attacker_model": "mistral/mistral-small-latest",
            "attacker_family": "mistral",
        }
    with patch.object(ae, "_llm_response", side_effect=fake_llm):
        r = ae.run_adversarial_eval(
            [SAMPLE_CLAIM], mode="llm",
            producer_model="nvidia/meta/llama-3.3-70b-instruct",
        )
    assert r["method"] == "hybrid-v2"
    assert r["fallback_count"] == 2
    assert r["llm_call_count"] == 4


def test_llm_attack_user_prompt_includes_claim_metadata() -> None:
    prompt = ae._llm_attack_user_prompt(SAMPLE_CLAIM, "falsifier_probe")
    assert "C-1" in prompt
    assert "8 free-tier providers" in prompt
    assert "confidence: 7/10" in prompt
    assert "L-22" in prompt
    assert "claim_defended" in prompt


def test_method_note_documents_env_toggle() -> None:
    r = ae.run_adversarial_eval([SAMPLE_CLAIM])
    assert "BERT_ADVERSARIAL_MODE" in r["_method_note"]


def test_proof_packet_default_mode_is_heuristic() -> None:
    os.environ.pop("BERT_ADVERSARIAL_MODE", None)
    src = (LAB_ROOT / "core" / "proof_packet.py").read_text()
    assert 'os.environ.get("BERT_ADVERSARIAL_MODE"' in src
    assert '"heuristic")' in src


def test_proof_packet_passes_producer_model_to_eval() -> None:
    src = (LAB_ROOT / "core" / "proof_packet.py").read_text()
    assert 'producer_provider = cycle_json.get("provider")' in src
    assert 'producer_model_str = cycle_json.get("model")' in src
    assert "f\"{producer_provider}/{producer_model_str}\"" in src
    assert "producer_model=producer_model" in src


def test_attempts_have_required_fields() -> None:
    r = ae.run_adversarial_eval([SAMPLE_CLAIM])
    required = {"attack_id", "claim_id", "attack_type",
                "prompt", "response", "verdict", "rationale"}
    for a in r["attempts"]:
        missing = required - set(a.keys())
        assert not missing, f"attempt missing {missing}"


def main() -> int:
    tests = [
        test_module_exports,
        test_valid_verdicts_locked,
        test_heuristic_mode_unchanged,
        test_invalid_mode_raises,
        test_parse_well_formed_attack_response,
        test_parse_strips_code_fences,
        test_parse_rejects_invalid_verdict,
        test_parse_rejects_missing_field,
        test_parse_rejects_empty_strings,
        test_parse_handles_garbage,
        test_llm_mode_no_producer_model_all_fallback,
        test_llm_mode_all_succeed_method_is_llm_driven_v2,
        test_llm_mode_all_fallback_method_is_hybrid_v2,
        test_llm_mode_mixed_fallback,
        test_llm_attack_user_prompt_includes_claim_metadata,
        test_method_note_documents_env_toggle,
        test_proof_packet_default_mode_is_heuristic,
        test_proof_packet_passes_producer_model_to_eval,
        test_attempts_have_required_fields,
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
