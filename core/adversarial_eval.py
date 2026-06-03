"""Adversarial-eval-by-design (I.3 — strongest authenticity signal).

Per the May-2026 research, almost no agent product ships adversarial
eval by default. Most labs publish self-eval ("we graded ourselves
with GPT-4o-as-judge") which reads to investors like "we audited
ourselves." This module makes adversarial eval STRUCTURAL — every
proof packet includes an `eval/adversarial.json` with the red-team
agent's attack attempts and outcomes.

The red-team agent generates attacks against each declared claim:
  - "What evidence would FALSIFY claim C-N?"
  - "Under what distribution shift does claim C-N break?"
  - "What's the smallest counterexample that defeats C-N?"
  - "Is claim C-N's conclusion-confidence justified by its sample size?"

Two attack-generation modes:

  mode="heuristic"  → static rules over claim metadata
    (`limitationRefs`, `confidence_1to10`). No live LLM calls.
    Framework-reliable: always exports a packet.
    `method = "heuristic-v1"`.

  mode="llm"        → cross-family LLM-driven attacks.
    Per DD.1 (Gap #3 honest-disclosure): we route each attack to
    a model from a DIFFERENT FAMILY than the producer (Mistral
    attacks a Llama producer, Llama attacks a Mistral producer,
    Qwen attacks both, etc.) via `core.subagent.pick_evaluator_model`.
    This is what bert v2 ships in proof packets. Each LLM attempt
    that fails to parse a verdict falls back to heuristic for that
    one attempt — never lets the packet export break.
    `method = "llm-driven-v2"` (or `"hybrid-v2"` if any attempt
    fell back).

Attack record format:
{
    "attack_id": "atk-001",
    "claim_id": "C-1",
    "attack_type": "falsifier_probe" | "distribution_shift" | "sample_size" | "counterexample",
    "prompt": "What evidence would falsify C-1?",
    "response": "<adversary's response>",
    "verdict": "claim_defended" | "claim_weakened" | "claim_falsified",
    "rationale": "...",
    "attacker_model": "provider/model" (only in llm mode),
    "attacker_family": "mistral|llama|qwen|gemma|..." (only in llm mode),
    "fallback_reason": str (only present when LLM attempt fell back to heuristic)
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

LOG = logging.getLogger(__name__)

# Heuristic attack templates per attack_type. v2 swaps these for
# LLM-driven attacks.
ATTACK_TEMPLATES = {
    "falsifier_probe":
        "What concrete evidence, if observed, would falsify claim {claim_id}: '{claim_text}'? "
        "Name the observation that would force a retraction.",
    "distribution_shift":
        "Under what distribution shift (input regime, time horizon, "
        "scale, adversarial input) does claim {claim_id} break first?",
    "sample_size":
        "Is the conclusion-confidence in claim {claim_id} justified by the "
        "sample size? What's the minimum N needed for this to hold?",
    "counterexample":
        "Construct the smallest plausible counterexample to claim {claim_id}.",
}


def _heuristic_response(claim: dict, attack_type: str) -> dict:
    """Generate a heuristic adversarial response. Honest about being
    heuristic — never pretends to be a live model judgment.

    Returns dict with `response`, `verdict`, `rationale`.
    """
    text = claim.get("text", "")[:200]
    confidence = claim.get("confidence_1to10")
    has_limitations = bool(claim.get("limitationRefs"))

    if attack_type == "falsifier_probe":
        if has_limitations:
            return {
                "response": f"Claim {claim['id']} carries declared limitations "
                            f"{claim.get('limitationRefs')}. The cycle has named "
                            f"what could falsify it.",
                "verdict": "claim_defended",
                "rationale": "self-flagged limitations; honest about failure modes",
            }
        return {
            "response": f"Claim {claim['id']} declares no limitation "
                        f"cross-references. A falsifier could exist that the "
                        f"cycle has not enumerated.",
            "verdict": "claim_weakened",
            "rationale": "unflagged claim → adversary cannot rule out hidden failure modes",
        }
    if attack_type == "distribution_shift":
        return {
            "response": f"Claim {claim['id']} (text: {text!r}) was emitted in the "
                        f"cycle's specific dispatch context. Distribution shift "
                        f"to a new role, model family, or input regime is not "
                        f"covered by this packet alone.",
            "verdict": "claim_defended" if has_limitations else "claim_weakened",
            "rationale": "single-cycle packets don't generalize unless cross-cycle lineage is present",
        }
    if attack_type == "sample_size":
        if confidence is None:
            return {
                "response": f"Claim {claim['id']} has no declared confidence_1to10.",
                "verdict": "claim_weakened",
                "rationale": "no confidence → cannot defend sample-size adequacy",
            }
        if confidence >= 8:
            return {
                "response": f"Claim {claim['id']} declares confidence {confidence}/10. "
                            f"A high confidence on a single-cycle observation invites "
                            f"the standard adversary objection: large N requirement.",
                "verdict": "claim_weakened",
                "rationale": "high confidence + small sample is the classic over-confidence pattern",
            }
        return {
            "response": f"Claim {claim['id']} declares confidence {confidence}/10, "
                        f"appropriately conservative for a single-cycle observation.",
            "verdict": "claim_defended",
            "rationale": "conservative confidence is consistent with a single-cycle sample",
        }
    if attack_type == "counterexample":
        return {
            "response": f"A counterexample to {claim['id']} would require an input "
                        f"the cycle did NOT process. The packet does not enumerate "
                        f"such inputs. Specific construction depends on claim semantics.",
            "verdict": "claim_weakened" if not has_limitations else "claim_defended",
            "rationale": "counterexample-construction needs claim-specific knowledge the packet omits",
        }
    return {
        "response": f"Unknown attack type {attack_type!r}",
        "verdict": "claim_defended",
        "rationale": "unsupported attack",
    }


VALID_VERDICTS = ("claim_defended", "claim_weakened", "claim_falsified")


_LLM_ATTACK_SYSTEM = (
    "You are a red-team adversarial reviewer. Your job is to attack a claim "
    "made by another AI system, from a DIFFERENT model family than the producer "
    "— cross-family review surfaces failure modes that same-family review hides "
    "(P-VS-02 cross-family discipline). Do not concede; probe for weakness, "
    "missing falsifiers, under-supported confidence, and distribution shifts. "
    "Return ONLY a JSON object with keys `verdict`, `rationale`, `response`. "
    "Allowed verdicts: claim_defended, claim_weakened, claim_falsified."
)


def _llm_attack_user_prompt(claim: dict, attack_type: str) -> str:
    """Build the per-attack user prompt for the LLM adversary."""
    claim_text = (claim.get("text", "") or "")[:600]
    template = ATTACK_TEMPLATES.get(attack_type, "")
    prompt = template.format(claim_id=claim["id"], claim_text=claim_text)
    meta_lines = [
        f"Claim id: {claim['id']}",
        f"Claim text: {claim_text}",
    ]
    if claim.get("confidence_1to10") is not None:
        meta_lines.append(f"Producer's declared confidence: {claim['confidence_1to10']}/10")
    if claim.get("limitationRefs"):
        meta_lines.append(f"Producer's declared limitations: {claim['limitationRefs']}")
    return (
        f"# Adversarial probe — {attack_type}\n\n"
        + "\n".join(meta_lines)
        + f"\n\n## Your attack question\n{prompt}\n\n"
        "## Output\nReturn JSON exactly of this shape (no prose outside it):\n"
        '{\n  "verdict": "claim_defended | claim_weakened | claim_falsified",\n'
        '  "response": "≤500 char specific adversarial response to the probe",\n'
        '  "rationale": "≤300 char justification for the verdict"\n}\n'
    )


def _parse_llm_attack_json(raw: str) -> dict | None:
    """Extract the first JSON object from the model's response and validate
    it has verdict/response/rationale with allowed verdict values. Uses
    json.loads only — never any code-execution path."""
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    verdict = obj.get("verdict")
    response = obj.get("response")
    rationale = obj.get("rationale")
    if verdict not in VALID_VERDICTS:
        return None
    if not isinstance(response, str) or not response.strip():
        return None
    if not isinstance(rationale, str) or not rationale.strip():
        return None
    return {
        "verdict": verdict,
        "response": response.strip()[:600],
        "rationale": rationale.strip()[:400],
    }


def _llm_response(claim: dict, attack_type: str, *,
                  producer_model: str | None) -> dict:
    """Cross-family LLM-driven adversarial attack against `claim`.

    Picks an attacker model from a family that differs from `producer_model`'s
    family (via core.subagent.pick_evaluator_model). Calls
    core.provider.call() with a strict JSON-output prompt. Returns the
    parsed verdict + rationale + telemetry.

    Failure modes (each falls back to heuristic for THIS attempt only,
    never crashes the full eval):
      - no producer_model declared → "missing_producer_model"
      - cross-family selection raises → "evaluator_selection_failed"
      - provider call returns text "[bert] unknown provider..." → "provider_call_failed"
      - response JSON unparseable / missing fields → "unparseable_response"
    """
    # Lazy imports to avoid circular dep: adversarial_eval → subagent →
    # agent → tools → adversarial_eval.
    from core import provider as _prov
    from core import subagent as _sub

    if not producer_model:
        return {**_heuristic_response(claim, attack_type),
                "fallback_reason": "missing_producer_model",
                "attacker_model": None, "attacker_family": None}

    try:
        attacker_pm = _sub.pick_evaluator_model(producer_model)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("adversarial: evaluator selection failed: %s", exc)
        return {**_heuristic_response(claim, attack_type),
                "fallback_reason": f"evaluator_selection_failed:{exc.__class__.__name__}",
                "attacker_model": None, "attacker_family": None}

    if "/" in attacker_pm:
        attacker_provider, attacker_model = attacker_pm.split("/", 1)
    else:
        attacker_provider, attacker_model = attacker_pm, None
    attacker_family = _sub.slot_family_of(attacker_provider, attacker_model)

    messages = [
        {"role": "system", "content": _LLM_ATTACK_SYSTEM},
        {"role": "user", "content": _llm_attack_user_prompt(claim, attack_type)},
    ]
    try:
        resp = _prov.call(
            attacker_provider, messages,
            model=attacker_model, max_tokens=600, temperature=0.4,
            response_format={"type": "json_object"}, timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("adversarial: provider call raised: %s", exc)
        return {**_heuristic_response(claim, attack_type),
                "fallback_reason": f"provider_exception:{exc.__class__.__name__}",
                "attacker_model": attacker_pm, "attacker_family": attacker_family}

    if resp.finish_reason == "error" or resp.text.startswith("[bert]"):
        return {**_heuristic_response(claim, attack_type),
                "fallback_reason": "provider_call_failed",
                "attacker_model": attacker_pm, "attacker_family": attacker_family,
                "provider_error_text": resp.text[:200]}

    parsed = _parse_llm_attack_json(resp.text)
    if parsed is None:
        return {**_heuristic_response(claim, attack_type),
                "fallback_reason": "unparseable_response",
                "attacker_model": attacker_pm, "attacker_family": attacker_family,
                "raw_response_excerpt": resp.text[:200]}
    return {
        "response": parsed["response"],
        "verdict": parsed["verdict"],
        "rationale": parsed["rationale"],
        "attacker_model": attacker_pm,
        "attacker_family": attacker_family,
    }


def run_adversarial_eval(
    claims: list[dict],
    *,
    attack_types: tuple[str, ...] | None = None,
    seed: str | None = None,
    mode: str = "heuristic",
    producer_model: str | None = None,
) -> dict:
    """Run the red-team agent against a list of claims.

    Args:
      claims: list of claim dicts. Each must have at minimum {id, text}.
        Optional: confidence_1to10, limitationRefs.
      attack_types: which attacks to run per claim. Defaults to all 4.
      seed: deterministic seed for attack_id generation. If None,
        attack ids are derived from claim+attack content hash.
      mode: "heuristic" (no LLM calls; framework-reliable) or "llm"
        (cross-family LLM-driven attacks; per-attempt fallback to
        heuristic on parse / call failures).
      producer_model: "provider/model" of the agent that PRODUCED the
        claims, needed for cross-family attacker selection. Required
        when mode="llm"; ignored in heuristic mode.

    Returns:
      dict with `attempts` (list of attack records), `summary` (counts
      by verdict + attack_type + fallback if mode="llm"), `method`
      ("heuristic-v1" or "llm-driven-v2" or "hybrid-v2" when some LLM
      attempts fell back), `mode`, `producer_model`.
    """
    if mode not in ("heuristic", "llm"):
        raise ValueError(f"unknown mode {mode!r}; expected 'heuristic' or 'llm'")
    attack_types = attack_types or tuple(ATTACK_TEMPLATES.keys())
    attempts: list[dict] = []
    summary_by_verdict: dict[str, int] = {}
    summary_by_type: dict[str, int] = {}
    fallback_count = 0
    llm_call_count = 0
    for claim in claims:
        if not isinstance(claim, dict) or not claim.get("id"):
            continue
        for atype in attack_types:
            template = ATTACK_TEMPLATES.get(atype)
            if not template:
                continue
            prompt = template.format(
                claim_id=claim["id"],
                claim_text=(claim.get("text", "") or "")[:200],
            )
            if mode == "llm":
                llm_call_count += 1
                response_data = _llm_response(
                    claim, atype, producer_model=producer_model)
                if response_data.get("fallback_reason"):
                    fallback_count += 1
            else:
                response_data = _heuristic_response(claim, atype)
            seed_str = seed or f"{claim['id']}|{atype}"
            attack_id = "atk-" + hashlib.sha256(seed_str.encode()).hexdigest()[:8]
            attempt = {
                "attack_id": attack_id,
                "claim_id": claim["id"],
                "attack_type": atype,
                "prompt": prompt,
                "response": response_data["response"],
                "verdict": response_data["verdict"],
                "rationale": response_data["rationale"],
            }
            for opt in ("attacker_model", "attacker_family",
                        "fallback_reason", "provider_error_text",
                        "raw_response_excerpt"):
                if opt in response_data and response_data[opt] is not None:
                    attempt[opt] = response_data[opt]
            attempts.append(attempt)
            v = response_data["verdict"]
            summary_by_verdict[v] = summary_by_verdict.get(v, 0) + 1
            summary_by_type[atype] = summary_by_type.get(atype, 0) + 1
    if mode == "heuristic":
        method = "heuristic-v1"
        method_note = (
            "v1 ships heuristic attacks for framework reliability. "
            "v2 = LLM-driven attacks via cross-family routing — toggle via "
            "mode='llm' or BERT_ADVERSARIAL_MODE=llm env var."
        )
    elif fallback_count == 0 and llm_call_count > 0:
        method = "llm-driven-v2"
        method_note = (
            "Every attack was generated by an LLM from a family different "
            "from the producer's (cross-family discipline P-VS-02). No "
            "attempts fell back to heuristic."
        )
    elif fallback_count == llm_call_count:
        method = "hybrid-v2"
        method_note = (
            f"All {llm_call_count} LLM attempts fell back to heuristic. "
            "Check provider availability — every attempt's `fallback_reason` "
            "field explains why. The packet still exports cleanly."
        )
    else:
        method = "hybrid-v2"
        method_note = (
            f"{llm_call_count - fallback_count} LLM-driven + {fallback_count} "
            f"heuristic fallback (of {llm_call_count} attempts). Each "
            "fallback's `fallback_reason` field explains the cause."
        )
    return {
        "method": method,
        "_method_note": method_note,
        "mode": mode,
        "producer_model": producer_model if mode == "llm" else None,
        "total_attempts": len(attempts),
        "llm_call_count": llm_call_count,
        "fallback_count": fallback_count,
        "summary": {
            "by_verdict": summary_by_verdict,
            "by_attack_type": summary_by_type,
        },
        "attempts": attempts,
    }


__all__ = ["run_adversarial_eval", "ATTACK_TEMPLATES", "VALID_VERDICTS"]
