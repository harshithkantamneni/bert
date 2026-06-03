"""Smoke test for the URGENT 2026-05-07 Cerebras Qwen3-235B → Qwen3-32B migration.

Verifies:
  1. core/provider.py default Cerebras model is now `qwen-3-32b` (NOT the
     deprecating-2026-05-27 `qwen-3-235b-a22b-instruct-2507`).
  2. core/subagent.py MODEL_FAMILIES still maps `cerebras` → `qwen` (family
     unchanged; the migration is within-family).
  3. core/subagent.py pick_evaluator_model returns a NON-Qwen-family
     evaluator when given a Cerebras (Qwen-family) producer — i.e.,
     P-VS-02 cross-family adversarial review still works post-migration.

Per FINAL_implementation_plan_2026-05-07.md §5.0 (Phase URGENT) acceptance.

Run: `python tests/_smoke_cerebras_qwen3_32b.py`
Exit 0 = pass; non-zero = fail.
"""

import sys
from pathlib import Path

# Ensure bert-lab repo root is on sys.path
LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import provider as provider_mod  # noqa: E402
from core import subagent as subagent_mod  # noqa: E402


def test_cerebras_default_is_llama31_8b() -> None:
    """Post-R13 live-API discovery (2026-05-07): R12's recommendation of
    qwen-3-32b was wrong — that model returns 404 on bert's free tier.
    Actual accessible models are qwen-3-235b (deprecating 2026-05-27)
    and llama3.1-8b. Migration target: llama3.1-8b."""
    cerebras_spec = provider_mod.PROVIDERS["cerebras"]
    assert cerebras_spec.default_model == "llama3.1-8b", (
        f"Expected Cerebras default 'llama3.1-8b' post-R13 fix; got "
        f"'{cerebras_spec.default_model}'."
    )


def test_cerebras_family_now_llama() -> None:
    """Cerebras family collapses to 'llama' post-deprecation since
    llama3.1-8b is the only accessible default. Loses Qwen-family slot
    via Cerebras; NVIDIA NIM has qwen/* models for explicit dispatch."""
    family = subagent_mod.MODEL_FAMILIES.get("cerebras")
    assert family == "llama", (
        f"Expected Cerebras family 'llama' post-R13 fix; got '{family}'."
    )


def test_pick_evaluator_returns_non_llama_family_for_cerebras_producer() -> None:
    """Cerebras (Llama family) producer → Evaluator from non-Llama family
    per P-VS-02 cross-family rule.

    Post-R14 (2026-05-07 quality-first redo): the cross-family slot
    registry routes the Qwen seat to NVIDIA's
    qwen/qwen3-next-80b-a3b-thinking (an 80B thinking-mode Qwen-family
    model live-verified on NVIDIA free tier), not Cerebras's 8B
    llama3.1-8b. So a Llama-family producer should get a Qwen-family
    judge first, not just any non-Llama provider.

    Family check uses slot_family_of (not provider-level family_of)
    because NVIDIA hosts both Llama family (default) and Qwen family
    (via explicit model) — the slot-level family is what matters for
    P-VS-02 cross-family.
    """
    evaluator_str = subagent_mod.pick_evaluator_model("cerebras/llama3.1-8b")
    assert "/" in evaluator_str
    # Split provider from model — model may itself contain '/' (e.g., "qwen/qwen3-next-...")
    eval_provider, _, eval_model = evaluator_str.partition("/")
    eval_family = subagent_mod.slot_family_of(eval_provider, eval_model)
    assert eval_family != "llama", (
        f"P-VS-02 violation: Cerebras (Llama) producer got '{evaluator_str}' "
        f"(slot-family='{eval_family}'). Family MUST differ."
    )
    # Quality-first assertion: the first slot is the Qwen-via-NVIDIA seat,
    # so a Llama producer should land there specifically.
    assert eval_family == "qwen", (
        f"R14 expected Qwen-family judge for Llama producer; got family='{eval_family}'"
    )


def main() -> int:
    tests = [
        test_cerebras_default_is_llama31_8b,
        test_cerebras_family_now_llama,
        test_pick_evaluator_returns_non_llama_family_for_cerebras_producer,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
