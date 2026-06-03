"""Falsifier #1+#2 — threshing pass behavior.

Falsifier #1: Threshing fires on contested-decision dispatches at ≥80%
              rate when dispatch_altitude in {META, SPEC}
Falsifier #2: Threshing produces verdict=SCOPE_STOP at 100% rate
              (non-SCOPE_STOP = bug; schema-rejected so ≤0% is the
              guaranteed bound)

This is a SAMPLE Inspect AI eval — NOT YET RUNNABLE because Inspect AI
isn't installed. Per evals/README.md: package installs are deferred to
the live-API calibration window. The migration pattern is documented
here.

When Inspect AI is installed and the live-API window opens:

  inspect eval evals/p_vs_06_threshing.py \
      --model nvidia/meta/llama-3.3-70b-instruct \
      --log-dir ./inspect_logs

Expected output: 14 baseline numbers, this eval contributing
2 of them.
"""

from __future__ import annotations

# When Inspect AI is installed, these imports become real:
#   from inspect_ai import Task, eval, task
#   from inspect_ai.dataset import Sample
#   from inspect_ai.scorer import scorer, Score
#   from inspect_ai.solver import generate

# For the scaffolding pass, we document the eval shape without running:

EVAL_SPEC = {
    "name": "p-vs-06-threshing",
    "description": (
        "Falsifier #1+#2: threshing fires on contested dispatches; "
        "always produces SCOPE_STOP."
    ),
    "model_pool": [
        # bert's free-tier reasoning models (May 2026 validated):
        "nvidia/deepseek-ai/deepseek-r1",       # primary
        "cerebras/qwen-3-32b",                   # post-URGENT migration
        "ollama/deepseek-r1-distill-qwen-32b",  # local fallback
    ],
    "falsifiers": [
        {
            "id": "FALS-9-1",
            "description": "Threshing fires on dispatch_altitude in {META, SPEC} contested decisions",
            "target": ">= 0.80",  # ≥80% fire rate
            "data_source": "core.observability.calibration_count('threshing_dispatch')",
        },
        {
            "id": "FALS-9-2",
            "description": "Threshing produces verdict=SCOPE_STOP",
            "target": "== 1.00",  # 100% (schema-enforced)
            "data_source": "core.observability.calibration_count('verdict', {'role': 'threshing_pass', 'verdict': 'SCOPE_STOP'})",
        },
    ],
    "calibration_window": "30 dispatches",
    "below_threshold_action": "P-001 three-strikes pivot",
}


# When Inspect AI is wired:
#
# @task
# def p_vs_06_threshing():
#     """Inspect AI Task for falsifiers #1+#2."""
#     return Task(
#         dataset=_synthetic_contested_decisions(),
#         solver=[
#             # bert's threshing dispatch invocation
#             generate(),
#         ],
#         scorer=_check_scope_stop_rate(),
#     )
#
# @scorer
# def _check_scope_stop_rate():
#     async def score(state, target):
#         from core.observability import calibration_count
#         total = calibration_count("threshing_dispatch")
#         scope_stop = calibration_count("verdict", {
#             "role": "threshing_pass", "verdict": "SCOPE_STOP",
#         })
#         rate = scope_stop / total if total else 0.0
#         return Score(value=rate, metadata={"total": total, "scope_stop": scope_stop})
#     return score


if __name__ == "__main__":
    # Print the eval spec for documentation/preview without running
    import json
    print(json.dumps(EVAL_SPEC, indent=2))
