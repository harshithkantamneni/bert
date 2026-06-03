"""Falsifier #11 — cache-aware token reduction on contested pipeline.

Falsifier #11: ≥60% token reduction on contested-decision pipeline
(threshing → clearness phase 1 → phase 2 → cross-family judge).

Sample Inspect AI eval; deferred install per evals/README.md.
"""

from __future__ import annotations

EVAL_SPEC = {
    "name": "cache-token-reduction",
    "description": (
        "Falsifier #11: end-to-end token consumption on contested-"
        "decision pipeline shows ≥60% reduction vs cache-naive baseline "
        "(24K → 6.2K tokens = 75% reduction; "
        "60% is the floor target)."
    ),
    "scenario": (
        "Synthetic contested decision dispatched through threshing → "
        "clearness phase 1 → clearness phase 2 (cross-family judge fires "
        "for META altitude). Compare token-count under three configurations: "
        "(a) cache-naive baseline (no provider-cache observability), "
        "(b) provider-side caching only (Gemini implicit + Groq automatic), "
        "(c) full provider-cache observability (provider cache + LLMLingua compression on "
        "cross-family leg)."
    ),
    "falsifier": {
        "id": "cache-token-reduction-11",
        "description": "Total input tokens (c) ≤ 0.40 × (a) on contested pipeline",
        "target": "(a - c) / a >= 0.60",
        "data_source": (
            "sum of usage_prompt_tokens - usage_cached_tokens across "
            "the 4 dispatches in the pipeline, per ProviderResponse "
            "telemetry"
        ),
    },
    "live_API_required": True,
    "calibration_window": "5 contested decisions (dry-run scenarios)",
}


if __name__ == "__main__":
    import json
    print(json.dumps(EVAL_SPEC, indent=2))
