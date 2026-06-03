"""Falsifier #3+#4 — clearness phase 1 query quality.

Falsifier #3: Clearness phase 1 produces ≥3 open queries (median 5)
Falsifier #4: Phase-1 query is_leading=false rate ≥95% (auto-J classifier)

Sample Inspect AI eval — same scaffolding caveat as p_vs_06_threshing.py.
"""

from __future__ import annotations

EVAL_SPEC = {
    "name": "p-vs-07-clearness-phase1",
    "description": (
        "Falsifier #3+#4: clearness phase-1 produces enough open queries; "
        "≥95% are non-leading per auto-J classifier."
    ),
    "model_pool": [
        "nvidia/meta/llama-3.3-70b-instruct",
        "cerebras/qwen-3-32b",
        "groq/llama-3.3-70b-versatile",
    ],
    "falsifiers": [
        {
            "id": "FALS-9-3",
            "description": "Phase-1 query count ≥3 (median 5)",
            "target": "median >= 5 AND min >= 3",
            "data_source": "len(packet.clearness_queries) for each phase-1 packet",
        },
        {
            "id": "FALS-9-4",
            "description": "Phase-1 is_leading=false rate ≥95%",
            "target": ">= 0.95",
            "data_source": "calibration_count('clearness_phase1_dispatch', {'is_leading': False}) / total",
            "note": (
                "Schema layer rejects is_leading=true; this falsifier "
                "checks that the LLM-as-judge auto-J classifier agrees "
                "with the schema enforcement on borderline cases (e.g., "
                "questions that pass schema but feel leading on read)."
            ),
        },
    ],
}


if __name__ == "__main__":
    import json
    print(json.dumps(EVAL_SPEC, indent=2))
