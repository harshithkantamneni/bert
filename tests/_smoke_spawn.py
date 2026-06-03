"""LIVE-TEST — requires real model provider credentials. Smoke test for Spawn → Researcher dispatch.

Not a pytest — a manual end-to-end runner. Call as:
    uv run python tests/_smoke_spawn.py

Sends a deliberately tiny task to the Researcher so it doesn't crawl the web,
just exercises the dispatch path: Spawn → validate spec → run sub-agent loop →
validate ResultPacket → return summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402

SMOKE_SPEC = {
    "dispatch_altitude": "INFRA",
    "role": "researcher",
    "cycle": 99,  # 99 = smoke / non-canonical
    "task": (
        "SMOKE TEST — do NOT actually crawl any source. "
        "Write the file at output_path with a single fake signal entry "
        "(\"placeholder smoke signal\", URL: https://example.com/smoke). "
        "Then write the ResultPacket with verdict='APPROVE' (NOT "
        "'APPROVE_WITH_CAVEATS' — you have no concerns to embed for a smoke "
        "test). The calibration_reasoning field MUST be at least 80 characters "
        "long (schema requirement) — describe what was tested and why "
        "confidence is what it is in 2 sentences. Then stop. This is a "
        "dispatch-pipeline test, not real research."
    ),
    "success_criterion": (
        "Both findings/test_smoke.md and the ResultPacket JSON exist and validate."
    ),
    "output_path": "findings/test_smoke.md",
    # NVIDIA llama-3.3-70b is bert's production tool-call workhorse; the small
    # cerebras/llama3.1-8b doesn't reliably emit tool_use in one iteration so
    # the smoke flakes there. Per R13 (2026-05-07) the Cerebras default is now
    # llama3.1-8b (qwen-3-32b / 235b / glm-4.7 / gpt-oss-120b all 404 or
    # deprecating) but for this end-to-end smoke we want the bigger model.
    "model": "nvidia/meta/llama-3.3-70b-instruct",
    "process_hygiene": (
        "Smoke test: no real web scraping; minimal output; compliant ResultPacket."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if either output file is missing or fails schema validation."
    ),
}


def main() -> int:
    print("=" * 60)
    print("Smoke test: Spawn → Researcher")
    print("=" * 60)

    summary = subagent.run_subagent(SMOKE_SPEC)
    print()
    print("--- Summary returned to Director ---")
    print(json.dumps(summary, indent=2, default=str))
    print()
    print("--- Spec valid:", summary["spec_valid"])
    print("--- Result valid:", summary["result_valid"])
    print("--- Verdict:", summary["verdict"])
    print("--- Errors:", summary["errors"])
    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
