"""Direct Strategist dispatch — validates the new prompts/strategist.md.

Runs a Strategist over the existing C1 + C2 + C3 Researcher findings in
findings/ and agents/researcher/, asks for a candidate matrix. Validates:
- prompts/strategist.md loads cleanly via _load_system_prompt
- The Strategist follows the 6-step methodology
- Pre-registers falsifiers per pi_notes mandate
- Returns valid ResultPacket

Run as:
    PYTHONPATH=. uv run python tests/_real_strategist_dispatch.py
"""
from __future__ import annotations

import json

from core import subagent

DISPATCH = {
    "dispatch_altitude": "META",
    "role": "strategist",
    "cycle": 4,
    "task": (
        "STRATEGIST SYNTHESIS — Phase 0 cycle 4. Read all current Researcher "
        "findings:\n"
        "- findings/researcher_technical_C1.md (technical lens, 5 signals)\n"
        "- findings/researcher_trend_C2.md (trend velocity lens, 5 signals)\n"
        "- findings/technical_landscape_C2_researcher.md (technical lens, 12 inflection points)\n"
        "- agents/researcher/output_cycle3.md (technical lens, 7 inflection points)\n\n"
        "Cluster the signals across the technical and trend-velocity lenses, "
        "generate 4-7 candidate product targets aligned with bert-lab's "
        "Phase 0 free-tier constraints, evaluate each against the 6-dim "
        "constraint matrix in prompts/strategist.md, pre-register 2-3 "
        "falsifiers per surviving candidate, and rank. Top 3 should each "
        "get a full evaluation paragraph."
    ),
    "success_criterion": (
        "agents/strategist/output_cycle4.md exists with ≥4 candidates, each "
        "scored on the 6 dimensions, with falsifiers pre-registered, and "
        "ResultPacket validates."
    ),
    "output_path": "agents/strategist/output_cycle4.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "Cross-check every candidate against memories/killed.md before "
        "ranking. Cite real URLs from the source findings. Mark single-"
        "source claims [WEAK]. Falsifiers must be observable, not vague."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: <4 candidates, scoring matrix incomplete, "
        "no falsifiers pre-registered, killed-ideas check skipped, "
        "ResultPacket schema-invalid."
    ),
    "verification_command": (
        "F=agents/strategist/output_cycle4.md && test -f $F && "
        "FAL=$(grep -ciE 'falsifier' $F) && "
        "CAN=$(grep -cE '^### Candidate ' $F) && "
        "echo \"falsifier_lines=$FAL candidates=$CAN\" && "
        "[ \"$FAL\" -ge 3 ] && [ \"$CAN\" -ge 4 ]"
    ),
    "verification_timeout_secs": 30,
}


def main() -> int:
    print("=" * 70)
    print("Real Phase 0 dispatch: Strategist / cycle 4")
    print("=" * 70)
    print()

    summary = subagent.run_subagent(DISPATCH)

    print()
    print("=" * 70)
    print("Summary returned to Director")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))
    print()
    print("--- Headline ---")
    print(f"  spec_valid:    {summary['spec_valid']}")
    print(f"  result_valid:  {summary['result_valid']}")
    print(f"  verdict:       {summary['verdict']}")
    print(f"  confidence:    {summary['confidence_1to10']}/10")
    print(f"  findings:      {summary['findings_count']}")
    print(f"  output_path:   {summary['output_path']}")
    print(f"  errors:        {summary['errors']}")
    tel = summary.get("telemetry", {})
    print(f"  tokens:        {tel.get('tokens_in')}/{tel.get('tokens_out')}, latency={tel.get('latency_secs')}s")

    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
