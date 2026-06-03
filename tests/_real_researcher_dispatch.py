"""End-to-end Researcher dispatch — Technical lens, ArXiv 90-day window.

Real dispatch (uses real provider tokens) — exercises the full chain:
  Spawn → DispatchSpec validation → agent loop → WebSearch + WebFetch +
  memory_search + Write → ResultPacket validation → return summary.

Run as:
    PYTHONPATH=. uv run python tests/_real_researcher_dispatch.py

Scoped tightly:
- Technical lens only (1 of 5)
- ArXiv only (don't crawl GitHub / Reddit / HN — keeps token budget bounded)
- 90-day window (not full historical scan)
- Goal: 3-5 ranked signals with multi-source verification

Provider: Cerebras Qwen-3 235B (validated to handle parallel tool calls in
the smoke test; high RPM ceiling; falls back gracefully on 429 via the
universal RETRYABLE_STATUSES retry).
"""
from __future__ import annotations

import json

from core import subagent

DISPATCH = {
    "dispatch_altitude": "SPEC",
    "role": "researcher",
    "cycle": 1,
    "task": (
        "TECHNICAL LANDSCAPE LENS — bert-lab Phase 0, cycle 1.\n\n"
        "Scan ArXiv (cs.AI / cs.CL / cs.LG) for papers published in the LAST 90 "
        "DAYS that signal capability inflection points relevant to bert-lab's "
        "free-tier autonomous R&D-to-production constraint. Focus areas: "
        "(a) agentic LLM tool-use reliability + cost reduction, "
        "(b) on-device or BYO-key inference improvements, "
        "(c) multi-agent orchestration patterns proven in production, "
        "(d) memory / retrieval architectures that beat vanilla RAG. "
        "Use WebSearch with site:arxiv.org filters; WebFetch the top 5-7 "
        "abstracts. Cluster the signals; pick 3-5 high-leverage ones; "
        "for each note distance from bert's free-tier constraint."
    ),
    "success_criterion": (
        "findings/researcher_technical_C1.md exists with ≥3 ranked signals, "
        "each citing ≥1 ArXiv URL, with [WEAK] markers on single-source claims, "
        "AND ResultPacket validates against schemas/result_packet.json."
    ),
    "output_path": "findings/researcher_technical_C1.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "Multi-source verification — mark [WEAK] for any signal with only one "
        "supporting paper. Cross-reference against memories/killed.md before "
        "ranking signals. Cite real ArXiv URLs (not synthesized)."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: <3 signals, any citation is fabricated, no [WEAK] "
        "discipline applied to single-source claims, ResultPacket schema-invalid."
    ),
    "verification_command": (
        "F=findings/researcher_technical_C1.md && test -f $F && "
        "C=$(grep -cE 'https?://|arXiv:[0-9]{4}' $F) && "
        "echo \"citations=$C\" && [ \"$C\" -ge 3 ]"
    ),
    "verification_timeout_secs": 30,
}


def main() -> int:
    print("=" * 70)
    print("Real Phase 0 dispatch: Researcher / Technical lens / cycle 1")
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
    print(f"  result_path:   {summary['result_path']}")
    print(f"  errors:        {summary['errors']}")

    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
