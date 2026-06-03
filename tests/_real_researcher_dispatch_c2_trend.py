"""End-to-end Researcher dispatch — Trend velocity lens, cycle 2.

Second real dispatch (after C1 technical lens). Validates:
1. memory_search surfaces the C1 findings so the Researcher doesn't redo work.
2. The telemetry fix produces real numbers (not the hallucinated values we
   saw on the C1 dispatch).
3. A different tool mix (Bash + WebFetch on GitHub API + trending pages)
   exercises the Bash escape-hatch path for sites where WebFetch's UA gets
   blocked.

Scope:
- Trend velocity lens only (1 of 5)
- Focus on agentic LLM tooling: GitHub stars over time, npm download
  velocity for popular agent frameworks, ArXiv citation accumulation.
- Goal: 3-5 ranked velocity signals (durable curves vs hype-cycle flameouts).

Provider: Mistral small (validated on C1; clean tool-call lane).
"""
from __future__ import annotations

import json

from core import subagent

DISPATCH = {
    "dispatch_altitude": "SPEC",
    "role": "researcher",
    "cycle": 2,
    "task": (
        "TREND VELOCITY LENS — bert-lab Phase 0, cycle 2.\n\n"
        "FIRST: call memory_search with 'agentic LLM tool-use' or similar to "
        "surface the C1 technical-lens findings — your job here is to add "
        "VELOCITY context to those signals, not redo the technical scan.\n\n"
        "Then: assess durability of the agentic-LLM trend by looking at: "
        "(a) GitHub stars over time for representative agentic frameworks "
        "(LangGraph, AutoGen, CrewAI, Letta, mcp-agent, smolagents — pick "
        "3-5 to compare) — fetch their public repo metadata via the GitHub "
        "API at https://api.github.com/repos/<owner>/<repo>; "
        "(b) ArXiv submission velocity in cs.AI agentic categories over the "
        "last 6-12 months; "
        "(c) npm/PyPI download counts as proxies where applicable.\n\n"
        "Distinguish 3-month hype-cycle flameouts from durable multi-year "
        "curves. Return 3-5 ranked signals with VELOCITY characterization "
        "(rising / plateau / declining; how steep). For each, note "
        "constraint-alignment with bert-lab's free-tier R&D-to-production "
        "stance."
    ),
    "success_criterion": (
        "findings/researcher_trend_C2.md exists with ≥3 velocity-classified "
        "signals, each citing real metric URLs (GitHub repo / ArXiv list / "
        "npm), AND ResultPacket validates against schemas/result_packet.json "
        "AND at least one signal references a paper from C1 findings (proves "
        "memory_search was used)."
    ),
    "output_path": "findings/researcher_trend_C2.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "Cite real URLs (no fabrication). Use Bash + curl as the GitHub-API "
        "transport (raw JSON cleaner than WebFetch). Cross-reference C1 "
        "findings via memory_search before ranking; if no C1 overlap, that's "
        "noted in the ResultPacket caveats."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: <3 signals; any URL is fabricated; no velocity "
        "characterization (rising/plateau/declining); no cross-reference to "
        "C1 findings; ResultPacket schema-invalid OR self-reported telemetry "
        "(post-fix it should be the real injected numbers)."
    ),
    "verification_command": (
        "F=findings/researcher_trend_C2.md && test -f $F && "
        "C=$(grep -cE 'https?://|arXiv:[0-9]{4}' $F) && "
        "echo \"citations=$C\" && [ \"$C\" -ge 3 ]"
    ),
    "verification_timeout_secs": 30,
}


def main() -> int:
    print("=" * 70)
    print("Real Phase 0 dispatch: Researcher / Trend Velocity / cycle 2")
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
    print()
    tel = summary.get("telemetry", {})
    print(f"  telemetry: provider={tel.get('provider')!r} model={tel.get('model_used')!r}")
    print(f"             tokens={tel.get('tokens_in')}/{tel.get('tokens_out')} latency={tel.get('latency_secs')}s")

    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
