"""Re-spec C1 — Researcher: canvas patterns for structured artifacts.

After PI's 0.001% evaluation paused Phase 1 launch, this is the first
of three re-spec dispatches: Researcher → Strategist → Architect.

The Researcher's job here is NOT to surface ArXiv papers. It's to scan
how consumer-grade canvas / workflow tools actually handle structured
artifacts and produce patterns we can build on. Outputs feed directly
into the next Strategist dispatch (S1) which produces the v0.5 spec.

Run: PYTHONPATH=. uv run python tests/_real_researcher_canvas_patterns_R1.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "META",
    "role": "researcher",
    "cycle": 8,  # post-Phase-0; first cycle of the re-spec arc
    "task": (
        "RE-SPEC LANDSCAPE — canvas/workflow tools and how they handle\n"
        "structured artifacts.\n\n"
        "Context: PI declared the current CanvasAgent skeleton is 0.001%\n"
        "of what it should be. The single load-bearing differentiator —\n"
        "structured artifacts flowing through a typed graph — isn't there.\n"
        "Your job: surface PATTERNS from existing consumer/prosumer tools\n"
        "so the next Strategist (S1) can write a concrete v0.5 spec.\n\n"
        "READ FIRST:\n"
        "- memories/governance/pi_notes.md '0.001% evaluation' section\n"
        "- findings/strategist_phase0_C7.md (the constraint set we're\n"
        "  still bound by — UX/structured-data/representation/interactability)\n"
        "- memories/mission.md (CanvasAgent's lockdown details)\n"
        "- phase1/canvasagent/src/AgentNode.tsx (the current state — note\n"
        "  the single node type and <pre> text output)\n\n"
        "TARGETS — pick 5-7 of these tools, NOT all of them:\n"
        "  Tier A (graph-based agent canvases — most directly relevant):\n"
        "    - Langflow (https://github.com/langflow-ai/langflow)\n"
        "    - AutoGen Studio (Microsoft, https://microsoft.github.io/autogen/)\n"
        "    - ComfyUI (image-gen, but the typed-port pattern is gold)\n"
        "    - Flowise (open-source LangChain canvas)\n"
        "  Tier B (workflow automation — node libraries + typed I/O):\n"
        "    - n8n (https://n8n.io)\n"
        "    - Pipedream (https://pipedream.com)\n"
        "    - Zapier visual editor\n"
        "  Tier C (structured-data tools — schema-first artifact thinking):\n"
        "    - Airtable (typed columns, view abstractions)\n"
        "    - Notion databases (typed properties + multiple views)\n"
        "    - Retool components (typed component library)\n"
        "  Pick 2-3 from A, 2 from B, 1-2 from C.\n\n"
        "FOR EACH PICKED TOOL — answer concretely (cite real docs / blog\n"
        "posts / GitHub READMEs / videos with URLs; mark [WEAK] for\n"
        "single-source claims):\n"
        "  1. Node-type library — how many distinct node types? Examples\n"
        "     of the most-used 5-10. How are they categorized in the UI\n"
        "     (sidebar groupings, search, drag-drop)?\n"
        "  2. Output schemas — what TYPES of outputs does a node produce?\n"
        "     Free-text? Typed JSON? Tables? Files? Multi-port outputs?\n"
        "  3. Editing affordances — once an output appears, can the user\n"
        "     edit it directly? Drill into nested structure? Re-run from\n"
        "     a specific point with edits?\n"
        "  4. Branching / replay — can users fork from a prior run, edit,\n"
        "     and re-run forward? How is run history surfaced?\n"
        "  5. Visual language — sketch-level (in words). What's the\n"
        "     density per node? How are types visually distinguished?\n"
        "     Animations? Polish level (consumer or developer-utilitarian)?\n"
        "  6. Consumer-vs-developer framing — onboarding flow, jargon\n"
        "     density, default samples. Who do they actually serve?\n\n"
        "THEN CLUSTER — cross-tool synthesis is the highest-value output:\n"
        "  - Patterns that ALL the picked tools converge on\n"
        "    (these are table-stakes for v0.5; bert can't skip them)\n"
        "  - Patterns that only the best tools get right\n"
        "    (these are differentiators bert can compound on)\n"
        "  - Patterns that ALL the tools still get wrong\n"
        "    (these are the gaps the user noticed when they said 0.001%;\n"
        "     bert's opportunity space)\n\n"
        "FINISH WITH a 'recommended seed candidates for v0.5 spec' section\n"
        "(8-15 specific node types you think bert should ship, with one-\n"
        "line rationale each). The Strategist S1 will pick the final 6-10.\n\n"
        "TOOLS YOU USE: WebSearch (heavy use), WebFetch (cite real URLs),\n"
        "memory_search to recall any prior bert findings touching these\n"
        "tools (probably few; this is mostly fresh research). Multi-source\n"
        "verification: ≥2 independent URLs per non-WEAK claim. Avoid\n"
        "ArXiv-style citation framing — these are tools, not papers.\n\n"
        "WRITE INCREMENTALLY: this is a long synthesis. The PRIOR attempt\n"
        "exhausted output tokens trying to write the entire report in one\n"
        "model response. To avoid that:\n"
        "  1. Early: gather data, then Write a SKELETON to "
        "findings/researcher_canvas_patterns_R1.md with section headers\n"
        "     plus a 'tools selected:' list. Save and commit to disk.\n"
        "  2. Then in subsequent iterations, use Edit to insert content\n"
        "     for ONE tool at a time. After each Edit, you have a fresh\n"
        "     output budget. Don't try to write all 5-7 tools in one shot.\n"
        "  3. The cross-tool synthesis section can be the final Edit pass.\n"
        "  4. Final: write the ResultPacket JSON to result_path.\n"
        "Treat each Write/Edit as committing a small piece. The harness\n"
        "doesn't reset state between iterations — partial progress is\n"
        "preserved. Watch your output budget across each model response."
    ),
    "success_criterion": (
        "findings/researcher_canvas_patterns_R1.md exists with: 5-7 tool "
        "characterizations (each covering the 6 dimensions), a cross-"
        "tool synthesis section with 'all converge / best ones / all "
        "miss' clusters, and a 'recommended seed candidates' list of "
        "8-15 node types. Total ≥30 real URLs cited."
    ),
    "output_path": "findings/researcher_canvas_patterns_R1.md",
    # Cerebras Qwen-3 235B — bigger output ceiling than mistral-small,
    # better for long synthesis dispatches like this one. Validated on
    # earlier smoke + C99 dispatches.
    "model": "cerebras/qwen-3-235b-a22b-instruct-2507",
    "process_hygiene": (
        "PATTERNS NOT CITATIONS. This is product research, not academic. "
        "The Strategist will eat this raw — make it concrete: 'Langflow "
        "uses a left sidebar grouped by Models / Memory / Chains / Tools; "
        "drag any block to canvas; output of one block hooks into typed "
        "input of next via colored handles per type' beats 'Langflow has "
        "a node-based interface.' "
        "Actually USE WebFetch on each tool's docs/README/screenshot URL "
        "— don't just summarize from search snippets. Cite real URLs not "
        "fabricated ones. Bash + curl for GitHub API on stargazer counts. "
        "[WEAK] discipline applied to single-source claims."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) <5 tools characterized; (b) tool entries "
        "miss any of the 6 required dimensions; (c) no cross-tool "
        "synthesis section; (d) <30 cited URLs; (e) <8 seed candidates "
        "for v0.5 spec; (f) ResultPacket schema-invalid."
    ),
    "verification_command": (
        "set -eo pipefail && "
        "F=findings/researcher_canvas_patterns_R1.md && test -f $F && "
        "URLS=$(grep -cE 'https?://' $F) && "
        "echo \"urls=$URLS\" && [ \"$URLS\" -ge 30 ]"
    ),
    "verification_timeout_secs": 60,
}


def main() -> int:
    print("=" * 72)
    print("RE-SPEC C1 — Researcher: canvas patterns for structured artifacts")
    print("=" * 72)
    print()

    summary = subagent.run_subagent(DISPATCH)

    print()
    print("=" * 72)
    print("Researcher return")
    print("=" * 72)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "calibration_reasoning"},
                     indent=2, default=str))
    print()
    print("--- calibration_reasoning ---")
    print(summary.get("calibration_reasoning", "")[:1500])
    print()
    tel = summary.get("telemetry", {})
    verify = tel.get("verification") if isinstance(tel, dict) else None
    if verify:
        print(f"--- verification: ok={verify['ok']} ({verify.get('elapsed_ms', 0)}ms) ---")
        print(f"  stdout tail: {verify.get('stdout', '')[-400:]}")
    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
