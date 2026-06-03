"""Run the remaining 3 Phase 0 Researcher lenses sequentially.

We've covered Technical (C1, C2, C3) and Trend Velocity (C2). The remaining
three lenses from pi_notes.md:
  - User Pain: Reddit, HN, GitHub issues, X complaints
  - Market Gap: App Store / Play Store / ProductHunt / Chrome Web Store
  - Constraint Alignment: each surfaced signal vs bert's free-tier / on-device
    / single-dev / standard-distribution constraints

This script runs them as 3 separate sub-agent dispatches so each gets a
clean context window. Each writes:
  - agents/researcher/output_cycle5_<lens>.md (full report)
  - state/results/researcher_C5_<tag>.json (ResultPacket)

After this completes, all 5 lenses will have findings and the Strategist
can be re-run with full-coverage input.

Run:
    PYTHONPATH=. uv run python tests/_real_remaining_lenses_dispatch.py
"""
from __future__ import annotations

import sys

from core import subagent

MODEL = "mistral/mistral-small-latest"
CYCLE = 5

LENSES = [
    {
        "name": "user_pain",
        "altitude": "SPEC",
        "task": (
            "USER PAIN LANDSCAPE LENS — bert-lab Phase 0, cycle 5.\n\n"
            "Surface 4-6 active pain points from end-users where current AI "
            "tooling falls short. Sources to query (use WebSearch then "
            "WebFetch on best results, or Bash+curl for HN's API at "
            "https://hacker-news.firebaseio.com/v0/topstories.json):\n"
            "- Reddit: r/LocalLLaMA, r/MachineLearning, r/programming, "
            "  r/productivity, r/selfhosted, r/iOS, r/Android — search for "
            "  'pain', 'frustrated', 'wish there was', 'can't believe'.\n"
            "- HackerNews: top stories + comments tagged 'AI'/'LLM' from "
            "  the last 60 days.\n"
            "- GitHub: issues labeled 'help wanted' or 'frustration' on "
            "  popular AI/agent repos (CrewAI, AutoGen, LangGraph, Letta).\n"
            "- X: search for 'AI sucks' / 'AI hallucinates' / 'tried agentic'.\n\n"
            "For each pain point: cluster by frequency × severity. Cite ≥2 "
            "independent sources or mark [WEAK]. Note which would be solved "
            "by bert's free-tier / on-device / agentic-LLM stack."
        ),
        "output_path": "findings/researcher_user_pain_C5.md",
        "success_criterion": (
            "findings/researcher_user_pain_C5.md exists with ≥4 "
            "ranked pain points each citing ≥1 real URL, [WEAK] applied to "
            "single-source claims, AND ResultPacket schema-validates."
        ),
        "process_hygiene": (
            "Multi-source verification — Reddit threads + HN comments + "
            "GitHub issues; mark [WEAK] for single-source. Cite real URLs."
        ),
        "falsifier_text": (
            "Failure if any of: <4 pain points; any URL fabricated; no [WEAK] "
            "discipline; ResultPacket schema-invalid."
        ),
    },
    {
        "name": "market_gap",
        "altitude": "SPEC",
        "task": (
            "MARKET GAP LANDSCAPE LENS — bert-lab Phase 0, cycle 5.\n\n"
            "Surface 3-5 'this should exist but doesn't' product gaps in "
            "consumer AI tooling. Sources to query:\n"
            "- ProductHunt top launches (last 90 days) — what categories are "
            "  underserved? Use WebSearch site:producthunt.com.\n"
            "- Mac App Store / Microsoft Store top charts — productivity, "
            "  developer tools, AI categories. Use WebSearch site:apps.apple.com.\n"
            "- Chrome Web Store — extensions categories.\n"
            "- GitHub trending — what concepts are getting starred but lack "
            "  polished consumer products? Use WebFetch on github.com/trending.\n"
            "- HackerNews 'Show HN' threads — what people are launching that "
            "  hints at unsatisfied demand. Bash+curl to "
            "  https://hn.algolia.com/api/v1/search?tags=show_hn&hitsPerPage=20.\n\n"
            "For each gap: name what should exist, identify the unmet need, "
            "list nearest substitutes that fall short. Cite ≥2 sources per "
            "gap; mark single-source claims [WEAK]."
        ),
        "output_path": "findings/researcher_market_gap_C5.md",
        "success_criterion": (
            "findings/researcher_market_gap_C5.md exists with ≥3 "
            "named gaps with substitute analysis and real source URLs, AND "
            "ResultPacket schema-validates."
        ),
        "process_hygiene": (
            "Multi-source — ProductHunt + App Store charts + HN/GitHub "
            "trending; cite URLs; mark [WEAK] for single-source."
        ),
        "falsifier_text": (
            "Failure if any of: <3 gaps; substitutes unspecified; URLs "
            "fabricated; ResultPacket schema-invalid."
        ),
    },
    {
        "name": "constraint_alignment",
        "altitude": "SPEC",
        "task": (
            "CONSTRAINT ALIGNMENT LENS — bert-lab Phase 0, cycle 5.\n\n"
            "Re-evaluate the strongest signals from prior Researcher findings "
            "against bert's hard constraints. Use memory_search to recall:\n"
            "- 'agentic LLM tool-use' (C1 technical)\n"
            "- 'GitHub stars trend agent' (C2 trend velocity)\n"
            "- 'on-device LLM Llama 4 Nemotron' (C2 technical landscape)\n"
            "- 'real-time video edge inference' (C3 technical)\n\n"
            "For each top signal, score against the 4 constraints:\n"
            "  1. Free-tier inference (NVIDIA/Cerebras/Groq/Gemini/Mistral/"
            "OpenRouter/HF Router/Ollama) — does it fit?\n"
            "  2. On-device or BYO-key — can users run it locally or with "
            "their own keys, no shared SaaS spend?\n"
            "  3. Single-developer build — can one person ship v1 in ≤30 days?\n"
            "  4. Standard distribution — Mac/Windows/iOS/Android stores, "
            "Chrome Web Store, GitHub releases, npm — no exotic channels?\n\n"
            "For each signal output: 4-row constraint table + total score "
            "(0-12) + verdict (PROCEED, NEEDS_MITIGATION, OUT_OF_SCOPE). "
            "Recommend the top 3 constraint-aligned signals for the "
            "Strategist's next ranking pass."
        ),
        "output_path": "findings/researcher_constraint_alignment_C5.md",
        "success_criterion": (
            "findings/researcher_constraint_alignment_C5.md exists "
            "with ≥6 signals scored against the 4 constraints, top-3 "
            "recommendation, AND ResultPacket schema-validates."
        ),
        "process_hygiene": (
            "Use memory_search to ground in prior findings; do not invent "
            "new signals here — this lens evaluates existing ones. Score "
            "honestly; OUT_OF_SCOPE is fine and useful."
        ),
        "falsifier_text": (
            "Failure if any of: <6 signals scored; constraint table missing; "
            "no recommendation; ResultPacket schema-invalid."
        ),
    },
]


def run_one(lens: dict) -> dict:
    output_path = lens["output_path"]
    spec = {
        "dispatch_altitude": lens["altitude"],
        "role": "researcher",
        "cycle": CYCLE,
        "task": lens["task"],
        "success_criterion": lens["success_criterion"],
        "output_path": output_path,
        "model": MODEL,
        "process_hygiene": lens["process_hygiene"],
        "confidence_required": True,
        "falsifier_text": lens["falsifier_text"],
        # External verification: file exists AND has ≥3 URL citations.
        # If the agent didn't actually research, this catches it.
        "verification_command": (
            f"F={output_path} && test -f $F && "
            "C=$(grep -cE 'https?://|arXiv:[0-9]{4}' $F) && "
            "echo \"citations=$C\" && [ \"$C\" -ge 3 ]"
        ),
        "verification_timeout_secs": 30,
    }
    print(f"\n{'='*70}\nDispatching: {lens['name']}\n{'='*70}")
    summary = subagent.run_subagent(spec)
    return summary


def main() -> int:
    results = []
    for lens in LENSES:
        summary = run_one(lens)
        results.append((lens["name"], summary))
        print(f"  → verdict={summary['verdict']} confidence={summary['confidence_1to10']}/10 "
              f"valid={summary['result_valid']} errors={summary['errors']}")

    print()
    print("=" * 70)
    print("3-lens dispatch summary")
    print("=" * 70)
    all_ok = True
    for name, s in results:
        ok = s["spec_valid"] and s["result_valid"]
        all_ok = all_ok and ok
        tel = s.get("telemetry", {})
        print(f"  [{name:22s}] verdict={s['verdict']:25s} "
              f"conf={s['confidence_1to10']}/10 "
              f"tokens={tel.get('tokens_in', 0)}/{tel.get('tokens_out', 0)} "
              f"latency={tel.get('latency_secs', 0)}s  "
              f"{'OK' if ok else 'FAIL ' + str(s['errors'])}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
