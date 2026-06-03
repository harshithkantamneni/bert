"""Phase 0 EXIT — Strategist re-run with refined goal as a HARD constraint.

PI feedback after the C6 ranking: the goal should be optimized for
  (a) ease of use,
  (b) experience working with structured data,
  (c) data represented properly (visual, not just text),
  (d) best-in-class interactability.

This dispatch re-applies the same 5-lens corpus but adds these as a
hard filter. Candidates that don't satisfy them get demoted regardless
of other strengths. Output:
  findings/strategist_phase0_C7.md  — full re-evaluation matrix
  state/proposals_pending_pi.md     — overwritten with new top-3 brief

Run:
    PYTHONPATH=. uv run python tests/_real_strategist_phase0_exit_v2.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "META",
    "role": "strategist",
    "cycle": 7,
    "task": (
        "PHASE 0 EXIT v2 — re-rank under refined goal.\n\n"
        "PI HAS REFINED THE GOAL. Apply this as a HARD CONSTRAINT, not a "
        "tiebreaker. Demote candidates that don't satisfy it regardless of "
        "their other strengths:\n\n"
        "  GOAL: bert's first product should optimize for\n"
        "    (a) EASE OF USE — consumer-grade, low-friction, polished UX;\n"
        "        no CLI-only, no manual config, no dev-tool framing\n"
        "    (b) STRUCTURED DATA — primary value comes from working with\n"
        "        structured artifacts (tables, graphs, timelines, trees,\n"
        "        action-item boards), not just text/audio streams\n"
        "    (c) PROPER REPRESENTATION — data is rendered visually,\n"
        "        spatially, hierarchically; not blob-of-text\n"
        "    (d) RICH INTERACTABILITY — direct manipulation: click, drag,\n"
        "        edit-in-place, drill-down, replay, branch alternatives\n\n"
        "ALL FOUR must be present. A candidate that's beautifully designed "
        "but only outputs text fails (b)+(c). A candidate with great "
        "structured output but a dev-CLI fails (a). A candidate where AI "
        "outputs are read-only fails (d).\n\n"
        "INPUTS: read all existing Phase 0 lens findings:\n"
        "- findings/researcher_technical_C1.md\n"
        "- findings/researcher_trend_C2.md\n"
        "- findings/technical_landscape_C2_researcher.md\n"
        "- agents/researcher/output_cycle3.md\n"
        "- findings/researcher_user_pain_C5.md\n"
        "- findings/researcher_market_gap_C5.md\n"
        "- findings/researcher_constraint_alignment_C5.md\n"
        "- agents/strategist/output_cycle4.md (prior C4 matrix)\n"
        "- findings/strategist_phase0_C6.md (prior C6 PI exit proposal)\n\n"
        "Re-cluster all 89 prior signals through the refined goal. Honestly "
        "demote VisionMemo (text-first), AgentOS (dev framework, fails ease-"
        "of-use), CrossModel Chat (chat UI = unstructured). Promote candidates "
        "where AI agent output IS the structured artifact: e.g., visual agent "
        "graphs, structured note canvases, interactive timeline/topic boards, "
        "knowledge-graph editors, replayable decision trees.\n\n"
        "Generate 5-7 candidates, evaluate each on:\n"
        "  - the original 6-dim constraint matrix (free-tier / on-device / "
        "    single-dev / dist / substitutability / interestingness)\n"
        "  - the NEW 4-dim goal matrix (ease-of-use / structured-data / "
        "    representation / interactability), each scored 0-3\n"
        "  - cross-lens triangulation (must triangulate ≥3 lenses)\n"
        "  - pre-registered observable falsifiers (≥2 per finalist)\n\n"
        "OUTPUT 1: findings/strategist_phase0_C7.md — full matrix.\n\n"
        "OUTPUT 2 (MANDATORY): use Write to populate "
        "state/proposals_pending_pi.md (overwriting the C6 brief) with a "
        "≤500-word brief containing ONLY:\n"
        "  - Executive summary (≤80 words) including how this re-ranking "
        "    differs from C6\n"
        "  - Top 3 candidates ranked, each with: name + ≤25-word value prop, "
        "    distribution channel, tech stack, 2-3 falsifiers, time-to-"
        "    prototype, NEW-goal scores (UX/SD/REP/INT each 0-3)\n"
        "  - 'What changed from C6' paragraph: which prior finalists got "
        "    demoted and why\n"
        "  - Open questions for PI\n"
        "  - Final line: 'Awaiting PI approval to begin Phase 1 on candidate "
        "    #N.'"
    ),
    "success_criterion": (
        "BOTH findings/strategist_phase0_C7.md AND a re-written "
        "state/proposals_pending_pi.md exist. The PI brief ranks 3 finalists, "
        "each scoring ≥9/12 on the 4-dim NEW goal matrix and satisfying ALL "
        "FOUR refined criteria. Falsifiers are observable signals."
    ),
    "output_path": "findings/strategist_phase0_C7.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "Be honest about the demotion of prior candidates — name VisionMemo, "
        "AgentOS, CrossModel Chat explicitly and explain why each fails one "
        "or more of the new criteria. The new criteria are a HARD filter, "
        "not a tiebreaker. If a prior candidate can be re-shaped to pass "
        "(e.g., VisionMemo → structured-memo-canvas), call out the pivot "
        "explicitly so PI sees it as a refinement not a contradiction."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: state/proposals_pending_pi.md missing or >500 "
        "words; <3 finalists; any finalist scores <9/12 on the new goal "
        "matrix; falsifiers not observable; ResultPacket schema-invalid."
    ),
    "verification_command": (
        "test -f findings/strategist_phase0_C7.md && "
        "test -f state/proposals_pending_pi.md && "
        "PI_WORDS=$(wc -w < state/proposals_pending_pi.md) && "
        "FAL=$(grep -ciE 'falsifier' findings/strategist_phase0_C7.md) && "
        "echo \"pi_brief_words=$PI_WORDS falsifier_lines=$FAL\" && "
        "[ \"$PI_WORDS\" -le 900 ] && [ \"$FAL\" -ge 3 ]"
    ),
    "verification_timeout_secs": 30,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 0 EXIT v2 — Strategist re-rank under refined goal")
    print("=" * 72)
    print()

    summary = subagent.run_subagent(DISPATCH)

    print()
    print("=" * 72)
    print("Summary returned to Director")
    print("=" * 72)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "calibration_reasoning"},
                     indent=2, default=str))
    print()
    print("--- calibration_reasoning ---")
    print(summary.get("calibration_reasoning", ""))
    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
