"""Phase 0 EXIT — final Strategist dispatch with full 5-lens coverage.

Inputs: all 8 finding files across the 5 lenses, plus the C4 Strategist
matrix (which only had 2 lenses). Goal: produce a PI-readable proposal
ranking 3 finalists with the discipline pi_notes requires:
  (a) value proposition
  (b) distribution channel
  (c) tech stack within free-tier constraints
  (d) pre-registered falsifiers (2-3 per finalist)
  (e) explicit ask for user approval

Dual-write contract:
  output_path → findings/strategist_phase0_C6.md   (full evaluation matrix)
  Strategist must ALSO use Write to populate
                  state/proposals_pending_pi.md     (distilled PI brief, ≤500 words)

Run: PYTHONPATH=. uv run python tests/_real_strategist_phase0_exit.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "META",
    "role": "strategist",
    "cycle": 6,
    "task": (
        "PHASE 0 EXIT SYNTHESIS — final ranking before PI approval.\n\n"
        "Read ALL existing Phase 0 lens findings:\n"
        "- findings/researcher_technical_C1.md (technical, 5 signals)\n"
        "- findings/researcher_trend_C2.md (trend velocity, 5 signals)\n"
        "- findings/technical_landscape_C2_researcher.md (technical, 12 inflection points)\n"
        "- agents/researcher/output_cycle3.md (technical, 7 inflection points)\n"
        "- findings/researcher_user_pain_C5.md (user pain, 6 ranked points)\n"
        "- findings/researcher_market_gap_C5.md (market gap, 4 named gaps)\n"
        "- findings/researcher_constraint_alignment_C5.md (9 prior signals scored)\n"
        "- agents/strategist/output_cycle4.md (prior C4 Strategist 6-candidate matrix)\n\n"
        "Cluster signals across all 5 lenses. Promote candidates that triangulate "
        "across ≥3 lenses (technical viability + user pain + market gap + trend "
        "velocity + constraint alignment). Apply the 6-dim constraint matrix "
        "from prompts/strategist.md. Pre-register 2-3 falsifiers per finalist.\n\n"
        "OUTPUT 1 (full matrix): write the full evaluation to "
        "findings/strategist_phase0_C6.md — same structure as prior strategist "
        "outputs (Executive Summary, ranked candidates with full eval, killed-"
        "ideas check, open questions for PI).\n\n"
        "OUTPUT 2 (PI brief, MANDATORY): use the Write tool to populate "
        "state/proposals_pending_pi.md with a ≤500-word distilled brief "
        "containing ONLY:\n"
        "  - One-paragraph executive summary (≤80 words) with your top "
        "    recommendation and confidence.\n"
        "  - Top 3 candidates ranked, each with:\n"
        "      * Name + one-line value prop (≤25 words)\n"
        "      * Distribution channel (one line)\n"
        "      * Tech stack within bert's free-tier constraints (one line)\n"
        "      * 2-3 falsifiers (each ≥30 chars, observable)\n"
        "      * Time-to-prototype estimate (one line)\n"
        "  - One paragraph 'open questions for PI' — what additional "
        "    constraints or signals would change your ranking.\n"
        "  - Final line: 'Awaiting PI approval to begin Phase 1 on candidate #N.'"
    ),
    "success_criterion": (
        "BOTH findings/strategist_phase0_C6.md AND state/proposals_pending_pi.md "
        "exist after the cycle. The PI brief at state/proposals_pending_pi.md is "
        "≤500 words, ranks exactly 3 finalists, each with all 5 required fields, "
        "and ResultPacket schema-validates."
    ),
    "output_path": "findings/strategist_phase0_C6.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "Cross-cycle synthesis — cite each candidate's supporting evidence "
        "from the original Researcher finding it came from. Falsifiers must "
        "be observable signals (e.g., 'GitHub stars stagnate below X over Y "
        "months', 'PyPI downloads <Z/week after launch'), not vague qualitative. "
        "The PI brief is the document that gates Phase 1 entry — clarity > "
        "comprehensiveness."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) state/proposals_pending_pi.md missing or >500 "
        "words, (b) <3 finalists, (c) any finalist missing falsifiers, (d) "
        "ResultPacket schema-invalid, (e) any candidate cited without source "
        "from a real prior finding file."
    ),
    # Verifies BOTH the full matrix AND the dual-write to PI brief
    "verification_command": (
        "test -f findings/strategist_phase0_C6.md && "
        "test -f state/proposals_pending_pi.md && "
        "PI_WORDS=$(wc -w < state/proposals_pending_pi.md) && "
        "FAL=$(grep -ciE 'falsifier' findings/strategist_phase0_C6.md) && "
        "echo \"pi_brief_words=$PI_WORDS falsifier_lines=$FAL\" && "
        "[ \"$PI_WORDS\" -le 900 ] && [ \"$FAL\" -ge 3 ]"
    ),
    "verification_timeout_secs": 30,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 0 EXIT — Strategist final synthesis")
    print("=" * 72)
    print()

    summary = subagent.run_subagent(DISPATCH)

    print()
    print("=" * 72)
    print("Summary returned to Director")
    print("=" * 72)
    print(json.dumps(summary, indent=2, default=str))

    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
