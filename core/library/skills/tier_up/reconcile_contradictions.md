---
name: reconcile_contradictions
version: "1.0"
description: |
  When the lab has accumulated two or more findings that contradict
  each other on a key factual or interpretive point, this skill
  forces resolution: enumerate the contradictions, identify which
  evidence base is stronger or more recent, and produce a
  reconciled stance (or explicitly declare "irreducible
  disagreement — keep both visible to user").
trigger_when: |
  Detected by self_improvement when:
    - two findings have semantically opposite conclusions on the
      same sub-question (cosine distance > 0.8 + opposite polarity)
    - a critic flags a finding as contradicting an earlier one
    - the user explicitly asks "but I thought X — now we're saying Y?"
  Without this skill, contradictory findings rot the corpus and
  poison every downstream synthesis.
inputs:
  findings_a:   {type: dict,   required: true, description: "{cycle, agent, summary, evidence}"}
  findings_b:   {type: dict,   required: true, description: "{cycle, agent, summary, evidence}"}
  question:     {type: string, required: true, description: "The sub-question they disagree on"}
outputs:
  resolution_type: {type: string, description: "reconciled | a_wins | b_wins | irreducible"}
  reconciled_stance: {type: string, description: "The new official lab position (or 'see both findings' for irreducible)"}
  losing_findings:  {type: list,  description: "Findings whose claims should be archived/superseded"}
  rationale:        {type: string, description: "Why this resolution, with evidence pointers"}
tools_required: [reconcile_pair]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: arbitrate
    tool: reconcile_pair
    args:
      a: "{{findings_a}}"
      b: "{{findings_b}}"
      question: "{{question}}"
    capture: arbitration
  - id: pluck_type
    tool: identity
    args:
      value: "{{arbitration.resolution_type}}"
    capture: resolution_type
  - id: pluck_stance
    tool: identity
    args:
      value: "{{arbitration.reconciled_stance}}"
    capture: reconciled_stance
  - id: pluck_losing
    tool: identity
    args:
      value: "{{arbitration.losing_findings}}"
    capture: losing_findings
  - id: pluck_rationale
    tool: identity
    args:
      value: "{{arbitration.rationale}}"
    capture: rationale
  - id: ledger
    skill: ledger_row_authoring
    args:
      event_type: "contradiction_reconciled"
      cycle_id: 0
      agent: "reconcile_contradictions"
      payload:
        question: "{{question}}"
        resolution_type: "{{arbitration.resolution_type}}"
        rationale: "{{arbitration.rationale}}"
    capture: ledger_row
failure_modes:
  - condition: "Both findings cite same evidence base — meta-disagreement"
    handler: "emit_same_evidence_irreducible"
  - condition: "reconcile_pair returns malformed verdict"
    handler: retry
    max_retries: 1
---

# reconcile_contradictions

**Quality bar**: never silently drop the losing finding from the corpus —
it stays in the source-of-record ledger marked as superseded. Hiding past
positions destroys the lab's ability to learn from its own corrections.
