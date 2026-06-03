---
name: adversarial_audit
version: "1.0"
description: |
  Deeper than red_team_pass: this skill assumes the project's current
  output WILL be scrutinized by a hostile reviewer (investor, peer,
  competitor) and pre-emptively attacks it from every angle:
  methodology, definitions, sample selection, benchmark validity,
  external validity, missing comparisons, cherry-picking, statistical
  weakness. Outputs the gauntlet the artifact must survive.
trigger_when: |
  Before a finalize_project pass on a high-stakes artifact (proof
  packet, investor demo, public release). Also triggered by
  governance_gate when the artifact's risk_score exceeds threshold.
  This is "show me the worst-case interpretation" mode.
inputs:
  artifact_path: {type: string, required: true}
  context:       {type: string, default: "",   description: "Mission objective + audience (e.g., 'series A pitch')"}
  audience:      {type: string, default: "investor", description: "investor | peer | hostile_competitor | regulator"}
outputs:
  attacks:       {type: list,   description: "[{vector, hostile_claim, evidence_required, survival_probability: 0-1}]"}
  must_address:  {type: list,   description: "Subset of attacks where survival_probability < 0.4 — MUST be fixed before release"}
  ship_decision: {type: string, description: "ship | revise | block_release"}
  blocker_count: {type: int,    description: "Number of must_address items"}
tools_required: [Read, hostile_review]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: load_artifact
    tool: Read
    args:
      file_path: "{{artifact_path}}"
    capture: artifact_body
  - id: hostile_pass
    tool: hostile_review
    args:
      artifact: "{{artifact_body}}"
      context: "{{context}}"
      audience: "{{audience}}"
    capture: attacks_raw
  - id: classify_severity
    tool: classify_attacks_by_survival
    args:
      attacks: "{{attacks_raw.attacks}}"
    capture: classified
  - id: ship_gate
    tool: ship_gate_decision
    args:
      attacks: "{{classified.attacks}}"
      audience: "{{audience}}"
    capture: gate
  - id: pluck_attacks
    tool: identity
    args:
      value: "{{classified.attacks}}"
    capture: attacks
  - id: pluck_must
    tool: identity
    args:
      value: "{{classified.must_address}}"
    capture: must_address
  - id: pluck_ship
    tool: identity
    args:
      value: "{{gate.decision}}"
    capture: ship_decision
  - id: pluck_blockers
    tool: identity
    args:
      value: "{{gate.blocker_count}}"
    capture: blocker_count
failure_modes:
  - condition: "Artifact file unreadable"
    handler: "emit_artifact_unreadable"
  - condition: "hostile_review returns < 5 attack vectors for investor audience"
    handler: retry
    max_retries: 1
---

# adversarial_audit

**Quality bar**: a `ship` decision with `blocker_count > 0` is a
contradiction — never lie to yourself about what's already been
flagged. Likewise, an audit that produces fewer than 5 attacks for
investor audience didn't try hard enough.
