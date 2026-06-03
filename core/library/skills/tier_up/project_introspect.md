---
name: project_introspect
version: "1.0"
description: |
  Tier-up skill: when a routine role hits a self-doubt threshold
  ("am I in the right loop?", "is my goal still the right goal?"),
  this skill steps back and reads the project's seed_brief.md, the
  most recent N findings, and the lab/state/now.md, then produces
  a structured assessment of whether the current trajectory still
  serves the original objective.
trigger_when: |
  Activated by self_improvement signals when:
    - acceptance_rate is dropping cycle-over-cycle
    - the same finding keeps surfacing without new info
    - a role's tasks have repeated the same prompt 3+ times
    - the director explicitly requests reorientation
  This is a meta-cognitive skill — it interrupts the work to check
  if the work is still right.
inputs:
  seed_path:     {type: string, default: "lab/seed_brief.md"}
  state_path:    {type: string, default: "lab/state/now.md"}
  recent_n:      {type: int,    default: 10, description: "How many recent findings to consider"}
  findings_dir:  {type: string, default: "findings/"}
outputs:
  on_track:      {type: bool,   description: "Is the lab still aligned with seed objective?"}
  drift_observations: {type: list, description: "[{type, evidence, severity: low|med|high}]"}
  recommended_action: {type: string, description: "continue | reorient | reset_cycle | escalate_to_human"}
  rationale:     {type: string, description: "1-2 paragraph plain-language reasoning"}
tools_required: [Read, list_recent_findings]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: read_seed
    tool: Read
    args:
      file_path: "{{seed_path}}"
    capture: seed_text
  - id: read_state
    tool: Read
    args:
      file_path: "{{state_path}}"
    capture: state_text
  - id: list_recent
    tool: list_recent_findings
    args:
      dir: "{{findings_dir}}"
      n: "{{recent_n}}"
    capture: recent
  - id: analyze_alignment
    tool: introspect_alignment
    args:
      seed: "{{seed_text}}"
      current_state: "{{state_text}}"
      recent_findings: "{{recent.findings}}"
    capture: assessment
  - id: pluck_on_track
    tool: identity
    args:
      value: "{{assessment.on_track}}"
    capture: on_track
  - id: pluck_drift
    tool: identity
    args:
      value: "{{assessment.drift_observations}}"
    capture: drift_observations
  - id: pluck_action
    tool: identity
    args:
      value: "{{assessment.recommended_action}}"
    capture: recommended_action
  - id: pluck_rationale
    tool: identity
    args:
      value: "{{assessment.rationale}}"
    capture: rationale
failure_modes:
  - condition: "seed file missing"
    handler: "emit_seed_missing"
  - condition: "introspect_alignment returns malformed result"
    handler: retry
    max_retries: 1
---

# project_introspect

**Quality bar**: `on_track=true` with `drift_observations` containing
high-severity items is a contradiction; the verifier should reject.
The whole point of this skill is to be willing to say "we're off track" —
if it never says that, it's useless.
