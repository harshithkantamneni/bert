---
name: roadmap_from_findings
version: "1.0"
description: |
  Convert a body of findings + gaps + risks into a sequenced roadmap
  of next-cycle missions. Output is a director-ready plan: what to
  attempt next, why, in what order, and what we expect to learn.
trigger_when: |
  At end of a "discovery" phase, before committing to a next
  "execution" phase. The skill bridges research into action.
inputs:
  findings:     {type: list, default: []}
  gaps:         {type: list, default: []}
  risks:        {type: list, default: []}
  budget_cycles: {type: int, default: 5, description: "How many cycles can fit in the plan"}
outputs:
  missions:     {type: list, description: "[{order, mission, hypothesis_to_test, expected_artifact, dependencies}]"}
  expected_value: {type: float, description: "Total expected learning value (sum of mission scores)"}
tools_required: [draft_roadmap]
steps:
  - id: draft
    tool: draft_roadmap
    args:
      findings: "{{findings}}"
      gaps: "{{gaps}}"
      risks: "{{risks}}"
      budget: "{{budget_cycles}}"
    capture: drafted
  - id: pluck_missions
    tool: identity
    args:
      value: "{{drafted.missions}}"
    capture: missions
  - id: pluck_value
    tool: identity
    args:
      value: "{{drafted.expected_value}}"
    capture: expected_value
failure_modes:
  - condition: "Findings empty AND gaps empty"
    handler: "emit_nothing_to_roadmap"
---

# roadmap_from_findings

**Quality bar**: every mission must have a falsifiable
`hypothesis_to_test`. Missions stated as "explore X" without a
hypothesis are vacation, not work.
