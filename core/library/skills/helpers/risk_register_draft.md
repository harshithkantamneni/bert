---
name: risk_register_draft
version: "1.0"
description: |
  Generate a structured risk register from current findings + open
  gaps + adversarial-audit attacks. Each risk has likelihood,
  impact, owner-role, and a concrete mitigation.
trigger_when: |
  Before finalize_project on high-stakes artifacts; before investor
  demos; OR when governance_gate explicitly requests risk surface.
inputs:
  findings:     {type: list, default: []}
  gaps:         {type: list, default: []}
  adversarial_attacks: {type: list, default: []}
outputs:
  risks:        {type: list, description: "[{id, description, likelihood: 1-5, impact: 1-5, owner, mitigation}]"}
  high_severity_count: {type: int}
tools_required: [draft_risks]
steps:
  - id: draft
    tool: draft_risks
    args:
      findings: "{{findings}}"
      gaps: "{{gaps}}"
      attacks: "{{adversarial_attacks}}"
    capture: drafted
  - id: pluck_risks
    tool: identity
    args:
      value: "{{drafted.risks}}"
    capture: risks
  - id: pluck_high
    tool: identity
    args:
      value: "{{drafted.high_severity_count}}"
    capture: high_severity_count
failure_modes:
  - condition: "draft_risks returns 0 risks but adversarial_attacks not empty"
    handler: retry
    max_retries: 1
---

# risk_register_draft

**Quality bar**: every risk must have a mitigation. "Risk: X.
Mitigation: be careful" is not a mitigation.
