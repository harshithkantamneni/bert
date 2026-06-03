---
name: red_team_pass
version: "1.0"
description: |
  Adversarial audit of a finding or claim. Generates a structured
  attack: what assumptions could be wrong, what evidence would
  falsify, what alternative interpretations exist, what selection
  bias might be in play. Output is a list of counter-claims with
  rebuttable specificity — never just "you might be wrong".
trigger_when: |
  A claim is about to be promoted to a high-trust surface (proof
  packet, decision memo, public artifact) AND it doesn't already
  have an adversarial review in its lineage.
inputs:
  finding:   {type: string, required: true, description: "The claim or conclusion under attack"}
  context:   {type: string, default: "",    description: "Optional: the reasoning chain that led to the finding"}
  evidence:  {type: list,   default: [],    description: "Optional: [{quote, source}] backing the finding"}
  depth:     {type: string, default: "standard", description: "quick | standard | thorough — controls how many attack vectors to generate"}
outputs:
  attacks:        {type: list,   description: "[{vector, counter_claim, falsifier, severity: low|medium|high}]"}
  highest_severity: {type: string, description: "Worst attack severity found"}
  recommendation: {type: string, description: "accept | revise | retract"}
tools_required: [generate_attacks]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: enumerate_attacks
    tool: generate_attacks
    args:
      finding: "{{finding}}"
      context: "{{context}}"
      evidence: "{{evidence}}"
      depth: "{{depth}}"
    capture: attack_output
  - id: rank_severity
    tool: rank_attack_severity
    args:
      attacks: "{{attack_output.attacks}}"
    capture: ranked_attacks
  - id: recommend
    tool: red_team_recommendation
    args:
      attacks: "{{ranked_attacks.attacks}}"
      evidence: "{{evidence}}"
    capture: recommendation_out
  - id: pluck_attacks
    tool: identity
    args:
      value: "{{ranked_attacks.attacks}}"
    capture: attacks
  - id: pluck_highest
    tool: identity
    args:
      value: "{{ranked_attacks.highest_severity}}"
    capture: highest_severity
  - id: pluck_recommendation
    tool: identity
    args:
      value: "{{recommendation_out.recommendation}}"
    capture: recommendation
failure_modes:
  - condition: "generate_attacks returns < 3 distinct vectors"
    handler: retry
    max_retries: 1
  - condition: "All attacks are 'low' severity"
    handler: "emit_insufficient_adversarial_depth"
---

# red_team_pass

**Quality bar**: an "accept" recommendation is meaningful only when at
least one attack of medium+ severity was attempted and survived. A
red-team pass that finds nothing is a pass that didn't try hard enough.
