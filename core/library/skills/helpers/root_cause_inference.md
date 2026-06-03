---
name: root_cause_inference
version: "1.0"
description: |
  Given a symptom (failing test, dropping quality score, unaccepted
  artifact) and an event timeline, infer the most likely root cause
  using "what changed?" + "what broke first?" + "5 whys" chains.
  Output is a ranked list of hypotheses with evidence weight.
trigger_when: |
  Investigator role debugging an incident; OR self_improvement
  detected a metric regression and wants the WHY before the FIX.
inputs:
  symptom:      {type: string, required: true}
  timeline:     {type: list,   required: true, description: "From event_log_walk.events"}
  recent_changes: {type: list, default: []}
outputs:
  hypotheses:   {type: list, description: "[{cause, evidence, confidence, refutation_test}]"}
  top_cause:    {type: string}
  fix_suggestions: {type: list}
tools_required: [infer_root_cause]
steps:
  - id: infer
    tool: infer_root_cause
    args:
      symptom: "{{symptom}}"
      timeline: "{{timeline}}"
      recent_changes: "{{recent_changes}}"
    capture: result
  - id: pluck_hypotheses
    tool: identity
    args:
      value: "{{result.hypotheses}}"
    capture: hypotheses
  - id: pluck_top
    tool: identity
    args:
      value: "{{result.top_cause}}"
    capture: top_cause
  - id: pluck_fixes
    tool: identity
    args:
      value: "{{result.fix_suggestions}}"
    capture: fix_suggestions
failure_modes:
  - condition: "infer_root_cause returns no hypotheses"
    handler: retry
    max_retries: 1
---

# root_cause_inference

**Quality bar**: every hypothesis must come with a `refutation_test`
— a concrete observation that would falsify it. Hypotheses without
falsifiers are vibes.
