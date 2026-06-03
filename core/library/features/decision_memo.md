---
name: decision_memo
display_name: "Decision Memo"
short_description: "Structured recommendation on a question with red team."
long_description: |
  Evaluates a decision question against named options + criteria.
  Produces a memo with recommendation, rationale, alternatives
  considered, and explicit falsification conditions.
parameters:
  - name: question
    type: string
    required: true
    placeholder: "Should we adopt X for our Y use case?"
    validate:
      min_words: 3
      max_words: 30
  - name: options
    type: list[string]
    min_items: 2
    max_items: 8
  - name: criteria
    type: list[string]
    min_items: 1
    max_items: 10
mission_template: |
  Make a decision on: {{question}}.
  Options to evaluate: {{options | comma}}.
  Criteria: {{criteria | comma}}.
  Produce a decision memo with: recommendation, rationale,
  alternatives_considered, falsification_conditions.
roster_override: ["analyst", "option_scorer", "red_team", "writer"]
skill_plan:
  - web_search_and_dedup
  - comparative_evaluation
  - red_team_pass
  - decision_memo_draft
  - finalize_project
quality_contract:
  correctness: 5
  completeness: 4
  provenance: 5
  defensibility: 5
  usability: 4
  honesty: 5
  reproducibility: 3
  efficiency: 3
  pass_threshold: 0.75
fitness_command: |
  test -s "{{output_path}}"
  && grep -q '^## Recommendation' "{{output_path}}"
  && grep -q '^## Falsification' "{{output_path}}"
output_shape: "lab/finalized/decision_{{lab_id}}.md"
estimated_cost_usd: 0.04
estimated_time_minutes: 6
typical_acceptance_rate: null
mcp_tool_signature:
  name: bert.decision_memo
  description: "Structured recommendation on a decision with red team."
  returns_schema: decision_memo_result.v1.json
---
# Decision Memo

Produces a structured recommendation on a question. The `red_team_pass`
+ `decision_memo_draft` skills force an alternatives-considered section
and explicit falsification conditions, so the memo discloses what would
change the recommendation — not just the recommendation itself.
