---
name: comparative_evaluation
version: "1.0"
description: |
  Score N candidate options against a set of weighted criteria and
  return a ranked list with reasoning. Forces apples-to-apples
  judgment: each option gets a score per criterion, weights are
  declared up front, and the winner has to clear a margin threshold
  or the verdict is "no clear winner — needs more evidence".
trigger_when: |
  Decision-making role faces multiple viable options and needs a
  defensible ranking — model choice, implementation strategy,
  vendor comparison, design alternative.
inputs:
  options:    {type: list, required: true, description: "[{name, summary}, ...]"}
  criteria:   {type: list, required: true, description: "[{name, weight: 0.0–1.0, definition}, ...] — weights should sum to ~1.0"}
  evidence:   {type: list, default: [], description: "[{option, criterion, snippet, source}] — optional supporting evidence"}
  margin:     {type: float, default: 0.10, description: "Min score margin between #1 and #2 to declare a winner"}
outputs:
  ranked:     {type: list,   description: "[{rank, name, total_score, per_criterion: {...}, justification}]"}
  winner:     {type: string, description: "Name of #1 option, or 'no_clear_winner' if margin below threshold"}
  margin_obs: {type: float,  description: "Observed margin between #1 and #2"}
tools_required: [score_options]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: validate_weights
    tool: weights_sum_check
    args:
      criteria: "{{criteria}}"
      tolerance: 0.05
    capture: weight_check
  - id: score
    tool: score_options
    args:
      options: "{{options}}"
      criteria: "{{criteria}}"
      evidence: "{{evidence}}"
    capture: scored
  - id: rank
    tool: rank_by_total
    args:
      scored: "{{scored.scored_options}}"
      margin: "{{margin}}"
    capture: rank_result
  - id: pluck_ranked
    tool: identity
    args:
      value: "{{rank_result.ranked}}"
    capture: ranked
  - id: pluck_winner
    tool: identity
    args:
      value: "{{rank_result.winner}}"
    capture: winner
  - id: pluck_margin
    tool: identity
    args:
      value: "{{rank_result.margin_obs}}"
    capture: margin_obs
failure_modes:
  - condition: "Weights sum outside tolerance"
    handler: "emit_invalid_weights"
  - condition: "score_options returns wrong cardinality"
    handler: retry
    max_retries: 1
---

# comparative_evaluation

**Quality bar**: never declare a winner with margin < 0.05. If you do,
the next reviewer will rightly call this evaluation theater.
