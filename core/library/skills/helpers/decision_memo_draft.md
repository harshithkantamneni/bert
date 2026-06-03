---
name: decision_memo_draft
version: "1.0"
description: |
  Draft a 1-page decision memo: context → options → recommendation
  → rationale → risks → fallback. Used when a decision is too
  consequential to be a one-line ledger row.
trigger_when: |
  Director or strategist needs to commit to a choice that will
  bind multiple downstream cycles. Skip for reversible micro-decisions.
inputs:
  context:      {type: string, required: true}
  options:      {type: list,   required: true, description: "From comparative_evaluation.ranked, ideally"}
  recommendation: {type: string, required: true, description: "Picked option name"}
  rationale:    {type: string, required: true}
  output_path:  {type: string, default: "lab/decisions/memo.md"}
outputs:
  memo_path:    {type: string}
  word_count:   {type: int}
  ledger_row_id: {type: string, description: "ID of the ledger row that records this decision"}
tools_required: [draft_memo, Write]
steps:
  - id: draft
    tool: draft_memo
    args:
      context: "{{context}}"
      options: "{{options}}"
      recommendation: "{{recommendation}}"
      rationale: "{{rationale}}"
    capture: memo
  - id: write
    tool: Write
    args:
      file_path: "{{output_path}}"
      content: "{{memo.body}}"
    capture: write_result
  - id: ledger
    skill: ledger_row_authoring
    args:
      event_type: "decision"
      cycle_id: 0
      agent: "decision_memo_draft"
      payload:
        memo_path: "{{output_path}}"
        recommendation: "{{recommendation}}"
    capture: ledger_row
  - id: pluck_path
    tool: identity
    args:
      value: "{{output_path}}"
    capture: memo_path
  - id: pluck_wc
    tool: identity
    args:
      value: "{{memo.word_count}}"
    capture: word_count
  - id: pluck_row
    tool: identity
    args:
      value: "{{ledger_row.row_id}}"
    capture: ledger_row_id
failure_modes:
  - condition: "draft_memo returns body < 200 words"
    handler: retry
    max_retries: 1
---

# decision_memo_draft

**Quality bar**: a decision memo without a `fallback` section is
not a memo, it's a press release. Force the rationale to acknowledge
what happens if the recommendation turns out wrong.
