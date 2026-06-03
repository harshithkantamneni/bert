---
name: key_decision_extract
version: "1.0"
description: |
  Scan recent cycles' artifacts + ledger for decisions that altered
  trajectory (e.g., "we will use Provider X", "scope reduced to Y").
  Returns a structured decision log with rationale + reversibility.
trigger_when: |
  Director or strategy role needs to recall lab's current decisions
  before approving a new one (avoid re-deciding settled questions).
inputs:
  ledger_path:  {type: string, default: "lab/sor/events.jsonl"}
  recent_n_cycles: {type: int, default: 20}
outputs:
  decisions:    {type: list, description: "[{id, cycle, statement, rationale, reversibility}]"}
  by_topic:     {type: dict}
tools_required: [read_ledger_rows, classify_decisions]
steps:
  - id: read_decisions
    tool: read_ledger_rows
    args:
      path: "{{ledger_path}}"
      event_types: ["decision"]
    capture: rows
  - id: classify
    tool: classify_decisions
    args:
      rows: "{{rows.rows}}"
      recent_n: "{{recent_n_cycles}}"
    capture: classified
  - id: pluck_decisions
    tool: identity
    args:
      value: "{{classified.decisions}}"
    capture: decisions
  - id: pluck_by_topic
    tool: identity
    args:
      value: "{{classified.by_topic}}"
    capture: by_topic
failure_modes:
  - condition: "Ledger missing"
    handler: "emit_ledger_missing"
---

# key_decision_extract

**Quality bar**: decisions without `rationale` should be flagged
'orphaned' — they are a bug, not a record.
