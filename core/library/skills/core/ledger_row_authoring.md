---
name: ledger_row_authoring
version: "1.0"
description: |
  Append a single structured row to the lab's source-of-record ledger
  (`lab/sor/events.jsonl` by convention). Validates schema against
  required fields, atomically appends, and returns the offset of the
  new row so callers can reference it as immutable provenance.
trigger_when: |
  Any time an artifact is accepted, a cycle is closed, or a decision
  is finalized. The ledger is bert's immutable audit trail — every
  important event should leave a row.
inputs:
  event_type:    {type: string, required: true, description: "e.g., 'artifact_accepted', 'cycle_close', 'decision'"}
  cycle_id:      {type: int,    required: true}
  agent:         {type: string, default: "system"}
  payload:       {type: dict,   default: {},     description: "Event-specific structured data"}
  ledger_path:   {type: string, default: "lab/sor/events.jsonl"}
outputs:
  row_offset:    {type: int,    description: "Byte offset of the new row in the ledger"}
  row_id:        {type: string, description: "Stable ID derived from {cycle, type, hash(payload)}"}
  appended_at:   {type: string, description: "ISO-8601 timestamp"}
tools_required: [validate_ledger_row, append_jsonl_atomic]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: validate
    tool: validate_ledger_row
    args:
      event_type: "{{event_type}}"
      cycle_id: "{{cycle_id}}"
      agent: "{{agent}}"
      payload: "{{payload}}"
    capture: validation
  - id: append
    tool: append_jsonl_atomic
    args:
      path: "{{ledger_path}}"
      row:
        event_type: "{{event_type}}"
        cycle_id: "{{cycle_id}}"
        agent: "{{agent}}"
        payload: "{{payload}}"
    capture: append_result
  - id: pluck_offset
    tool: identity
    args:
      value: "{{append_result.offset}}"
    capture: row_offset
  - id: pluck_id
    tool: identity
    args:
      value: "{{append_result.row_id}}"
    capture: row_id
  - id: pluck_ts
    tool: identity
    args:
      value: "{{append_result.appended_at}}"
    capture: appended_at
failure_modes:
  - condition: "Schema validation fails"
    handler: "emit_invalid_schema"
  - condition: "Atomic append fails (disk full, permission)"
    handler: retry
    max_retries: 2
---

# ledger_row_authoring

**Quality bar**: every successful artifact_accepted MUST have a
ledger row before the next cycle starts. If the row failed to write,
the cycle is not actually closed.
