---
name: event_log_walk
version: "1.0"
description: |
  Walk the lab's source-of-record ledger (events.jsonl) within a
  cycle range and return a structured timeline of what happened.
  Used by analyst, methodology_critic, and self_improvement to
  reason about cycle-over-cycle behavior.
trigger_when: |
  Any role needs to reason about lab history, NOT current state.
  Skip if you just need the latest finding — use memory_search instead.
inputs:
  ledger_path:  {type: string, default: "lab/sor/events.jsonl"}
  from_cycle:   {type: int,    default: 0}
  to_cycle:     {type: int,    default: 999999}
  event_filter: {type: list,   default: []}
outputs:
  events:       {type: list, description: "[{cycle_id, event_type, agent, payload, ts}]"}
  by_cycle:     {type: dict, description: "{cycle_id: [events]}"}
  count:        {type: int}
tools_required: [read_ledger_rows]
steps:
  - id: read
    tool: read_ledger_rows
    args:
      path: "{{ledger_path}}"
      from_cycle: "{{from_cycle}}"
      to_cycle: "{{to_cycle}}"
      event_types: "{{event_filter}}"
    capture: walk
  - id: pluck_events
    tool: identity
    args:
      value: "{{walk.rows}}"
    capture: events
  - id: pluck_by_cycle
    tool: identity
    args:
      value: "{{walk.by_cycle}}"
    capture: by_cycle
  - id: pluck_count
    tool: identity
    args:
      value: "{{walk.count}}"
    capture: count
failure_modes:
  - condition: "Ledger file missing"
    handler: "emit_ledger_missing"
---

# event_log_walk

**Quality bar**: this skill MUST be read-only on the ledger. Never
let it transitively call ledger_row_authoring or any writer.
