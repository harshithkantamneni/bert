---
name: lessons_extract
version: "1.0"
description: |
  After a cycle (or sprint) closes, walk what happened — the wins,
  the misses, the surprises — and extract durable lessons for the
  next cycle. Output goes into memory for retrieval by future cycles.
trigger_when: |
  At cycle close, or sprint retro. Called by lab_director or
  self_improvement.
inputs:
  cycle_id:     {type: int,    required: true}
  events:       {type: list,   required: true, description: "From event_log_walk filtered to this cycle"}
  acceptance_outcome: {type: string, required: true, description: "accepted | rejected | unresolved"}
outputs:
  lessons:      {type: list, description: "[{lesson, evidence, applicability_scope}]"}
  durable_count: {type: int,  description: "How many lessons rated 'high transferability'"}
tools_required: [extract_lessons, memory_create]
steps:
  - id: extract
    tool: extract_lessons
    args:
      cycle_id: "{{cycle_id}}"
      events: "{{events}}"
      outcome: "{{acceptance_outcome}}"
    capture: extracted
  - id: persist
    foreach: "extracted.lessons"
    tool: memory_create
    args:
      path: "lab/memory/lessons/cycle_{{cycle_id}}.md"
      content: "{{item.lesson}}"
    capture: persist_results
  - id: pluck_lessons
    tool: identity
    args:
      value: "{{extracted.lessons}}"
    capture: lessons
  - id: pluck_count
    tool: identity
    args:
      value: "{{extracted.durable_count}}"
    capture: durable_count
failure_modes:
  - condition: "extract_lessons returns 0 lessons for non-trivial cycle"
    handler: retry
    max_retries: 1
---

# lessons_extract

**Quality bar**: at least one lesson should target a *systemic* (not
single-incident) issue, otherwise the cycle is repeating mistakes.
