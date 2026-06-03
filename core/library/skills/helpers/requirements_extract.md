---
name: requirements_extract
version: "1.0"
description: |
  Read a spec / brief / RFC / customer-email and extract a clean
  list of testable requirements with type tagging (functional,
  non-functional, constraint, out-of-scope). Output is structured
  for downstream consumption by test_driven_implement.
trigger_when: |
  Engineer role receives an unstructured spec and needs an
  unambiguous task list before writing code.
inputs:
  source_text:  {type: string, default: ""}
  source_path:  {type: string, default: ""}
  audience_hint: {type: string, default: "engineer"}
outputs:
  requirements: {type: list, description: "[{id, text, type, testable: bool, priority}]"}
  ambiguities:  {type: list, description: "[{requirement_id, question}] — items needing clarification"}
  out_of_scope: {type: list}
tools_required: [Read, extract_requirements]
steps:
  - id: load
    if: "source_path != ''"
    tool: Read
    args:
      file_path: "{{source_path}}"
    capture: loaded_text
  - id: extract
    tool: extract_requirements
    args:
      text: "{{loaded_text or source_text}}"
      audience: "{{audience_hint}}"
    capture: extracted
  - id: pluck_requirements
    tool: identity
    args:
      value: "{{extracted.requirements}}"
    capture: requirements
  - id: pluck_ambiguities
    tool: identity
    args:
      value: "{{extracted.ambiguities}}"
    capture: ambiguities
  - id: pluck_scope
    tool: identity
    args:
      value: "{{extracted.out_of_scope}}"
    capture: out_of_scope
failure_modes:
  - condition: "Source text + path both empty"
    handler: "emit_no_source"
---

# requirements_extract

**Quality bar**: every requirement must be `testable: true` or flagged
as ambiguous. A non-testable requirement is a wish, not a requirement.
