---
name: structure_outline
version: "1.0"
description: |
  Given a topic and gathered evidence, produce a hierarchical
  outline (sections → subsections → paragraph stubs) that a writer
  role can fill in. Used as a pre-step to synthesize_polished_artifact
  when the artifact is long enough that structure-before-prose
  prevents narrative drift.
trigger_when: |
  Writer is about to produce an artifact > 1000 words, OR the
  artifact has > 4 top-level sections, OR cross-cutting themes
  across many evidence items need ordering.
inputs:
  objective:    {type: string, required: true}
  evidence:     {type: list,   default: []}
  depth:        {type: int,    default: 3, description: "Heading levels"}
outputs:
  outline:      {type: list, description: "[{level, heading, paragraph_stub, evidence_refs}]"}
  word_target:  {type: int}
tools_required: [draft_outline]
steps:
  - id: draft
    tool: draft_outline
    args:
      objective: "{{objective}}"
      evidence: "{{evidence}}"
      depth: "{{depth}}"
    capture: drafted
  - id: pluck_outline
    tool: identity
    args:
      value: "{{drafted.outline}}"
    capture: outline
  - id: pluck_target
    tool: identity
    args:
      value: "{{drafted.word_target}}"
    capture: word_target
failure_modes:
  - condition: "draft_outline returns < 2 top-level sections"
    handler: retry
    max_retries: 1
---

# structure_outline

**Quality bar**: an outline with one top-level section is not an
outline — it's a paragraph. Force breadth.
