---
name: synthesize_polished_artifact
version: "1.0"
description: |
  Second step of finalize_project: given a deduped evidence bundle
  and the objective, write a structured, citation-rich final artifact
  to `output_path`. Output is investor-grade prose with every claim
  traceable back to an evidence item via inline footnote IDs.
trigger_when: |
  Called by finalize_project as step 2 (after gather_project_evidence).
inputs:
  evidence:       {type: list,   required: true, description: "From gather_project_evidence.evidence"}
  objective:      {type: string, required: true}
  output_path:    {type: string, required: true}
  target_grade:   {type: string, default: "A",  description: "Quality target — affects depth & polish"}
  max_words:      {type: int,    default: 4000, description: "Soft cap; structure preserved over verbosity"}
outputs:
  artifact_path:  {type: string, description: "Path to the written artifact"}
  word_count:     {type: int}
  citations_used: {type: int,    description: "How many evidence items cited inline"}
  uncited_evidence: {type: list, description: "Evidence not used — flagged for gap analysis"}
tools_required: [synthesize_artifact_body, Write]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: synthesize
    tool: synthesize_artifact_body
    args:
      evidence: "{{evidence}}"
      objective: "{{objective}}"
      target_grade: "{{target_grade}}"
      max_words: "{{max_words}}"
    capture: synth_result
  - id: write_artifact
    tool: Write
    args:
      file_path: "{{output_path}}"
      content: "{{synth_result.body}}"
    capture: write_result
  - id: pluck_path
    tool: identity
    args:
      value: "{{output_path}}"
    capture: artifact_path
  - id: pluck_wc
    tool: identity
    args:
      value: "{{synth_result.word_count}}"
    capture: word_count
  - id: pluck_cites
    tool: identity
    args:
      value: "{{synth_result.citations_used}}"
    capture: citations_used
  - id: pluck_uncited
    tool: identity
    args:
      value: "{{synth_result.uncited_evidence}}"
    capture: uncited_evidence
failure_modes:
  - condition: "Evidence empty"
    handler: "emit_no_evidence"
  - condition: "synthesize_artifact_body produces < 20% citation density"
    handler: retry
    max_retries: 1
---

# synthesize_polished_artifact

**Quality bar**: minimum citation density is 1 inline citation per
~200 words. Below that, the artifact is essentially opinion masquerading
as research; the proof packet must reject it.
