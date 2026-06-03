---
name: disclose_honest_gaps
version: "1.0"
description: |
  Third step of finalize_project: produce a `gaps.md` document that
  honestly enumerates what the lab could NOT answer, what assumptions
  remain unverified, where the corpus is thin, and what would change
  the conclusion if discovered. This is the "failures.md" file per
  project_bert_proof_packet_schema — separately signed and shipped
  alongside the polished artifact.
trigger_when: |
  Called by finalize_project as step 3 (after synthesize). Never
  skip — a finalize without gaps.md is by definition lying-by-omission.
inputs:
  evidence:       {type: list,   required: true}
  artifact_path:  {type: string, required: true, description: "Path to the polished artifact, for cross-reference"}
  objective:      {type: string, required: true}
  gaps_output:    {type: string, default: "gaps.md"}
outputs:
  gaps_path:      {type: string, description: "Path to written gaps.md"}
  gap_count:      {type: int}
  unanswered_questions: {type: list, description: "Subset of objective the lab couldn't address"}
  honest_score:   {type: float,  description: "0.0-1.0 — caller can require >0.7 before sign"}
tools_required: [Read, analyze_evidence_holes, Write]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: read_artifact
    tool: Read
    args:
      file_path: "{{artifact_path}}"
    capture: artifact_text
  - id: find_holes
    tool: analyze_evidence_holes
    args:
      evidence: "{{evidence}}"
      artifact: "{{artifact_text}}"
      objective: "{{objective}}"
    capture: holes
  - id: write_gaps
    tool: Write
    args:
      file_path: "{{gaps_output}}"
      content: "{{holes.gaps_md}}"
    capture: write_result
  - id: pluck_path
    tool: identity
    args:
      value: "{{gaps_output}}"
    capture: gaps_path
  - id: pluck_count
    tool: identity
    args:
      value: "{{holes.gap_count}}"
    capture: gap_count
  - id: pluck_unanswered
    tool: identity
    args:
      value: "{{holes.unanswered_questions}}"
    capture: unanswered_questions
  - id: pluck_honest
    tool: identity
    args:
      value: "{{holes.honest_score}}"
    capture: honest_score
failure_modes:
  - condition: "analyze_evidence_holes returns 0 gaps for non-trivial artifact"
    handler: retry
    max_retries: 1
  - condition: "Artifact file unreadable"
    handler: "emit_artifact_unreadable"
---

# disclose_honest_gaps

**Quality bar**: `gap_count == 0` is essentially never true for a
real-world synthesis. If it returns 0, the analyzer was lazy — retry
with deeper prompting. A `gaps.md` claiming "no gaps" is the most
suspicious possible output.
