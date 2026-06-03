---
name: finalize_project
version: "1.0"
description: |
  Orchestrates the closing of a project: gathers all accumulated
  evidence, synthesizes a polished final artifact, honestly discloses
  what couldn't be answered (gaps.md), and grades + signs the result
  for the proof packet. This is the highest-stakes skill in the
  registry — it's what produces the deliverable.

  Composition: this skill calls four sub-skills in strict order.
  Failure in any sub-skill blocks the entire finalize; the cycle
  ends in 'unresolved' rather than producing a misleading artifact.
trigger_when: |
  Lab director declares ready_to_finalize=true (typically after a
  gap_finder run that returned coverage_pct >= 0.8 and no high-
  severity gaps).
inputs:
  objective:       {type: string, required: true}
  findings_dir:    {type: string, default: "findings/"}
  output_path:     {type: string, required: true, description: "Where the polished artifact goes"}
  ledger_path:     {type: string, default: "lab/sor/events.jsonl"}
  quality_contract: {type: dict,  required: false, description: "Mission QualityContract; auto-injected from the lab schema by skill_runner, or omitted -> balanced default"}
outputs:
  artifact_path:   {type: string, description: "Path to the published artifact"}
  gaps_path:       {type: string, description: "Path to gaps.md (honest disclosure, per project_bert_proof_packet)"}
  grade:           {type: string, description: "A | B | C | D | F"}
  signed_hash:     {type: string, description: "SHA-256 of (artifact + gaps + grade) — proof-packet anchor"}
  ready:           {type: bool,   description: "True iff grade >= B and gaps.md exists and ledger row was written"}
tools_required: []
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: gather
    skill: gather_project_evidence
    args:
      findings_dir: "{{findings_dir}}"
      objective: "{{objective}}"
    capture: evidence_bundle
  - id: synthesize
    skill: synthesize_polished_artifact
    args:
      evidence: "{{evidence_bundle.evidence}}"
      objective: "{{objective}}"
      output_path: "{{output_path}}"
    capture: synth_result
  - id: disclose
    skill: disclose_honest_gaps
    args:
      evidence: "{{evidence_bundle.evidence}}"
      artifact_path: "{{synth_result.artifact_path}}"
      objective: "{{objective}}"
    capture: gaps_result
  - id: sign
    skill: grade_and_sign
    args:
      artifact_path: "{{synth_result.artifact_path}}"
      gaps_path: "{{gaps_result.gaps_path}}"
      evidence_count: "{{evidence_bundle.count}}"
      quality_contract: "{{quality_contract}}"
    capture: sign_result
  - id: record_in_ledger
    skill: ledger_row_authoring
    args:
      event_type: "artifact_accepted"
      cycle_id: 0
      agent: "finalize_project"
      payload:
        artifact_path: "{{synth_result.artifact_path}}"
        gaps_path: "{{gaps_result.gaps_path}}"
        grade: "{{sign_result.grade}}"
        signed_hash: "{{sign_result.signed_hash}}"
      ledger_path: "{{ledger_path}}"
    capture: ledger_entry
  - id: pluck_artifact
    tool: identity
    args:
      value: "{{synth_result.artifact_path}}"
    capture: artifact_path
  - id: pluck_gaps
    tool: identity
    args:
      value: "{{gaps_result.gaps_path}}"
    capture: gaps_path
  - id: pluck_grade
    tool: identity
    args:
      value: "{{sign_result.grade}}"
    capture: grade
  - id: pluck_hash
    tool: identity
    args:
      value: "{{sign_result.signed_hash}}"
    capture: signed_hash
  - id: compute_ready
    tool: finalize_ready_check
    args:
      grade: "{{sign_result.grade}}"
      gaps_path: "{{gaps_result.gaps_path}}"
      ledger_row_id: "{{ledger_entry.row_id}}"
    capture: ready
failure_modes:
  - condition: "Any sub-skill fails"
    handler: "emit_unresolved"
  - condition: "Grade is D or F"
    handler: "emit_below_threshold"
---

# finalize_project

**Quality bar**: this skill produces investor-grade output. If you can't
ship at grade A or B, ship `emit_unresolved` and explain — never ship
a misleading polished artifact. The proof packet's integrity depends on
this skill being honest.

**Composition order**: gather → synthesize → disclose → sign → ledger.
Reordering breaks the audit trail.
