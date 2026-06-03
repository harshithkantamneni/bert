---
name: grade_and_sign
version: "1.0"
description: |
  Fourth step of finalize_project: assign a letter grade (A-F) to
  the artifact based on evidence quality, citation density, gap
  honesty, and adversarial-audit survival; then compute a
  SHA-256 over (artifact + gaps + grade_envelope) for the proof
  packet's signed_hash field. Final output is the anchor that
  Sigstore + SLSA in-toto will attest over.
trigger_when: |
  Called by finalize_project as step 4 (after disclose_honest_gaps).
  Never skip — without a signed hash, the proof packet has nothing
  to verify against.
inputs:
  artifact_path:   {type: string, required: true}
  gaps_path:       {type: string, required: true}
  evidence_count:  {type: int,    required: true}
  rubric_path:     {type: string, default: "core/library/grading_rubric.yaml"}
  quality_contract: {type: dict,  required: false, description: "Mission QualityContract (8 weights + pass_threshold); omitted -> balanced default"}
outputs:
  grade:           {type: string, description: "A | B | C | D | F"}
  grade_components: {type: dict,  description: "{evidence_q: 0-1, citation_density: 0-1, gap_honesty: 0-1, adversarial_survival: 0-1}"}
  signed_hash:     {type: string, description: "SHA-256 hex of (artifact || gaps || envelope_json)"}
  envelope:        {type: dict,   description: "The JSON envelope that was hashed — included for verifier debugging"}
tools_required: [Read, evaluate_artifact_rubric, sha256_envelope]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: read_artifact
    tool: Read
    args:
      file_path: "{{artifact_path}}"
    capture: artifact_text
  - id: read_gaps
    tool: Read
    args:
      file_path: "{{gaps_path}}"
    capture: gaps_text
  - id: evaluate
    tool: evaluate_artifact_rubric
    args:
      artifact: "{{artifact_text}}"
      gaps: "{{gaps_text}}"
      evidence_count: "{{evidence_count}}"
      rubric_path: "{{rubric_path}}"
      contract: "{{quality_contract}}"
    capture: eval_result
  - id: sign
    tool: sha256_envelope
    args:
      artifact: "{{artifact_text}}"
      gaps: "{{gaps_text}}"
      grade: "{{eval_result.grade}}"
      components: "{{eval_result.components}}"
    capture: sign_result
  - id: pluck_grade
    tool: identity
    args:
      value: "{{eval_result.grade}}"
    capture: grade
  - id: pluck_components
    tool: identity
    args:
      value: "{{eval_result.components}}"
    capture: grade_components
  - id: pluck_hash
    tool: identity
    args:
      value: "{{sign_result.hash}}"
    capture: signed_hash
  - id: pluck_envelope
    tool: identity
    args:
      value: "{{sign_result.envelope}}"
    capture: envelope
failure_modes:
  - condition: "Rubric file missing"
    handler: "emit_rubric_missing"
  - condition: "evaluate_artifact_rubric returns score with all 1.0s"
    handler: retry
    max_retries: 1
---

# grade_and_sign

**Quality bar**: all-perfect grade components is essentially never real;
that's a hallucinating evaluator. The hash must always be over the
final canonical text — if artifact_path is rewritten after signing,
the hash is invalid by design.
