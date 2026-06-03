---
name: gather_project_evidence
version: "1.0"
description: |
  First step of finalize_project: walk the findings directory and the
  source-of-record ledger to assemble a deduped evidence bundle ready
  for synthesis. Each evidence item carries provenance (which cycle,
  which agent, which artifact, which line range) so the writer can
  cite without hallucinating.
trigger_when: |
  Called by finalize_project as step 1. Not typically called
  standalone — too narrow. If you want a quick corpus snapshot,
  use brief_assembler instead.
inputs:
  findings_dir: {type: string, default: "findings/"}
  ledger_path:  {type: string, default: "lab/sor/events.jsonl"}
  objective:    {type: string, required: true, description: "Used to filter relevant evidence"}
  min_quality:  {type: float,  default: 0.5,   description: "Minimum quality_score to include"}
outputs:
  evidence:     {type: list, description: "[{type, source_path, content, provenance, quality_score}]"}
  count:        {type: int}
  cycles_covered: {type: list, description: "Set of cycle IDs represented in evidence"}
tools_required: [list_findings, read_ledger_rows]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: list_findings_files
    tool: list_findings
    args:
      dir: "{{findings_dir}}"
      min_quality: "{{min_quality}}"
    capture: findings_list
  - id: read_ledger
    tool: read_ledger_rows
    args:
      path: "{{ledger_path}}"
      event_types: ["artifact_accepted", "contradiction_reconciled", "decision"]
    capture: ledger_rows
  - id: assemble
    tool: assemble_evidence_bundle
    args:
      findings: "{{findings_list.files}}"
      ledger_rows: "{{ledger_rows.rows}}"
      objective: "{{objective}}"
    capture: bundle
  - id: pluck_evidence
    tool: identity
    args:
      value: "{{bundle.evidence}}"
    capture: evidence
  - id: pluck_count
    tool: identity
    args:
      value: "{{bundle.count}}"
    capture: count
  - id: pluck_cycles
    tool: identity
    args:
      value: "{{bundle.cycles_covered}}"
    capture: cycles_covered
failure_modes:
  - condition: "Empty findings + empty ledger"
    handler: "emit_no_evidence_to_gather"
  - condition: "All findings below min_quality threshold"
    handler: "emit_quality_floor_unmet"
---

# gather_project_evidence

**Quality bar**: empty evidence bundle → finalize_project MUST stop;
shipping a polished artifact from no evidence is the worst possible
behavior. Caller is responsible for catching emit_* and refusing to
proceed.
