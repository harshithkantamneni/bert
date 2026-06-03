---
name: gap_finder
version: "1.0"
description: |
  Given a corpus of findings + an explicit objective, identify what's
  missing — unaddressed sub-questions, unverified claims, missing
  baselines, blind spots in source diversity. Output is a structured
  gap report that the next cycle can address as concrete tasks.
trigger_when: |
  After N cycles of accumulation, before declaring a research project
  "done". Also useful before finalize_project to ensure the artifact
  isn't lying-by-omission.
inputs:
  objective:    {type: string, required: true, description: "The mission's north-star question"}
  findings:     {type: list,   required: true, description: "[{cycle, agent, summary, evidence}] — corpus to audit"}
  required_dimensions: {type: list, default: [], description: "Pre-declared coverage axes (e.g., ['quant_evidence', 'recent_2025+', 'opposing_view'])"}
outputs:
  gaps:         {type: list,   description: "[{type, description, severity: low|medium|high, suggested_action}]"}
  coverage_pct: {type: float,  description: "0.0–1.0 — fraction of required_dimensions hit"}
  ready_to_finalize: {type: bool, description: "True iff coverage_pct >= 0.8 AND no 'high' severity gaps"}
tools_required: [analyze_gaps]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: audit_corpus
    tool: analyze_gaps
    args:
      objective: "{{objective}}"
      findings: "{{findings}}"
      required_dimensions: "{{required_dimensions}}"
    capture: audit_result
  - id: compute_readiness
    tool: assess_finalize_readiness
    args:
      gaps: "{{audit_result.gaps}}"
      coverage_pct: "{{audit_result.coverage_pct}}"
    capture: readiness
  - id: pluck_gaps
    tool: identity
    args:
      value: "{{audit_result.gaps}}"
    capture: gaps
  - id: pluck_coverage
    tool: identity
    args:
      value: "{{audit_result.coverage_pct}}"
    capture: coverage_pct
  - id: pluck_ready
    tool: identity
    args:
      value: "{{readiness.ready}}"
    capture: ready_to_finalize
failure_modes:
  - condition: "Empty findings corpus"
    handler: "emit_no_corpus"
  - condition: "analyze_gaps returns no gaps with required_dimensions present"
    handler: retry
    max_retries: 1
---

# gap_finder

**Quality bar**: `ready_to_finalize=true` with `coverage_pct < 0.8`
is a contradiction; the verifier should reject. Similarly, a corpus
of 20 findings that produced zero gaps probably wasn't audited at all.
