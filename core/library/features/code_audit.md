---
name: code_audit
display_name: "Code Audit"
short_description: "Find issues in a codebase via a structured severity ledger."
long_description: |
  Audits a codebase across configurable lenses (security, complexity,
  test coverage, dependencies, documentation). Produces a structured
  ledger — one row per finding — with severity, file:line evidence, and
  a suggested fix, filtered by a severity threshold.
parameters:
  - name: repo_path
    type: string
    required: true
    placeholder: "/Users/me/projects/my_app"
  - name: scope
    type: list[string]
    default: ["security", "complexity", "test_coverage"]
    constraint:
      enum_each: ["security", "complexity", "test_coverage", "dependencies", "documentation"]
  - name: severity_threshold
    type: string
    default: "medium"
    constraint:
      enum: ["low", "medium", "high"]
mission_template: |
  Audit the codebase at {{repo_path}} for issues across these lenses:
  {{scope | comma}}. Produce a structured ledger (one row per finding)
  with severity, evidence (file:line), and suggested fix. Filter by
  severity threshold: {{severity_threshold}}.
roster_override: ["security_auditor", "code_reader", "refactor_specialist", "reviewer"]
skill_plan:
  - root_cause_inference
  - ledger_row_authoring
  - red_team_pass
  - adversarial_audit
  - finalize_project
quality_contract:
  correctness: 5
  completeness: 5
  provenance: 5
  defensibility: 5
  usability: 4
  honesty: 5
  reproducibility: 4
  efficiency: 3
  pass_threshold: 0.75
fitness_command: |
  test -s "{{output_path}}"
  && grep -cE '^\| .* \| .* \| .* \|' "{{output_path}}" | grep -qE '^[3-9]|^[1-9][0-9]'
output_shape: "lab/finalized/audit_{{lab_id}}.md"
estimated_cost_usd: 0.08
estimated_time_minutes: 12
typical_acceptance_rate: null
mcp_tool_signature:
  name: bert.code_audit
  description: "Audit a codebase; structured severity ledger output."
  returns_schema: code_audit_result.v1.json
---
# Code Audit

Audits a codebase and emits a structured severity ledger. The roster
(`security_auditor`, `code_reader`, `refactor_specialist`, `reviewer`)
reads the tree directly via Read/Grep/Glob tools; the `skill_plan`
covers the cross-cutting workflows: `root_cause_inference` (diagnose
each issue's root cause), `ledger_row_authoring` (one row per finding),
`red_team_pass` + `adversarial_audit` (hostile review of the findings),
then `finalize_project`.

> Skill-plan note: the v1.0 implementation spec listed `corpus_discover`
> + `code_search_in_repo` here, but those skills are not in the Sprint-2
> seed registry (the code roster handles repo traversal via tools, not
> skills). This plan uses the equivalent existing skills so every ref
> resolves. Revisit if/when corpus-discovery skills are synthesized.
