---
name: refactor_plan
display_name: "Refactor Plan"
short_description: "Phased code refactor proposal with sample diffs."
long_description: |
  Plans a refactor of a code scope toward a stated goal under explicit
  constraints. Output is a phased plan (ordered, parallelizable groups)
  plus sample diffs and a migration up/down pair so the change is
  reversible and test-guarded.
parameters:
  - name: scope
    type: string
    required: true
    placeholder: "core/retrieval.py hybrid fusion path"
  - name: goal
    type: string
    required: true
    placeholder: "decouple ranking from fusion so signals are swappable"
  - name: constraints
    type: list[string]
    default: []
    help: "Hard limits the refactor must respect (e.g. 'no API changes')"
mission_template: |
  Plan a refactor of {{scope}} to achieve {{goal}}. Constraints:
  {{constraints | comma}}. Output: a phased plan (ordered groups) +
  sample diffs + an up/down migration pair. Each phase must keep tests
  green.
roster_override: ["code_reader", "refactor_specialist", "reviewer"]
skill_plan:
  - dependency_order
  - test_driven_implement
  - migration_writer
  - finalize_project
quality_contract:
  correctness: 5
  completeness: 4
  provenance: 4
  defensibility: 4
  usability: 5
  honesty: 5
  reproducibility: 5
  efficiency: 3
  pass_threshold: 0.75
fitness_command: |
  test -s "{{output_path}}"
  && grep -q '^## Phase' "{{output_path}}"
output_shape: "lab/finalized/refactor_{{lab_id}}.md"
estimated_cost_usd: 0.09
estimated_time_minutes: 14
typical_acceptance_rate: null
mcp_tool_signature:
  name: bert.refactor_plan
  description: "Phased code refactor plan with diffs."
  returns_schema: refactor_plan_result.v1.json
---
# Refactor Plan

Plans a reversible, test-guarded refactor. The `skill_plan` sequences
`dependency_order` (topologically phase the change into parallel
groups), `test_driven_implement` (each phase keeps tests green),
`migration_writer` (up/down + preflight + verify), then
`finalize_project`.

> Skill-plan note: the v1.0 implementation spec listed
> `code_search_in_repo` + `refactor_with_tests_passing` here; those are
> not in the Sprint-2 seed registry. The `code_reader` /
> `refactor_specialist` roster roles handle repo reading via tools, and
> `test_driven_implement` covers the tests-passing discipline, so this
> plan uses existing skills and every ref resolves.
