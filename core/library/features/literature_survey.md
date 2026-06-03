---
name: literature_survey
display_name: "Literature Survey"
short_description: "Generate a comparison table of papers on a topic with real citations."
long_description: |
  Surveys recent papers/blogs/docs on a topic. Produces a comparison
  table with user-specified dimensions. Every cell cites a paper-shaped
  source. Acknowledges what was NOT covered.
parameters:
  - name: topic
    type: string
    required: true
    placeholder: "Vector databases Q2 2026"
    validate:
      min_words: 3
      max_words: 20
      error_if_vague: true
  - name: num_papers
    type: int
    default: 10
    min: 3
    max: 30
    help: "Target paper count; bert will note if fewer are available"
  - name: dimensions
    type: list[string]
    min_items: 3
    max_items: 12
    default: ["license", "latency", "recall@10", "RAM footprint"]
    help: "Columns in the comparison table"
  - name: time_horizon_months
    type: int
    default: 18
    help: "How recent the papers must be"
mission_template: |
  Survey papers/blogs/docs on {{topic}} from the last {{time_horizon_months}} months.
  Produce a comparison table with columns: {{dimensions | comma}}.
  Every row must cite a paper-shaped source (not just a homepage).
  Acknowledge which systems were NOT covered and why.
roster_override: null
skill_plan:
  - web_search_and_dedup
  - claim_verify_against_source
  - comparative_evaluation
  - gap_finder
  - red_team_pass
  - finalize_project
quality_contract:
  correctness: 5
  completeness: 4
  provenance: 5
  defensibility: 4
  usability: 3
  honesty: 5
  reproducibility: 3
  efficiency: 2
  pass_threshold: 0.70
fitness_command: |
  test -s "{{output_path}}"
  && grep -q '^# ' "{{output_path}}"
  && grep -qE '^\| .* \| .* \|' "{{output_path}}"
output_shape: "findings/{{topic_slug}}_survey.md"
estimated_cost_usd: 0.05
estimated_time_minutes: 8
typical_acceptance_rate: null
mcp_tool_signature:
  name: bert.literature_survey
  description: "Generate a comparison table of papers on a topic."
  returns_schema: literature_survey_result.v1.json
---
# Literature Survey

Surveys recent papers/blogs/docs on a topic and produces a defensible
comparison table. Every row cites a paper-shaped source; the survey
explicitly names systems it did NOT cover and why.

Mission is research-shaped — `roster_override` is null so the mission
classifier derives the roster (typically `literature_hunter → writer`).
