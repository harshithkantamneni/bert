---
name: findings_synthesize
version: "1.0"
description: |
  Mid-cycle synthesis: take a set of raw findings from current cycle
  and produce a structured intermediate write-up (NOT the final
  polished artifact — that's finalize_project). Used by the writer
  role to consolidate before moving to the next cycle.
trigger_when: |
  Writer role has accumulated 3+ findings in the current cycle
  and wants to compress them into a single readable handoff for
  the next cycle's roles.
inputs:
  findings:     {type: list, required: true}
  objective:    {type: string, required: true}
  output_path:  {type: string, required: true}
  style:        {type: string, default: "research_memo", description: "research_memo | exec_brief | technical_note"}
outputs:
  synthesis_path: {type: string}
  themes_identified: {type: list, description: "Cross-finding themes"}
  unresolved_threads: {type: list, description: "Open questions surfaced during synthesis"}
tools_required: [cluster_findings, write_synthesis, Write]
steps:
  - id: cluster
    tool: cluster_findings
    args:
      findings: "{{findings}}"
    capture: clusters
  - id: write_synth
    tool: write_synthesis
    args:
      clusters: "{{clusters.clusters}}"
      objective: "{{objective}}"
      style: "{{style}}"
    capture: synth
  - id: write_file
    tool: Write
    args:
      file_path: "{{output_path}}"
      content: "{{synth.body}}"
    capture: write_result
  - id: pluck_path
    tool: identity
    args:
      value: "{{output_path}}"
    capture: synthesis_path
  - id: pluck_themes
    tool: identity
    args:
      value: "{{clusters.themes}}"
    capture: themes_identified
  - id: pluck_threads
    tool: identity
    args:
      value: "{{synth.unresolved_threads}}"
    capture: unresolved_threads
failure_modes:
  - condition: "cluster_findings finds < 2 themes from 3+ findings"
    handler: retry
    max_retries: 1
---

# findings_synthesize

**Quality bar**: a synthesis without `unresolved_threads` is a
synthesis pretending closure. Almost every cycle has open threads;
suppressing them rots the corpus.
