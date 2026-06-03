"""Inspect AI eval registry for bert's pre-registered falsifier targets.

Per FINAL_implementation_plan_2026-05-07.md §11.3 acceptance:
"all 14 A6 falsifier targets express as Inspect AI tests + run
reproducibly". Wires bert's `tools/falsifier_baseline.py` 14 targets
into UK AISI's Inspect AI framework so they run as standard eval
cases alongside the rest of the agentic-AI eval ecosystem.

Run all:
  inspect eval evals/inspect/falsifiers.py

Run one:
  inspect eval evals/inspect/falsifiers.py@threshing_structural_validity
"""
