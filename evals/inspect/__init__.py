"""Inspect AI eval registry for bert's pre-registered falsifier targets.

Wires bert's `tools/falsifier_baseline.py` 14 targets
into UK AISI's Inspect AI framework so they run as standard eval
cases alongside the rest of the agentic-AI eval ecosystem.

Run all:
  inspect eval evals/inspect/falsifiers.py

Run one:
  inspect eval evals/inspect/falsifiers.py@threshing_structural_validity
"""
