"""Role-capability measurement batteries.

Each module in this package defines a role-specific task battery that
runs against every active model in the cascade. Results land in
`lab/state/capability_matrix.jsonl`; the router (core/router.py,
core/subagent.py:pick_evaluator_model via core/capability_matrix.py)
reads that file each cycle for measurement-driven first-attempt
provider selection.

Each battery exposes:

  BATTERY_SPEC: dict     — metadata (name, description, scoring rubric)
  TASKS: list[Task]      — the task corpus
  score(task, output) -> float
                         — per-task scorer; returns 0..1
  run(provider, model, *, sample=None, ttl_secs=120) -> CapabilityRow
                         — dispatches the corpus and writes a row

The harness in tools/run_capability_harness.py iterates roles ×
providers, calling `run()` on each pair.

Phase 2 scope (this commit): scaffolds with deterministic offline
scorers (no live API) so cron can warm the matrix structure even when
network is down. The held-out human-rated reference sets land as
operational data in `evals/role_capability/reference_sets/` PRs over
time.
"""
