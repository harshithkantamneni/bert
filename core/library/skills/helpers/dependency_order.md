---
name: dependency_order
version: "1.0"
description: |
  Given a set of tasks with declared `depends_on` edges, return a
  topologically-sorted execution order, OR a clear cycle report if
  the graph is not a DAG. Used by orchestrator role to plan
  multi-step work.
trigger_when: |
  Orchestrator needs to schedule N tasks with interdependencies.
inputs:
  tasks:        {type: list, required: true, description: "[{id, name, depends_on: [ids]}]"}
outputs:
  ordered:      {type: list, description: "Task IDs in execution order"}
  cycle:        {type: list, description: "[ids forming a cycle] or []"}
  parallel_groups: {type: list, description: "[[ids that can run in parallel]] — level by level"}
tools_required: [topo_sort]
steps:
  - id: sort
    tool: topo_sort
    args:
      tasks: "{{tasks}}"
    capture: sort_result
  - id: pluck_ordered
    tool: identity
    args:
      value: "{{sort_result.ordered}}"
    capture: ordered
  - id: pluck_cycle
    tool: identity
    args:
      value: "{{sort_result.cycle}}"
    capture: cycle
  - id: pluck_parallel
    tool: identity
    args:
      value: "{{sort_result.parallel_groups}}"
    capture: parallel_groups
failure_modes:
  - condition: "topo_sort detects a cycle"
    handler: "emit_dependency_cycle"
---

# dependency_order

**Quality bar**: a non-empty `cycle` field with non-empty `ordered`
is a contradiction. Pick one.
