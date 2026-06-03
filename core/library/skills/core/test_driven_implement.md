---
name: test_driven_implement
version: "1.0"
description: |
  Implement a feature TDD-style: read the spec, write failing tests
  first, then minimal code to pass them, then run the test suite,
  iterate until green. The output is BOTH the production code AND
  the tests — never one without the other.
trigger_when: |
  Engineer/refactor role gets a spec or bug report that has a clear
  acceptance condition expressible as a test. Don't use for purely
  exploratory hacking — use for spec'd work.
inputs:
  spec:              {type: string, required: true, description: "Plain-language spec or bug description"}
  target_file:       {type: string, required: true, description: "Where the production code lives (or should be created)"}
  test_file:         {type: string, required: true, description: "Where the test goes"}
  pytest_command:    {type: string, default: ".venv/bin/python -m pytest -x -q"}
  max_iterations:    {type: int,    default: 5,    description: "How many red-green-refactor cycles to attempt"}
outputs:
  code_written:      {type: string, description: "Path of production file modified"}
  test_written:      {type: string, description: "Path of test file modified"}
  iterations_used:   {type: int,    description: "How many cycles were needed"}
  final_test_output: {type: string, description: "Last pytest stdout"}
  passing:           {type: bool,   description: "True iff the test suite is green at exit"}
tools_required: [Read, Write, Bash]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: read_spec_context
    tool: Read
    args:
      file_path: "{{target_file}}"
    capture: existing_code
  - id: write_failing_test
    tool: Write
    args:
      file_path: "{{test_file}}"
      content_hint: "{{spec}}"
    capture: test_written
  - id: run_test_red
    tool: Bash
    args:
      command: "{{pytest_command}} {{test_file}}"
      timeout: 120
    capture: red_output
  - id: implement
    tool: implement_to_pass_test
    args:
      spec: "{{spec}}"
      target_file: "{{target_file}}"
      test_file: "{{test_file}}"
      red_output: "{{red_output.stdout}}"
      max_iterations: "{{max_iterations}}"
    capture: implement_result
  - id: run_test_green
    tool: Bash
    args:
      command: "{{pytest_command}} {{test_file}}"
      timeout: 180
    capture: green_output
  - id: pluck_code
    tool: identity
    args:
      value: "{{implement_result.code_path}}"
    capture: code_written
  - id: pluck_iters
    tool: identity
    args:
      value: "{{implement_result.iterations}}"
    capture: iterations_used
  - id: pluck_final
    tool: identity
    args:
      value: "{{green_output.stdout}}"
    capture: final_test_output
  - id: check_passing
    tool: pytest_passing_check
    args:
      stdout: "{{green_output.stdout}}"
      exit_code: "{{green_output.returncode}}"
    capture: passing
failure_modes:
  - condition: "Pytest segfaults or times out repeatedly"
    handler: "emit_environment_failure"
  - condition: "max_iterations reached without green"
    handler: "emit_unresolved"
---

# test_driven_implement

**Quality bar**: do NOT skip the red-phase. If `red_output` shows a
pass without your test ever failing, your test is wrong (it's not
actually testing the feature). The cycle should be red → green, not
green → greener.
