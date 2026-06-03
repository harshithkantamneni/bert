---
name: migration_writer
version: "1.0"
description: |
  Author a forward + rollback migration pair for a schema, config,
  or data change. Output is two artifacts (up + down) plus a
  pre-flight check command and a verification command.
trigger_when: |
  Refactor or engineer role needs to change a schema, file format,
  or any state that's persisted across cycles. Never skip the
  rollback — it's the only thing that makes the change reversible.
inputs:
  description:  {type: string, required: true}
  target_layer: {type: string, required: true, description: "e.g., 'sqlite_memory', 'kuzu_kg', 'lab_schema'"}
  output_dir:   {type: string, default: "core/migrations/"}
outputs:
  up_path:      {type: string}
  down_path:    {type: string}
  preflight_cmd: {type: string}
  verify_cmd:   {type: string}
tools_required: [draft_migration_pair, Write]
steps:
  - id: draft
    tool: draft_migration_pair
    args:
      description: "{{description}}"
      target_layer: "{{target_layer}}"
    capture: drafted
  - id: write_up
    tool: Write
    args:
      file_path: "{{output_dir}}{{drafted.up_filename}}"
      content: "{{drafted.up_body}}"
    capture: up_write
  - id: write_down
    tool: Write
    args:
      file_path: "{{output_dir}}{{drafted.down_filename}}"
      content: "{{drafted.down_body}}"
    capture: down_write
  - id: pluck_up
    tool: identity
    args:
      value: "{{up_write.path}}"
    capture: up_path
  - id: pluck_down
    tool: identity
    args:
      value: "{{down_write.path}}"
    capture: down_path
  - id: pluck_preflight
    tool: identity
    args:
      value: "{{drafted.preflight_cmd}}"
    capture: preflight_cmd
  - id: pluck_verify
    tool: identity
    args:
      value: "{{drafted.verify_cmd}}"
    capture: verify_cmd
failure_modes:
  - condition: "draft_migration_pair returns no rollback"
    handler: retry
    max_retries: 1
  - condition: "Write to up or down path fails"
    handler: "emit_disk_failure"
---

# migration_writer

**Quality bar**: a migration without a rollback is not a migration, it's
a one-way door. Refuse to ship it.
