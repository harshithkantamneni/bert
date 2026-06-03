# note-cli — seed brief

## Problem

Knowledge workers lose ideas because the friction between "thought" and
"note recorded" is too high. Existing tools (Obsidian, Notion, Apple Notes)
require app-switching, login, or network. The terminal is always open;
notes-from-the-terminal would be 10× faster.

## Constraints

1. **Single binary** — no install ceremony, no requirements.txt
2. **Stdlib only** — no third-party deps for the core CLI
3. **Offline-first** — works on a plane, in a SCIF, on a dead VPN
4. **Sub-100ms capture** — from `note "thought"` to disk-flushed

## Success metric

Time from terminal-cold to thought-captured ≤ 2 seconds, 95th percentile,
on a 5-year-old laptop.

## Out of scope (for now)

- Sync to cloud
- GUI / web interface
- AI summarization (separate concern)
- Multi-device merge

## Cycle 1 target

Ship the `note capture` command: `note "<text>"` writes a Markdown file
to `~/.notes/YYYY-MM-DD.md` with frontmatter tags from `--tag` flags.

Tests verify: capture latency, file format, tag extraction.
