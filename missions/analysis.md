# Mission — Analysis: Audit Existing Findings for Stale Claims

Walk every file under `findings/` and identify claims that may have
gone stale. The deliverable is a structured stale-claim ledger, not
new research.

## What counts as a "stale claim"

- A specific numeric assertion (latency, recall, hit rate, %, $) that
  was made in a finding dated > 30 days ago. Either confirm it still
  holds (with a fresh citation) or mark it `STALE`.
- A reference to a tool, library, or paper. Check whether the
  tool/library has had a major release since the finding's date, or
  whether the paper's claims have been challenged.
- A "we will" or "we plan to" statement older than 14 days that
  hasn't been delivered.

## Required columns (one row per claim)

- **finding_path** — relative path of the finding file
- **finding_date** — from file mtime or front-matter
- **claim_excerpt** — one quoted sentence containing the claim
- **claim_type** — `metric` / `tool_reference` / `plan` / `other`
- **status** — `HOLDS` / `STALE` / `UNVERIFIABLE` / `WITHDRAWN`
- **evidence** — citation or one-sentence rationale

## Procedure per cycle

- **Researcher**: pick 1-3 finding files not yet audited this run;
  read each carefully; extract claims; check each against current
  knowledge or memory_search prior cycles for already-known
  contradictions. Use memory_search heavily — this mission is about
  cross-referencing.
- **Strategist**: review the researcher's audit rows; flag any that
  smell like fabricated "evidence"; specify which findings should be
  audited next.

## Constraints

- **Every `STALE` or `WITHDRAWN` row needs a specific reason.** "Old"
  is not a reason; "the claim cited X tool at v1.0, but v2.0 changed
  the relevant behavior, citation: <url>" is.
- Do NOT modify the original findings — this mission produces a
  ledger, not edits.
- A finding can be skipped if it's purely architectural commentary
  (no falsifiable claims).

## Expected output

Each cycle extends `findings/stale_claims_ledger.md`. The strategist
may add commentary on patterns (e.g., "many findings cited NVIDIA NIM
as a commercial provider — that's now known to be non-commercial-only
per their AUP").

## Non-goals

- Fixing the stale claims.
- Generating new findings.
- Auditing files outside `findings/`.
