# Mission — Research: Production Vector Database Comparison (Q2 2026)

Survey production-grade open-source vector databases as they exist in
Q2 2026. Produce a comparison table covering at least 5 systems.

## Required columns

For each system, fill:

- **License** (Apache 2.0 / MIT / Elastic License / etc.)
- **Mode** (embedded library / standalone server / both)
- **Index types** (HNSW / IVF / DiskANN / SCANN / …)
- **Quantization** (PQ / SQ / OPQ / none)
- **Recall@10 on a published benchmark** (cite the benchmark + the number)
- **Approximate p99 query latency at 1M vectors** (cite the source)
- **RAM footprint at 1M × 768-dim** (cite the source)
- **Sharding / clustering support** (yes/no + brief)
- **Hybrid search support** (dense + sparse)
- **Best-fit use case** (one sentence)

## Procedure per cycle

- **Researcher**: pick 1-2 systems not yet covered; gather their
  documentation + benchmark papers; record sources. Use memory_search
  to check what prior cycles already covered to avoid duplicating work.
- **Strategist**: review the researcher's row(s); flag missing
  citations, suspect numbers (anything that smells fabricated),
  inconsistencies vs published benchmarks; recommend the next system
  to cover.

## Constraints

- **Every cell must have a real, verifiable citation.** No
  `example.com`, no placeholder URLs, no "[Author et al.]" without an
  actual paper.
- If a number isn't independently verifiable, write `unknown` —
  fabricating to fill the cell is a REJECT.
- Stay on free-tier providers.

## Expected output

Each cycle's finding extends a running comparison table in
`findings/vector_db_comparison.md`. The strategist may rearrange or
correct rows but should not delete prior cycle's contributions.

## Non-goals

- Closed-source / paid-only systems (Pinecone, etc.) — skip.
- Performance claims without published benchmarks.
