# bert memory-MCP benchmark — m1 (cross-context-window recall)

**What this tests.** Recall over a software project's accumulated PROSE memory (decision logs, post-mortems, standups) — bert's actual job — where the answer is *not re-derivable from present files*: it lives in a past beyond the context window, is phrased with no greppable keyword overlap (questions are paraphrased), and is buried among realistic filler. Single reader (Claude); only the memory mechanism varies. Judge-graded (3 non-Claude judges, majority).

Arms: `A0` no-memory · `A1` full-context (recency-truncate to window) · `A2` agentic grep over the notes · `A3` naive vector-RAG (cosine top-k, no hybrid/rerank) · `A4` **bert via live MCP** (`memory_search`).

## Corpus sizes

| size | sessions | ~tokens | fits the 200K reader window? |
|---|---|---|---|
| S | 606 | 98,519 | yes |
| M | 7,531 | 1,261,774 | no, exceeds |

## Crossover — accuracy by corpus size

| arm | method | S | M |
|---|---|---|---|
| `A0` | no-memory | **0.10** [0.02,0.20] | **0.04** [0.00,0.10] |
| `A1` | full-context | **0.90** [0.80,0.98] | **0.08** [0.02,0.16] |
| `A2` | agentic-grep | **0.92** [0.84,0.98] | **0.90** [0.82,0.98] |
| `A3` | naive-vector-RAG | **0.50** [0.36,0.64] | **0.40** [0.26,0.54] |
| `A4` | bert via live MCP | **0.96** [0.90,1.00] | **0.90** [0.82,0.98] |

_The story is the slope: `A1` full-context should drop sharply from S→M→L as the needle falls outside the window, while `A4`/`A3` stay flatter (retrieval is size-insensitive) and `A2` pays in latency/turns to scan a growing haystack._

## Significance at size M — `A4` bert-MCP vs each (paired McNemar, Holm)

| vs | Δacc (A4 − other) | 95% CI | Holm p | significant |
|---|---|---|---|---|
| `A0` no-memory | +0.860 | [+0.76,+0.94] | 0.000 | **yes** |
| `A1` full-context | +0.820 | [+0.70,+0.92] | 0.000 | **yes** |
| `A2` agentic-grep | +0.020 | [-0.08,+0.12] | 1.000 | no |
| `A3` naive-vector-RAG | +0.500 | [+0.36,+0.64] | 0.000 | **yes** |

## Per-category accuracy at size M

| category | A0 | A1 | A2 | A3 | A4 |
|---|---|---|---|---|---|
| abandoned_approach | 0.20 | 0.20 | 1.00 | 0.20 | 0.80 |
| arch_decision | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| config_value | 0.00 | 0.00 | 0.75 | 0.00 | 1.00 |
| experiment_result | 0.09 | 0.00 | 1.00 | 0.64 | 0.91 |
| incident_postmortem | 0.00 | 0.00 | 0.70 | 0.50 | 0.90 |
| knowledge_update | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| multi_hop | 0.00 | 0.38 | 1.00 | 0.50 | 1.00 |
| tooling_choice | 0.00 | 0.00 | 1.00 | 0.30 | 0.90 |

## Token cost at size M

| arm | tokens/query | tokens/correct | accuracy |
|---|---|---|---|
| `A0` no-memory | 28,915 | 722,879 | 0.04 |
| `A1` full-context | 152,022 | 1,900,278 | 0.08 |
| `A2` agentic-grep | 282,335 | 314,418 | 0.90 |
| `A3` naive-vector-RAG | 29,864 | 74,660 | 0.40 |
| `A4` bert via live MCP | 148,949 | 165,499 | 0.90 |

## What the results show (honest)

- **Full-context collapses once memory exceeds the window:** `A1` falls from **0.90 (S) → 0.08 (M)** — it can only keep the most-recent ~10% of a 1.26M-token corpus, so it misses almost every older fact. This is the core result: stuffing the context is not a memory system.
- **bert-MCP holds at the top** (0.96 → 0.90) and **ties agentic-grep** (0.90, not significant) — but at **roughly half the token cost** (≈149k vs ≈282k tokens/query): grep scans many files across many turns; bert retrieves a focused slice. Same answers, much cheaper — bert's real edge at the tie.
- **bert-MCP decisively beats naive vector-RAG** (+0.50, p<0.001): hybrid + rerank is worth it; a plain embedding top-k is far weaker on paraphrased recall.
- Consistent with the code benchmark: an agent that can read+reason (grep) is a strong baseline bert *matches* rather than beats — bert wins on **cost** and on **beating the simpler memory approaches**, not by out-accurate-ing the agent.

## Limitations

- **The full-context arm used a 200K-context reader (Claude Sonnet), which rejected prompts above ~180K.** A 1M-context Claude would hold ~80% of this 1.26M corpus and would NOT collapse to 0.08; the over-window result is specific to a 200K window. To show the same wall for a 1M window, the corpus has to clear 1M by a wide margin (the L=4M size) with a 1M-context reader. The bert-vs-naive-vector gap and the token-cost win are window-independent.
- **Needle placement** skews single-evidence facts toward the first half of the timeline, so full-context's recency window catches few; with uniform placement `A1` would reach ~0.10 rather than 0.08. It collapses either way, but the exact floor is placement-dependent.
- Synthetic project memory (one fictional project); single reader (Claude); judge-graded n=50; `knowledge_update` is under-sampled (a contradiction/temporal track is future work).
- `A2`/`A4` token costs are real but reader-specific; the ranking is what generalizes.
