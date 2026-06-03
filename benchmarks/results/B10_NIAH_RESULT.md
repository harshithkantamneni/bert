# B10 — Needle-in-a-Haystack (NIAH), the standard long-context test

_NIAH (Greg Kamradt) is the de-facto test labs cite for context-window claims ("near-perfect recall to N tokens"). A distinctive needle sentence is inserted at depth into a long filler haystack; score = did the reader recall the needle. We run it as two arms on the SAME haystack and sweep length across the 1M window. Filler: local sympy+numpy+scipy text (~9M tokens, no network). Needle: the classic Kamradt SF/Dolores-Park sentence. Full-context reader = Max Opus 1M bridge; bert-RAG reader = free llama-3.3-70b._

| haystack length | full-context (standard NIAH) | bert-RAG | RAG input tokens | needle in top-10 |
|---|---|---|---|---|
| 50K | **recall 1.0** | **recall 1.0** | 3,534 | ✓ |
| 200K | **recall 1.0** | **recall 1.0** | 2,781 | ✓ |
| **2.0M (2× the 1M window)** | **INFEASIBLE — the wall** | **recall 1.0** | 3,599 | ✓ |

**Result:** full-context has the near-perfect recall labs report — *up to the window* — then a hard infeasibility wall at 2M (a 2M-token prompt cannot enter a 1M-token window). **bert-RAG holds perfect needle recall at every length, including 2× the window, at a flat ~3K input tokens** (the needle chunk is retrieved in the top-10 every time). Full-context input cost grows linearly (50K → 200K → infeasible); RAG is constant.

**Why this matters:** this is the recognized NIAH methodology, extended into the regime that is bert's entire reason to exist — haystacks *larger than the window*, where the standard full-context arm cannot run and retrieval is the only thing that works. It anchors the custom B9 wall result to the harness Anthropic/OpenAI/Google use for context-length claims.

**Robustness (RAG depth-sweep, free):** the headline table above is one sample per cell. A follow-up sweep at 50K across 5 needle depths (0/25/50/75/100%) gives **bert-RAG recall = 5/5, with the needle retrieved at rank 1 every time** — i.e. RAG is **depth-invariant** (full-context famously degrades for needles "lost in the middle"; retrieval pulls the needle chunk regardless of position).

**Honest caveats:** the **full-context** cells (50K/200K) are **one sample each** — each is a real Max-Opus quota call, so that arm is not depth-swept; "recall 1.0" there means "the single test passed," and a full depth×length heatmap for the Opus arm would need more quota. The needle is verified genuine (the filler contains neither "Dolores Park" nor "sandwich", and the needle is a made-up fact unanswerable without retrieval). Minor retrieval non-determinism was observed on *exact* rank in isolated runs, but end-to-end recall is stable (5/5 in the sweep). Together: NIAH shows the long-context wall on the standard harness; B9 + BEIR show retrieval quality under realistic distractors.
