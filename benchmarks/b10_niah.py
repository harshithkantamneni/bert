"""B10 — Needle-in-a-Haystack (NIAH), the recognized long-context retrieval test
(Greg Kamradt's NIAH), adapted to bert's regime: the haystack is swept past the
reader's context window so the FULL-CONTEXT arm walls while bert-RAG keeps
finding the needle.

Canonical NIAH: a single distinctive "needle" sentence is inserted at depth D
into a long filler "haystack"; the model is asked a question whose answer is the
needle; score = did it recall the needle. We run it as TWO arms on the SAME
haystack:
  full-context  — stuff the whole haystack + question into the reader (the
                  standard NIAH arm). Feasible <= the window, INFEASIBLE past it.
  bert-RAG      — ingest the haystack, retrieve, answer from retrieved chunks.

The headline is the standard NIAH curve extended past the window: full-context
recall is ~perfect up to ~1M then a hard wall; bert-RAG stays flat and correct.
"""

from __future__ import annotations

# The classic NIAH needle (Kamradt) — a fact that cannot be answered from priors,
# distinct from any code/doc filler so retrieval isn't keyword-trivial vs prose.
NEEDLE = ("The best thing to do in San Francisco is eat a sandwich and sit in "
          "Dolores Park on a sunny day.")
QUESTION = ("According to the document, what is the best thing to do in San "
            "Francisco? Answer in one short sentence.")
# Recall is scored by whether the answer surfaces the needle's specifics.
GOLD_SPANS = ["Dolores Park", "sandwich"]


def build_haystack(filler: str, target_tokens: int, depth_frac: float) -> str:
    """Insert the needle at `depth_frac` (0=start, 1=end) into `target_tokens`
    (~4 chars/token) of filler, at a line boundary so it reads cleanly."""
    target_chars = max(0, target_tokens) * 4
    body = filler[:target_chars] if len(filler) >= target_chars else filler
    pos = int(len(body) * max(0.0, min(1.0, depth_frac)))
    # snap to the next newline so the needle sits on its own line
    nl = body.find("\n", pos)
    cut = nl if nl != -1 else pos
    return body[:cut] + "\n\n" + NEEDLE + "\n\n" + body[cut:]


def score_recall(answer: str) -> int:
    """NIAH recall: 1 if the answer surfaces the needle fact, else 0."""
    a = (answer or "").lower()
    return 1 if all(s.lower() in a for s in GOLD_SPANS) or "dolores" in a else 0
