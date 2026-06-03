"""B9 — Long-context RAG benchmark: arm context-builders + the runner glue.

The thesis bert is actually built for: when the relevant context exceeds the
model's window, retrieval (RAG) holds answer quality flat while input cost stays
sub-linear, whereas full-context stuffing either pays linearly or hits an
infeasibility wall. Arms (all share the SAME reader model; only the CONTEXT
differs):

  A0 full     — stuff the ENTIRE corpus (quality ceiling; infeasible past window)
  A1 naive    — first-N-tokens truncation (the "just paste what fits" baseline)
  A2 smart    — file manifest + head of each file (steelman truncation)
  A2prime     — smart truncation budget-MATCHED to A3's retrieved-chunk tokens
  A3 rag      — hybrid_retrieve -> reranked chunks only (the thesis arm)
  A4 vec      — vector-only retrieval (ablation: how much hybrid+rerank adds)
  A5 bm25     — sparse-only retrieval (ablation: RRF fusion value)

The pure context-assembly lives here (unit-tested); the corpus ingest, retrieval,
reader call, and grading are wired in b9_rag_runner (touches memory + providers).
"""

from __future__ import annotations

from benchmarks import b9_rag_stats as _st

RAG_ARMS = ("A3", "A4", "A5")        # arms that retrieve
FULL_ARMS = ("A0",)                  # arms that stuff the whole corpus
TRUNC_ARMS = ("A1", "A2", "A2prime")  # arms that truncate to a budget


def _concat(corpus_files: list[dict]) -> str:
    return "\n\n".join(f"### {f['path']}\n{f['content']}" for f in corpus_files)


def _truncate_to_tokens(text: str, budget_tokens: int) -> str:
    """Keep the head up to ~budget_tokens (~4 chars/token)."""
    return text[: max(0, budget_tokens) * 4]


def _smart_truncate(corpus_files: list[dict], budget_tokens: int) -> str:
    """Steelman truncation: a manifest of ALL file paths, then the head of each
    file, round-robin, until the token budget is spent. Beats naive head-only
    because it gives the reader awareness of the whole tree + a slice of each."""
    manifest = "FILE MANIFEST:\n" + "\n".join(f"- {f['path']}" for f in corpus_files)
    budget_chars = max(0, budget_tokens) * 4 - len(manifest)
    if budget_chars <= 0 or not corpus_files:
        return manifest
    per_file = budget_chars // len(corpus_files)
    heads = [f"### {f['path']}\n{f['content'][:per_file]}" for f in corpus_files]
    return manifest + "\n\n" + "\n\n".join(heads)


def build_context(arm: str, *, corpus_files: list[dict],
                  retrieved_chunks: list[str], budget_tokens: int | None) -> str:
    """Assemble the context an arm's reader is allowed to see. Pure."""
    if arm in FULL_ARMS:
        return _concat(corpus_files)
    if arm in RAG_ARMS:
        return "\n\n".join(f"[chunk {i+1}]\n{c}" for i, c in enumerate(retrieved_chunks))
    if arm == "A1":
        return _truncate_to_tokens(_concat(corpus_files), budget_tokens or 0)
    if arm == "A2":
        return _smart_truncate(corpus_files, budget_tokens or 0)
    if arm == "A2prime":
        # Budget-matched to what the RAG arm actually retrieved — the fairest
        # truncation control (same token budget, no semantic selection).
        budget = (budget_tokens if budget_tokens is not None
                  else _st.est_tokens("".join(retrieved_chunks)))
        return _smart_truncate(corpus_files, budget)
    raise ValueError(f"unknown RAG arm: {arm!r}")


def a0_feasible(corpus_tokens: int, *, reader_window: int = 1_000_000,
                margin: float = 0.02, overhead_tokens: int = 2000) -> tuple[bool, int]:
    """Pre-flight gate for the A0 full-context arm: can the whole corpus + the
    question + output headroom fit the reader's context window? Returns
    (feasible, corpus_tokens). Past the window this is the WALL — recorded as a
    binary result, never sent (so no wasted quota, no garbage answer)."""
    budget = reader_window * (1.0 - margin) - overhead_tokens
    return (corpus_tokens <= budget, corpus_tokens)


def arms_for_tier(corpus_tokens: int, *, reader_window: int = 1_000_000,
                  base_arms=("A0", "A1", "A2", "A3", "A4")) -> list[str]:
    """Arms to run at a given corpus size. A0 (full-context) is dropped once the
    corpus exceeds the window — that drop IS the wall. Retrieval/truncation arms
    run at every size."""
    feasible, _ = a0_feasible(corpus_tokens, reader_window=reader_window)
    return [a for a in base_arms if a != "A0" or feasible]


def reader_prompt(question: str, context: str) -> str:
    """The reader sees ONLY the assembled context — identical instruction across
    arms so the only variable is WHAT context reached it."""
    return (
        "Answer the question using ONLY the provided context. If the context "
        "does not contain the answer, say so explicitly — do not guess.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely."
    )
