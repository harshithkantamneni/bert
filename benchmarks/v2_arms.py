"""v2 arms — every context-gathering strategy, instrumented (cost + latency +
provider), with the controls v1 lacked:

  A0   closed-book      reader answers with NO context (parametric-knowledge baseline)
  A1   naive-trunc      head of corpus to a token budget
  A2   smart-trunc      manifest + round-robin file heads to a token budget
  A3   hybrid-RAG       vector + BM25 + RRF + cross-encoder rerank, top-k chunks
  A4   vector-RAG       dense only, top-k chunks
  A5   bm25-RAG         sparse only, top-k chunks
  A6   graph-RAG        real Aider RepoMap (PageRank) -> chunks
  A7w  agentic-grep     model iteratively greps/reads — WEAK free agent (llama-3.3-70b)
  A7f  agentic-grep     same loop driven by a FRONTIER agent (Claude via `claude -p`
                        with its real read-only Grep/Read/Glob tools, run in the corpus dir)

A0 is the critical missing control from v1: without it we cannot separate "the
retriever found it" from "the reader already knew it" (popular libs leak into
pretraining). A7f removes the v1 confound where agentic-grep was handicapped by
running on the weak reader model.

Reader is PINNED to a single provider (no silent cascade); the provider that
actually served each call is recorded so serving-stack variance is observable.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO))

from benchmarks import b9_agentic_grep as AG  # noqa: E402
from benchmarks import b9_aider_retrieve as AID  # noqa: E402
from benchmarks import b9_rag as RAG  # noqa: E402
from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import b9_rag_stats as ST  # noqa: E402

ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7w", "A7f"]
_RAG_METHOD = {"A3": "hybrid", "A4": "vector", "A5": "bm25"}

# free reader: PINNED to a SINGLE model/provider (no groq fallback) so the whole
# benchmark uses exactly one llama build — nvidia's meta/llama-3.3-70b-instruct.
# (groq's llama-3.3-70b-versatile is a different build; mixing would confound the
# model-controlled comparison. The retry in make_pinned_reader still gives nvidia
# two attempts on a transient error.)
FREE_READER = [("nvidia", "meta/llama-3.3-70b-instruct")]


def make_pinned_reader(cascade=FREE_READER, max_tokens=600):
    """reader(prompt) -> (text, provider_used). Pinned to cascade[0]; falls back
    only on hard error, tagging which provider actually served the call."""
    from core import provider as prov

    def _read(prompt: str):
        import time as _t
        # 2 passes over the cascade with a backoff — under parallel load the free
        # tiers occasionally 429; a retry keeps a transient limit from turning a
        # correct answer into a false-negative.
        for attempt in range(2):
            for prov_name, model in cascade:
                try:
                    r = prov.call(prov_name, [{"role": "user", "content": prompt}],
                                  model=model, max_tokens=max_tokens, temperature=0.0,
                                  timeout=60.0)
                    if r.finish_reason != "error" and not (r.text or "").startswith("[bert]"):
                        return (r.text, prov_name)
                except Exception:  # noqa: BLE001
                    continue
            if attempt == 0:
                _t.sleep(2.0)
        return ("[reader failed: all lanes errored]", "none")
    return _read


def _closed_book_prompt(question: str) -> str:
    return ("Answer the question from your own knowledge. If you are not sure, "
            f"say so explicitly — do not guess.\n\nQUESTION: {question}\n\nAnswer concisely.")


def _rag_context(chunks: list[str]) -> str:
    return "\n\n".join(f"[chunk {i+1}]\n{c}" for i, c in enumerate(chunks))


def _retrieve(arm: str, question: str, lab: Path, corpus_root: Path,
              top_n: int) -> list[str]:
    """Return the chunk texts an arm's reader sees."""
    if arm in _RAG_METHOD:
        return [c for _id, c in RR.retrieve_for(question, lab, method=_RAG_METHOD[arm], top_n=top_n)]
    if arm == "A6":
        rm, files, root = _aider_cache(corpus_root)
        fc = AID.load_file_chunks(lab / "memory.db")
        return [c for _id, c in AID.aider_filerank_retrieve(
            question, rm=rm, all_files=files, root=root, corpus_dir=corpus_root,
            file_chunks=fc, top_n=top_n)]
    return []


_AIDER_CACHE: dict[str, tuple] = {}


def _aider_cache(corpus_root: Path):
    key = str(corpus_root)
    if key not in _AIDER_CACHE:
        _AIDER_CACHE[key] = AID.build_repomap(corpus_root)
    return _AIDER_CACHE[key]


def run_arm(arm: str, question: str, *, corpus_root: Path, lab: Path,
            corpus_files: list[dict], budget_tokens: int, reader, top_n: int = 10,
            frontier_model: str = "sonnet", precomputed_chunks: list[str] | None = None) -> dict:
    """Run one arm on one question. Returns dict with answer + instrumentation.
    precomputed_chunks: for RAG arms (A3-A6), retrieval is deterministic, so the
    runner precomputes it once (A6 in the aider venv) and passes the chunk texts
    here — avoids re-retrieving across k-repeats/budgets and the aider-venv split."""
    t0 = time.monotonic()
    provider_used = "n/a"
    cost_usd = 0.0
    steps = None
    ctx_tokens = 0

    if arm == "A0":
        ans, provider_used = reader(_closed_book_prompt(question))
    elif arm in ("A1", "A2"):
        ctx = RAG.build_context(arm, corpus_files=corpus_files, retrieved_chunks=[],
                                budget_tokens=budget_tokens)
        ctx_tokens = ST.est_tokens(ctx)
        ans, provider_used = reader(RAG.reader_prompt(question, ctx))
    elif arm in ("A3", "A4", "A5", "A6"):
        chunks = precomputed_chunks if precomputed_chunks is not None \
            else _retrieve(arm, question, lab, corpus_root, top_n)
        ctx = _rag_context(chunks)
        ctx_tokens = ST.est_tokens(ctx)
        ans, provider_used = reader(RAG.reader_prompt(question, ctx))
    elif arm == "A7w":
        res = AG.agentic_grep_answer(question, corpus_root, FREE_READER, max_steps=8)
        ans, steps, provider_used = res["answer"], res["steps"], "free-agent"
        ctx_tokens = sum(t.get("obs_len", 0) for t in res.get("tool_log", [])) // 4
    elif arm == "A7f":
        ans, cost_usd, steps = _frontier_grep(question, corpus_root, frontier_model)
        provider_used = f"claude-{frontier_model}"
    else:
        raise ValueError(f"unknown arm {arm!r}")

    return {
        "arm": arm, "answer": (ans or "").strip(), "method": arm,
        "provider": provider_used, "input_tokens": ctx_tokens,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "cost_usd": cost_usd, "steps": steps,
    }


def _frontier_grep(question: str, corpus_root: Path, model: str) -> tuple[str, float, int]:
    """Agentic grep driven by Claude via `claude -p` with its real read-only
    tools (Grep/Read/Glob), run IN the corpus dir. The faithful frontier-agent
    version of what Claude Code actually does. Returns (answer, cost_usd, n_turns)."""
    prompt = (f"Answer this factual question about the code in this directory by "
              f"searching it with grep/read. Quote the exact value. Be concise.\n\n{question}")
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", model, "--output-format", "json",
             "--permission-mode", "acceptEdits",
             "--allowedTools", "Grep", "Read", "Glob"],
            input=prompt, capture_output=True, text=True, cwd=str(corpus_root),
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return ("[A7f: timeout]", 0.0, 0)
    if proc.returncode != 0:
        return (f"[A7f failed rc={proc.returncode}: {proc.stderr[:200]}]", 0.0, 0)
    try:
        out = json.loads(proc.stdout)
        ans = out.get("result", "") or ""
        cost = float(out.get("total_cost_usd") or out.get("cost_usd") or 0.0)
        turns = int(out.get("num_turns") or 0)
        return (ans, cost, turns)
    except Exception:  # noqa: BLE001
        return (proc.stdout[:2000], 0.0, 0)


if __name__ == "__main__":
    import os
    lab = Path("/tmp/b9_grep_pilot_lab")
    if not (lab / "memory.db").exists():
        lab = Path("/tmp/b9_graph_lab")
    corpus = Path("/tmp/b9_corpus")
    corpus_files = RR.load_corpus_files(corpus)
    reader = make_pinned_reader()
    q = "In httpx, what is the default value of the keepalive_expiry parameter on the Limits class?"
    # A6 (graph/Aider) needs the aider venv; covered by the runner's precompute step.
    test_arms = ["A0", "A1", "A2", "A3", "A4", "A5", "A7w"]
    if os.environ.get("V2_TEST_FRONTIER"):
        test_arms.append("A7f")
    print(f"self-test on: {q}\n")
    for arm in test_arms:
        r = run_arm(arm, q, corpus_root=corpus, lab=lab, corpus_files=corpus_files,
                    budget_tokens=15000, reader=reader, top_n=10)
        print(f"  {r['arm']:4} prov={r['provider']:18} tok={r['input_tokens']:6} "
              f"{r['latency_ms']:6}ms cost=${r['cost_usd']:.4f} steps={r['steps']} "
              f":: {r['answer'][:70].strip()}")
    print("\nv2_arms self-test: OK")
