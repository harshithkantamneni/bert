"""B9 RAG benchmark runner — ties ingest -> retrieve -> build_context -> read ->
grade -> score into a per-query row, and orchestrates the sweep across arms,
corpus tiers, and the gold Q&A set.

run_one_query is the pure-ish core (deps injected, offline-testable). The real
ingest/retrieve/read/grade wiring (memory + providers) is in the helper
functions below and is exercised by the live pilot, not the unit tests.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks import b9_rag as rag  # noqa: E402
from benchmarks import b9_rag_stats as st  # noqa: E402


def run_one_query(question: str, gold_answer: str, gold_chunk_ids: list[str], *,
                  arm: str, corpus_files: list[dict], retrieved: list[tuple],
                  budget_tokens: int | None, reader, grader, k: int = 10,
                  gold_spans: list[str] | None = None) -> dict:
    """One (question, arm) trial. `retrieved` is a list of (chunk_id, content)
    for RAG arms (ignored by full/truncation arms). `reader(prompt)->answer` and
    `grader(question, gold_answer, answer)->0|1` are injected. When `gold_spans`
    is given, retrieval recall is content-match (the robust gold-set way);
    otherwise chunk-id overlap with `gold_chunk_ids`."""
    retrieved_ids = [cid for cid, _ in retrieved]
    retrieved_texts = [txt for _, txt in retrieved]
    ctx = rag.build_context(arm, corpus_files=corpus_files,
                            retrieved_chunks=retrieved_texts,
                            budget_tokens=budget_tokens)
    prompt = rag.reader_prompt(question, ctx)
    answer = reader(prompt)
    correct = int(grader(question, gold_answer, answer))
    row = {
        "arm": arm, "question": question, "correct": correct,
        "input_tokens": st.est_tokens(ctx), "answer": answer,
        "recall_at_10": None, "ndcg_at_10": None,
    }
    if arm in rag.RAG_ARMS:
        if gold_spans is not None:
            row["recall_at_10"] = st.recall_spans(retrieved_texts, gold_spans, k)
            row["hit_rate_at_10"] = st.hit_spans(retrieved_texts, gold_spans, k)
            row["ndcg_at_10"] = row["recall_at_10"]   # span-based: rank-agnostic proxy
        else:
            row["recall_at_10"] = st.recall_at_k(retrieved_ids, gold_chunk_ids, k)
            row["ndcg_at_10"] = st.ndcg_at_k(retrieved_ids, gold_chunk_ids, k)
            row["hit_rate_at_10"] = st.hit_rate_at_k(retrieved_ids, gold_chunk_ids, k)
    return row


def load_corpus_files(corpus_dir: Path,
                      exts=(".py", ".md", ".txt")) -> list[dict]:
    """Read a corpus tree into [{path, content}] for the full/truncation arms."""
    corpus_dir = Path(corpus_dir)
    out = []
    for f in sorted(corpus_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            try:
                out.append({"path": str(f.relative_to(corpus_dir)),
                            "content": f.read_text(encoding="utf-8", errors="replace")})
            except OSError:
                continue
    return out


# ── real wiring (live pilot; not unit-tested) ────────────────────────

_RAG_METHOD = {"A3": "hybrid", "A4": "vector", "A5": "bm25"}


def run_sweep(corpus_dir: Path, gold_questions: list[dict], *, arms: list[str],
              reader, grader, lab: Path, trunc_window_tokens: int = 15000,
              top_n: int = 10, ingest: bool = True,
              checkpoint_path: Path | None = None) -> dict:
    """Run the full RAG sweep: ingest the corpus once, then for each gold question
    retrieve (per RAG method) and run every arm through run_one_query. Prints
    per-question progress and (if checkpoint_path) persists rows after every
    question so a long free-tier run is observable and recoverable. Returns
    {rows, by_arm}."""
    import json as _json

    from benchmarks import b9_rag_stats as _st
    corpus_files = load_corpus_files(corpus_dir)
    if ingest:
        n = ingest_corpus_into_lab(corpus_dir, lab)
        print(f"[sweep] ingested {n} files; {len(gold_questions)} questions x "
              f"{len(arms)} arms", flush=True)
    rows: list[dict] = []
    total = len(gold_questions)
    for i, q in enumerate(gold_questions, 1):
        t0 = time.monotonic()
        question, gold_answer = q["question"], q.get("gold_answer", "")
        gold_spans = q.get("gold_spans") or []
        retrieved_by_method = {}
        for arm in arms:
            m = _RAG_METHOD.get(arm)
            if m and m not in retrieved_by_method:
                try:
                    retrieved_by_method[m] = retrieve_for(question, lab, method=m, top_n=top_n)
                except Exception as exc:  # noqa: BLE001
                    retrieved_by_method[m] = []
                    print(f"[retrieve warn] {m} {question[:40]}: {exc}", flush=True)
        marks = []
        for arm in arms:
            retrieved = retrieved_by_method.get(_RAG_METHOD.get(arm), [])
            try:
                row = run_one_query(
                    question, gold_answer, [], arm=arm, corpus_files=corpus_files,
                    retrieved=retrieved, budget_tokens=trunc_window_tokens,
                    reader=reader, grader=grader, k=top_n, gold_spans=gold_spans)
            except Exception as exc:  # noqa: BLE001
                row = {"arm": arm, "question": question, "correct": 0,
                       "input_tokens": 0, "error": str(exc),
                       "recall_at_10": None, "ndcg_at_10": None}
            row["tier"] = q.get("tier")
            rows.append(row)
            marks.append(f"{arm}={'✓' if row['correct'] else '✗'}")
        if checkpoint_path is not None:
            Path(checkpoint_path).write_text(_json.dumps({"rows": rows}, indent=2))
        print(f"[q {i}/{total}] {q.get('tier'):10} {round(time.monotonic()-t0)}s  "
              f"{' '.join(marks)}", flush=True)
    return {"rows": rows, "by_arm": _st.aggregate_by_arm(rows)}


def ingest_corpus_into_lab(corpus_dir: Path, lab: Path) -> int:
    """Ingest an external corpus into an isolated benchmark lab (WS0b path) and
    eagerly index it. Returns files ingested. Retrieval then scopes to this lab."""
    from core import lab_context, memory
    lab.mkdir(parents=True, exist_ok=True)
    tok = lab_context.set_active_lab_path(lab)
    try:
        return memory.ingest_corpus(corpus_dir, eager_index=True)
    finally:
        lab_context.reset_active_lab_path(tok)


def retrieve_for(question: str, lab: Path, *, method: str = "hybrid",
                 top_n: int = 10) -> list[tuple]:
    """Retrieve (chunk_id, content) for a question, scoped to the lab's corpus.
    method: 'hybrid' (A3), 'vector' (A4), 'bm25' (A5)."""
    from core import lab_context, memory
    tok = lab_context.set_active_lab_path(lab)
    try:
        if method == "vector":
            hits = memory.search(question, k=top_n)
            return [(str(h.get("id") or h.get("path")), h.get("content", "")) for h in hits]
        if method == "bm25":
            # Real BM25-only baseline. Previously "bm25" fell through to
            # hybrid_retrieve, making A5 a silent DUPLICATE of A3 (hybrid) —
            # which is why their accuracies were byte-identical.
            from core import bm25 as _bm25
            hits = _bm25.search(question, lab_path=lab, k=top_n)
            return [(f"bm25:{h.chunk_id}", h.content or "") for h in hits]
        from core import retrieval
        res = retrieval.hybrid_retrieve(question, top_n=top_n)
        # RetrievalResult has .id and .text; content may be an excerpt, so fall
        # back to the full chunk text from metadata when present.
        out = []
        for r in res:
            content = (r.metadata or {}).get("content") or r.text or ""
            out.append((str(r.id), content))
        return out
    finally:
        lab_context.reset_active_lab_path(tok)


def make_reader(cascade):
    """A reader bound to a (free) model cascade. reader(prompt) -> answer text."""
    from core import provider as prov

    def _read(prompt: str) -> str:
        for prov_name, model in cascade:
            try:
                resp = prov.call(prov_name, [{"role": "user", "content": prompt}],
                                 model=model, max_tokens=600, temperature=0.0,
                                 timeout=60.0)
                if resp.finish_reason != "error" and not resp.text.startswith("[bert]"):
                    return resp.text
            except Exception:  # noqa: BLE001
                continue
        return "[reader failed: all lanes errored]"
    return _read


def make_max_reader(model: str = "opus", timeout: float = 1200.0):
    """A0 full-context reader on the Max-plan bridge (Opus 1M, free on the
    subscription). The (huge) prompt is piped via STDIN — passing a ~1M-token
    prompt as a CLI arg would blow ARG_MAX. On Max, `opus` auto-upgrades to the
    1M window. A prompt over the window returns a non-zero/error -> recorded as
    the wall by the caller (but we pre-flight gate before ever calling)."""
    import subprocess

    def _read(prompt: str) -> str:
        try:
            proc = subprocess.run(
                ["claude", "-p", "--model", model, "--output-format", "json"],
                input=prompt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "[max reader: timeout]"
        if proc.returncode != 0:
            return f"[max reader failed rc={proc.returncode}: {proc.stderr[:200]}]"
        try:
            out = _json_loads(proc.stdout)
            if out.get("is_error"):
                return f"[max reader is_error: {str(out.get('result',''))[:200]}]"
            return out.get("result", "") or ""
        except Exception:  # noqa: BLE001
            return proc.stdout[:4000]
    return _read


def _json_loads(s):
    import json as _j
    return _j.loads(s)


def make_grader(cascade):
    """An answer-correctness grader (judge). grader(q, gold, ans) -> 0|1.
    The judge sees the gold answer and decides if the candidate is correct."""
    import json as _json

    from core import provider as prov

    def _grade(question: str, gold: str, answer: str) -> int:
        sysp = ("You grade whether a candidate answer is correct given the gold "
                "answer. Be strict about factual correctness, lenient about "
                "wording. Return ONLY JSON: {\"correct\": true|false}.")
        userp = (f"QUESTION: {question}\nGOLD ANSWER: {gold}\n"
                 f"CANDIDATE ANSWER: {answer}\n\nIs the candidate correct? JSON only.")
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": userp}]
        for prov_name, model in cascade:
            try:
                resp = prov.call(prov_name, msgs, model=model, max_tokens=80,
                                 temperature=0.0,
                                 response_format={"type": "json_object"}, timeout=40.0)
                if resp.finish_reason == "error" or resp.text.startswith("[bert]"):
                    continue
                return 1 if bool(_json.loads(resp.text).get("correct")) else 0
            except Exception:  # noqa: BLE001
                continue
        return 0
    return _grade
