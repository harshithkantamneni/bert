"""m1 memory-MCP benchmark — arm runner (single model = Claude).

For each question, every arm answers with the SAME reader (Claude); only how it
gets context differs. Run per haystack size to trace the crossover curve.

  A0 no-memory          closed-book, no access to the project memory
  A1 full-context       stuff the most-recent sessions up to the window, then truncate
  A2 agentic-grep       Claude with grep/read/glob over the session files
  A3 naive-vector-RAG   single dense embedding, cosine top-k sessions (no hybrid/rerank)
  A4 bert-MCP           ingest haystack into a bert lab; Claude calls memory_search live

Judge-graded (3 non-Claude judges, majority). Checkpointed per (id, arm, size).

  .venv/bin/python benchmarks/m1_arms.py --size S [--arms A0,A1,A2,A3,A4] [--n N]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "benchmarks/results/m1"

from benchmarks import v2_grader as GR  # noqa: E402
from benchmarks import v2_mcp_arm as MCP  # noqa: E402
from benchmarks.v2_tokenomics import (
    _claude,  # claude -p (+tools/cwd), returns (ans, tin, tout)  # noqa: E402
)

CTX_BUDGET = 120_000          # context chars/4 for A1; safely under Claude's input limit
                              # (180k tripped "Prompt is too long"); A1 still truncates the
                              # 1.26M-token M corpus to its most-recent ~10%
CHARS_PER_TOK = 4
TOPK = 10                     # sessions retrieved by A3 / shown by A4 reader


SESS = Path("/tmp/m1_haystacks")  # outside iCloud-synced ~/Desktop


def _sessions_dir(size: str) -> Path:
    return SESS / f"haystack_{size}" / "sessions"


def _load_sessions(size: str) -> list[tuple[int, str]]:
    d = _sessions_dir(size)
    # skip any stray non-numbered files (e.g. sync conflict copies)
    return [(int(p.stem), p.read_text()) for p in sorted(d.glob("*.md")) if p.stem.isdigit()]


def _closed_book(q: str) -> str:
    return (f"Answer from your own knowledge only; you have no access to the project's notes. "
            f"If you cannot know, say so briefly.\n\nQUESTION: {q}")


def _ctx_prompt(q: str, ctx: str) -> str:
    return (f"You are answering a question about a software project using the project-memory "
            f"excerpts below. Answer concisely from them; if absent, say it's not in the provided "
            f"notes.\n\n=== PROJECT MEMORY ===\n{ctx}\n=== END ===\n\nQUESTION: {q}")


# ── A1 full-context (recency-keep up to budget) ──────────────────────
def _full_context(sessions: list[tuple[int, str]]) -> str:
    budget = CTX_BUDGET * CHARS_PER_TOK
    out, used = [], 0
    for _idx, body in reversed(sessions):            # newest first
        if used + len(body) > budget:
            break
        out.append(body); used += len(body)
    return "\n".join(reversed(out))                  # restore chronological order


# ── A3 naive vector RAG (cosine top-k, no hybrid/rerank) ─────────────
_VEC = {}


def _vector_topk(size: str, sessions: list[tuple[int, str]], q: str, k: int = TOPK) -> list[int]:
    import numpy as np

    from core import memory as mem
    if size not in _VEC:
        cache = SESS / f"vec_{size}.npy"
        emb = mem._get_embedder()
        if cache.exists():
            mat = np.load(cache)
        else:
            texts = [b for _i, b in sessions]
            mat = emb.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
            mat = np.asarray(mat, dtype="float32")
            np.save(cache, mat)
        _VEC[size] = (mat, [i for i, _b in sessions])
    mat, idxs = _VEC[size]
    from core import memory as _m
    qv = _m._get_embedder().encode([q], normalize_embeddings=True, show_progress_bar=False)
    qv = __import__("numpy").asarray(qv, dtype="float32")[0]
    scores = mat @ qv
    top = scores.argsort()[::-1][:k]
    return [idxs[t] for t in top]


# ── A4 bert lab (ingest once per size) ───────────────────────────────
_LAB_READY = set()


def _ensure_bert_lab(size: str) -> str:
    lab = Path("/tmp/m1_labs") / size
    if size in _LAB_READY:
        return str(lab)
    from core import lab_context, memory
    lab.mkdir(parents=True, exist_ok=True)
    tok = lab_context.set_active_lab_path(lab)
    try:
        n = memory.ingest_corpus(_sessions_dir(size), eager_index=True)
        print(f"[m1] bert lab {size}: ingested {n} files -> {lab}", flush=True)
    finally:
        lab_context.reset_active_lab_path(tok)
    _LAB_READY.add(size)
    return str(lab)


def answer(arm: str, q: str, size: str, sessions, sdict, model: str):
    if arm == "A0":
        return _claude(_closed_book(q), model=model)
    if arm == "A1":
        return _claude(_ctx_prompt(q, _full_context(sessions)), model=model)
    if arm == "A2":
        return _claude(f"Answer this question about the project by searching the dated note files "
                       f"in this directory with grep/read. Quote the relevant note. {q}",
                       model=model, cwd=str(_sessions_dir(size)), tools=["Grep", "Read", "Glob"])
    if arm == "A3":
        ctx = "\n".join(sdict[i] for i in _vector_topk(size, sessions, q))
        return _claude(_ctx_prompt(q, ctx), model=model)
    if arm == "A4":
        lab = _ensure_bert_lab(size)
        r = MCP.ask_via_bert_mcp(q, lab, model)
        return (r["answer"], r["tokens_in"], r["tokens_out"])
    raise ValueError(arm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["S", "M", "L"], default="S")
    ap.add_argument("--arms", default="A0,A1,A2,A3,A4")
    ap.add_argument("--n", type=int, default=0, help="limit questions (0=all)")
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    MCP._write_mcp_config()

    gold = json.loads((OUT / f"gold_{args.size}.json").read_text())
    qs = gold["questions"][:args.n] if args.n else gold["questions"]
    arms = args.arms.split(",")
    sessions = _load_sessions(args.size)
    sdict = dict(sessions)
    if "A4" in arms:
        _ensure_bert_lab(args.size)                  # ingest once, before parallel fan-out
    if "A3" in arms:
        _vector_topk(args.size, sessions, "warm-up", k=1)  # build+cache matrix single-threaded

    ckpt = OUT / f"rows_{args.size}.jsonl"
    done = set()
    if ckpt.exists():
        for ln in ckpt.read_text().splitlines():
            try:
                r = json.loads(ln); done.add((r["id"], r["arm"]))
            except Exception:  # noqa: BLE001
                pass
    cells = [(g, a) for g in qs for a in arms if (g["id"], a) not in done]
    print(f"[m1-arms] size={args.size} | {len(qs)} Q × {arms} | {len(cells)} pending "
          f"(resume {len(done)})", flush=True)

    def work(cell):
        g, arm = cell
        try:
            ans, ti, to = answer(arm, g["question"], args.size, sessions, sdict, args.model)
        except Exception as e:  # noqa: BLE001
            return None, f"{arm}/{g['id']}: {e}"
        low = (ans or "").lower()
        if (not ans or "session limit" in low or "hit your" in low or "prompt is too long" in low
                or ans.startswith("[claude err") or ans.startswith("[mcp err")
                or (ti == 0 and to == 0)):
            return None, "transient"
        res = GR.grade_judges(g["question"], g.get("gold_answer", ""), ans)
        if res["n_valid"] == 0:
            return None, "judges-down"
        return {"id": g["id"], "category": g["category"], "size": args.size, "arm": arm,
                "correct": res["verdict"], "tokens_in": ti, "tokens_out": to,
                "answer": ans[:1500]}, None

    import os as _os
    workers = int(_os.environ.get("M1_WORKERS", "4"))
    lock = threading.Lock()
    f = ckpt.open("a"); n = 0; t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, c) for c in cells]):
            row, err = fut.result()
            if row is None:
                continue
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush(); n += 1
                if n % 10 == 0:
                    print(f"  {n}/{len(cells)} ({round(time.monotonic()-t0)}s)", flush=True)
    f.close()

    rows = [json.loads(ln) for ln in ckpt.read_text().splitlines() if ln.strip()]
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        d = agg[r["arm"]]; d[0] += r["correct"]; d[1] += 1
    print(f"\n[m1 size={args.size}] per-arm accuracy:")
    for a in arms:
        if a in agg:
            c, nn = agg[a]
            print(f"  {a:4} {c/nn:.3f} (n={nn})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
