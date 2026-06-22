"""v2 Claude-only track — every arm on ONE model (Claude). Runs over any gold
file, so it serves both the CODE track (exact-fact gold) and the SEMANTIC track
(conceptual gold). Single-model => the only variable is the retrieval method.

Arms (all Claude sonnet):
  A0 closed-book | A1 naive-trunc | A2 smart-trunc | A3 bert-hybrid-RAG |
  A4 vector | A5 bm25 | A6 graph/Aider | A7grep (Claude+grep) | A_mcp (Claude
  calls bert's memory_search LIVE over MCP)

Programmatic grading when the gold has answer_regex (code track); multi-judge
(non-Claude) when grade_mode='judge' (semantic track). Captures tokens for
tokenomics. Parallel; bert retrieval on CPU (set by the driver) so parallel
calls don't collide on the GPU.

  .venv/bin/python benchmarks/v2_claude_track.py --gold <f> --cache <f> \
      --acache <f> --out <rows.jsonl> [--n-per-corpus N] [--model sonnet]
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
OUT = REPO / "benchmarks/results/v2"

from benchmarks import b9_rag as RAG  # noqa: E402
from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import v2_arms as A  # noqa: E402
from benchmarks import v2_grader as GR  # noqa: E402
from benchmarks import v2_mcp_arm as MCP  # noqa: E402
from benchmarks.v2_tokenomics import _claude  # claude -p w/ tokens (+tools/cwd)  # noqa: E402

_MTH = {"A3": "hybrid", "A4": "vector", "A5": "bm25"}
CTX = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]


def _prompt(arm, g, corpus_files, rcache, acache):
    if arm == "A0":
        return A._closed_book_prompt(g["question"])
    if arm in ("A1", "A2"):
        c = RAG.build_context(arm, corpus_files=corpus_files, retrieved_chunks=[], budget_tokens=15000)
        return RAG.reader_prompt(g["question"], c)
    if arm in ("A3", "A4", "A5"):
        return RAG.reader_prompt(g["question"], A._rag_context((rcache.get(g["id"], {}) or {}).get(_MTH[arm], [])))
    if arm == "A6":
        return RAG.reader_prompt(g["question"], A._rag_context((acache.get(g["id"], {}) or {}).get("graph", [])))
    raise ValueError(arm)


def _grade(g, ans):
    if g.get("grade_mode") == "programmatic":
        return GR.grade_programmatic(ans, g.get("gold_answer", ""), g.get("answer_regex"))
    res = GR.grade_judges(g["question"], g.get("gold_answer", ""), ans)
    # judges all unreachable -> DON'T record a fake 0; signal retry
    return res["verdict"] if res["n_valid"] > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--acache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-corpus", type=int, default=0, help="0 = all")
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    MCP._write_mcp_config()

    gold = json.loads(Path(args.gold).read_text())
    if args.n_per_corpus:
        by = collections.defaultdict(list)
        for g in gold:
            by[g["corpus"]].append(g)
        gold = [g for gs in by.values() for g in gs[:args.n_per_corpus]]
    rcache = json.loads(Path(args.cache).read_text()) if Path(args.cache).exists() else {}
    acache = json.loads(Path(args.acache).read_text()) if Path(args.acache).exists() else {}
    man = {c["name"]: c for c in json.loads((REPO / "benchmarks/results/v2_corpora_manifest.json").read_text())}
    cfiles = {c: RR.load_corpus_files(Path(man[c]["root"])) for c in {g["corpus"] for g in gold} if c in man}

    ckpt = Path(args.out)
    done = set()
    if ckpt.exists():
        for ln in ckpt.read_text().splitlines():
            try:
                r = json.loads(ln); done.add((r["id"], r["arm"]))
            except Exception:  # noqa: BLE001
                pass
    arms = CTX + ["A7grep", "A_mcp"]
    cells = [(g, arm) for g in gold for arm in arms if (g["id"], arm) not in done]
    print(f"[claude-track] {args.model} | {len(gold)} Q x {len(arms)} arms | "
          f"{len(cells)} pending (resume {len(done)})", flush=True)

    def work(cell):
        g, arm = cell
        croot = str(Path(man[g["corpus"]]["root"])) if g["corpus"] in man else None
        if arm == "A7grep":
            ans, ti, to = _claude(f"Answer this about the code in this directory by searching with "
                                  f"grep/read. {g['question']}", model=args.model, cwd=croot,
                                  tools=["Grep", "Read", "Glob"])
        elif arm == "A_mcp":
            r = MCP.ask_via_bert_mcp(g["question"], f"/tmp/v2_labs/{g['corpus']}", args.model)
            ans, ti, to = r["answer"], r["tokens_in"], r["tokens_out"]
        else:
            ans, ti, to = _claude(_prompt(arm, g, cfiles.get(g["corpus"], []), rcache, acache), model=args.model)
        # don't checkpoint a transient failure / quota wall — let resume retry it
        low = (ans or "").lower()
        if (not ans or ans.startswith("[claude err") or ans.startswith("[mcp err")
                or ans.startswith("[bert]") or "session limit" in low or "usage limit" in low
                or "hit your" in low or (ti == 0 and to == 0)):
            return None
        grade = _grade(g, ans)
        if grade is None:  # judges unreachable — retry on resume rather than fake-zero
            return None
        return {"id": g["id"], "corpus": g["corpus"], "tier": g.get("tier"), "arm": arm,
                "correct": grade, "tokens_in": ti, "tokens_out": to, "answer": ans[:1500]}

    workers = int(__import__("os").environ.get("BERT_CLAUDE_WORKERS", "4"))
    lock = threading.Lock()
    f = ckpt.open("a"); n = 0; t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, c) for c in cells]):
            row = fut.result()
            if row is None:  # transient failure — not checkpointed, retried on resume
                continue
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush(); n += 1
                if n % 15 == 0:
                    print(f"  {n}/{len(cells)} ({round(time.monotonic()-t0)}s)", flush=True)
    f.close()

    rows = [json.loads(ln) for ln in ckpt.read_text().splitlines() if ln.strip()]
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # correct, n, tin, tout
    for r in rows:
        d = agg[r["arm"]]; d[0] += r["correct"]; d[1] += 1; d[2] += r["tokens_in"]; d[3] += r["tokens_out"]
    print(f"\n[claude-track {Path(args.out).stem}] per-arm accuracy + tokens/q:")
    for a in arms:
        if a in agg:
            c, n_, ti, to = agg[a]
            print(f"  {a:7} acc={c/n_:.3f} (n={n_})  tok/q={round((ti+to)/n_)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
