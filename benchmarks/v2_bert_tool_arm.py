"""v2 llama + bert-via-tool arm (the Tier-1 mirror of the Claude MCP arm).

Gives the FREE model (llama-3.3-70b) bert's `memory_search` as a native tool —
the model decides when to call it and with what query, exactly like the Claude
MCP arm, but on the free tier. The tool executes bert's REAL hybrid_retrieve
(the same engine the MCP server exposes), so this is "llama using bert" the way
the Claude arm is "Claude using bert". Sums real tokens across the loop.

Reusable: run_bert_tool_answer() is imported by the tokenomics pass too.

  .venv/bin/python benchmarks/v2_bert_tool_arm.py [n_per_corpus]
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "benchmarks/results/v2"

from benchmarks import v2_arms as A  # noqa: E402
from benchmarks import v2_grader as GR  # noqa: E402

TOP_N = 10
MAX_STEPS = 4
_BERT_TOOL = [{"type": "function", "function": {
    "name": "memory_search",
    "description": "Search the indexed project codebase for relevant code/chunks. "
                   "Returns the top matching chunks (bert hybrid retrieval).",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "what to look for"}},
        "required": ["query"]}}}]


def _exec_memory_search(query: str, lab_path: str) -> str:
    """bert's REAL hybrid retrieval — same engine the MCP memory_search exposes."""
    from core import lab_context, retrieval
    tok = lab_context.set_active_lab_path(Path(lab_path))
    try:
        res = retrieval.hybrid_retrieve(query, top_n=TOP_N)
        out = []
        for i, r in enumerate(res):
            content = (r.metadata or {}).get("content") or r.text or ""
            out.append(f"[chunk {i+1}]\n{content[:1500]}")
        return "\n\n".join(out) if out else "(no results)"
    except Exception as e:  # noqa: BLE001
        return f"(memory_search error: {e})"
    finally:
        lab_context.reset_active_lab_path(tok)


def run_bert_tool_answer(question: str, lab_path: str, cascade=None, max_steps=MAX_STEPS) -> dict:
    """llama answers by calling bert's memory_search tool. Returns
    {answer, steps, tokens_in, tokens_out, used_tool}."""
    from core import provider as prov
    cascade = cascade or A.FREE_READER
    sysp = ("You answer a factual question about a codebase. Call memory_search to "
            "retrieve relevant code, then answer with the exact value from it. "
            f"You have at most {max_steps} searches.")
    messages = [{"role": "system", "content": sysp},
                {"role": "user", "content": f"Question: {question}"}]
    usage = {"in": 0, "out": 0}
    steps = 0
    used = False
    for _ in range(max_steps):
        resp = None
        for pn, m in cascade:
            try:
                r = prov.call(pn, messages, model=m, tools=_BERT_TOOL,
                              max_tokens=700, temperature=0.0, timeout=70.0)
                if r.finish_reason != "error" and not (r.text or "").startswith("[bert]"):
                    usage["in"] += getattr(r, "usage_prompt_tokens", 0) or 0
                    usage["out"] += getattr(r, "usage_completion_tokens", 0) or 0
                    resp = r
                    break
            except Exception:  # noqa: BLE001
                continue
        if resp is None:
            return {"answer": "[bert-tool: lanes errored]", "steps": steps,
                    "tokens_in": usage["in"], "tokens_out": usage["out"], "used_tool": used}
        if resp.tool_calls:
            used = True
            messages.append({"role": "assistant", "content": resp.text or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                            for tc in resp.tool_calls]})
            for tc in resp.tool_calls:
                obs = _exec_memory_search(str((tc.arguments or {}).get("query", "")), lab_path)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs[:6000]})
            steps += 1
            continue
        return {"answer": (resp.text or "").strip(), "steps": steps,
                "tokens_in": usage["in"], "tokens_out": usage["out"], "used_tool": used}
    # budget hit: force answer
    messages.append({"role": "user", "content": "Answer now with your best single answer."})
    for pn, m in cascade:
        try:
            r = prov.call(pn, messages, model=m, max_tokens=400, temperature=0.0, timeout=60.0)
            if r.finish_reason != "error" and not (r.text or "").startswith("[bert]"):
                usage["in"] += getattr(r, "usage_prompt_tokens", 0) or 0
                usage["out"] += getattr(r, "usage_completion_tokens", 0) or 0
                return {"answer": (r.text or "").strip(), "steps": steps,
                        "tokens_in": usage["in"], "tokens_out": usage["out"], "used_tool": used}
        except Exception:  # noqa: BLE001
            continue
    return {"answer": "[bert-tool: no answer]", "steps": steps,
            "tokens_in": usage["in"], "tokens_out": usage["out"], "used_tool": used}


def main() -> int:
    npc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    gold = json.loads((OUT / "gold.json").read_text())
    prog = [g for g in gold if g.get("grade_mode") == "programmatic"]
    by = collections.defaultdict(list)
    for g in prog:
        by[g["corpus"]].append(g)
    subset = [g for gs in by.values() for g in gs[:npc]]
    ckpt = OUT / "bert_tool_rows.jsonl"
    done = set()
    if ckpt.exists():
        for ln in ckpt.read_text().splitlines():
            try:
                done.add(json.loads(ln)["id"])
            except Exception:  # noqa: BLE001
                pass
    print(f"[bert-tool/llama] {len(subset)} Q (resume {len(done)}) — llama calling bert memory_search", flush=True)
    # Parallel: bert retrieval runs on CPU here (postprocess sets BERT_*_DEVICE=cpu),
    # so concurrent threads don't deadlock the GPU; the llama calls are network-bound.
    import os as _os
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cells = [g for g in subset if g["id"] not in done]
    workers = int(_os.environ.get("BERT_LLAMA_WORKERS", "5"))

    def _work(g):
        r = run_bert_tool_answer(g["question"], f"/tmp/v2_labs/{g['corpus']}")
        corr = GR.grade_programmatic(r["answer"], g.get("gold_answer", ""), g.get("answer_regex"))
        return {"id": g["id"], "corpus": g["corpus"], "arm": "A_mcp_llama",
                "correct": corr, "steps": r["steps"], "used_tool": r["used_tool"],
                "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"], "answer": r["answer"][:200]}

    lock = threading.Lock()
    f = ckpt.open("a"); n = 0; t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_work, g) for g in cells]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush(); n += 1
                if n % 5 == 0:
                    print(f"  {n}/{len(cells)} ({round(time.monotonic()-t0)}s)", flush=True)
    f.close()
    rows = [json.loads(ln) for ln in ckpt.read_text().splitlines() if ln.strip()]
    nc = sum(r["correct"] for r in rows)
    tin = sum(r["tokens_in"] for r in rows) / len(rows) if rows else 0
    tout = sum(r["tokens_out"] for r in rows) / len(rows) if rows else 0
    summary = {"arm": "A_mcp_llama", "n": len(rows),
               "accuracy": round(nc / len(rows), 3) if rows else 0,
               "tool_call_rate": round(sum(1 for r in rows if r.get("used_tool")) / len(rows), 3) if rows else 0,
               "tokens_in_per_q": round(tin), "tokens_out_per_q": round(tout),
               "tokens_per_correct": round((tin + tout) * len(rows) / nc) if nc else None}
    (OUT / "bert_tool_arm.json").write_text(json.dumps(summary, indent=2))
    print(f"\nA_mcp_llama: acc={summary['accuracy']} tool_rate={summary['tool_call_rate']} "
          f"tok/q={round(tin+tout)} tok/correct={summary['tokens_per_correct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
