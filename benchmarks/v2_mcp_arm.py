"""v2 literal-MCP arm — the truest realistic test of bert.

Runs Claude (via `claude -p`) with bert's ACTUAL MCP server attached, so Claude
calls bert's `memory_search` tool LIVE (exactly as a real Claude Code user would)
to answer each question. Compared against A7f (the same Claude with grep/read
tools), this is the honest, model-controlled, deployment-realistic comparison:

    Claude + bert (MCP memory_search)   vs   Claude + grep

Same model both sides; only the retrieval method differs. This is what actually
decides whether Claude should call bert or just grep.

Run AFTER the free-tier factorial (uses Max-plan Claude; the factorial's A7f
cells also use Claude, so don't overlap). Sequential — each call spawns one bert
MCP server (single MPS process; concurrent servers would deadlock the GPU).

  .venv/bin/python benchmarks/v2_mcp_arm.py [n_per_corpus] [model]
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "benchmarks/results/v2"

from benchmarks import v2_grader as GR  # noqa: E402

MCP_CONFIG = "/tmp/bert_mcp.json"
TOOL = "mcp__bert__bert_memory_search"


def _write_mcp_config():
    # bert server on CPU so multiple parallel claude+server processes don't
    # deadlock the GPU (concurrent MPS contexts hang). CPU is slower per call but
    # parallel-safe — and retrieval is a small slice of each turn.
    cfg = {"mcpServers": {"bert": {
        "command": str(REPO / ".venv/bin/python"),
        "args": [str(REPO / "tools/mcp/bert_lab.py")],
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "BERT_EMBED_DEVICE": "cpu", "BERT_RERANKER_DEVICE": "cpu"}}}}
    Path(MCP_CONFIG).write_text(json.dumps(cfg))


def ask_via_bert_mcp(question: str, lab_path: str, model: str = "sonnet") -> dict:
    """Claude answers by calling bert's memory_search MCP tool. Returns
    {answer, turns, tokens_in, tokens_out, used_tool}."""
    prompt = (f"Use the bert_memory_search tool (lab='{lab_path}') to search the "
              f"indexed project for the answer, then answer. Quote the exact value "
              f"from the source. Question: {question}")
    try:
        p = subprocess.run(
            ["claude", "-p", "--model", model, "--output-format", "json",
             "--mcp-config", MCP_CONFIG, "--strict-mcp-config",
             "--permission-mode", "bypassPermissions", "--allowedTools", TOOL],
            input=prompt, capture_output=True, text=True, timeout=360)
        out = json.loads(p.stdout)
        ti = to = 0
        for mu in (out.get("modelUsage") or {}).values():
            ti += (mu.get("inputTokens", 0) + mu.get("cacheReadInputTokens", 0)
                   + mu.get("cacheCreationInputTokens", 0))
            to += mu.get("outputTokens", 0)
        return {"answer": out.get("result", "") or "", "turns": out.get("num_turns", 0),
                "tokens_in": ti, "tokens_out": to, "used_tool": out.get("num_turns", 0) >= 2}
    except Exception as e:  # noqa: BLE001
        return {"answer": f"[mcp err: {e}]", "turns": 0, "tokens_in": 0, "tokens_out": 0, "used_tool": False}


def main() -> int:
    npc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    model = sys.argv[2] if len(sys.argv) > 2 else "sonnet"
    _write_mcp_config()
    gold = json.loads((OUT / "gold.json").read_text())
    prog = [g for g in gold if g.get("grade_mode") == "programmatic"]
    by = collections.defaultdict(list)
    for g in prog:
        by[g["corpus"]].append(g)
    subset = [g for gs in by.values() for g in gs[:npc]]

    ckpt = OUT / "mcp_arm_rows.jsonl"
    done = set()
    if ckpt.exists():
        for ln in ckpt.read_text().splitlines():
            try:
                done.add(json.loads(ln)["id"])
            except Exception:  # noqa: BLE001
                pass
    print(f"[mcp-arm] model={model} {len(subset)} Q (resume {len(done)} done) — "
          f"Claude calling bert's memory_search live", flush=True)
    # Parallel: each claude+bert-server is its own CPU process (server on CPU per
    # the config), so concurrent spawns don't deadlock the GPU. Bounded for Max.
    import os as _os
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cells = [g for g in subset if g["id"] not in done]
    workers = int(_os.environ.get("BERT_MCP_WORKERS", "3"))

    def _work(g):
        r = ask_via_bert_mcp(g["question"], f"/tmp/v2_labs/{g['corpus']}", model)
        corr = GR.grade_programmatic(r["answer"], g.get("gold_answer", ""), g.get("answer_regex"))
        return {"id": g["id"], "corpus": g["corpus"], "arm": "A_mcp", "correct": corr,
                "turns": r["turns"], "used_tool": r["used_tool"],
                "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"], "answer": r["answer"][:200]}

    lock = threading.Lock()
    f = ckpt.open("a"); t0 = time.monotonic(); n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_work, g) for g in cells]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush(); n += 1
                print(f"  {n}/{len(cells)} {row['corpus']} {'✓' if row['correct'] else '✗'} "
                      f"turns={row['turns']} tok={row['tokens_in']}+{row['tokens_out']}", flush=True)
    f.close()

    rows = [json.loads(ln) for ln in ckpt.read_text().splitlines() if ln.strip()]
    nc = sum(r["correct"] for r in rows)
    tool_rate = sum(1 for r in rows if r.get("used_tool")) / len(rows) if rows else 0
    tin = sum(r["tokens_in"] for r in rows) / len(rows) if rows else 0
    tout = sum(r["tokens_out"] for r in rows) / len(rows) if rows else 0
    summary = {"arm": "A_mcp", "model": model, "n": len(rows),
               "accuracy": round(nc / len(rows), 3) if rows else 0,
               "tool_call_rate": round(tool_rate, 3),
               "tokens_in_per_q": round(tin), "tokens_out_per_q": round(tout),
               "tokens_per_correct": round((tin + tout) * len(rows) / nc) if nc else None}
    (OUT / "mcp_arm.json").write_text(json.dumps(summary, indent=2))
    print(f"\nA_mcp (Claude + bert via MCP): acc={summary['accuracy']} "
          f"tool_rate={summary['tool_call_rate']} tok/q={round(tin+tout)} "
          f"tok/correct={summary['tokens_per_correct']}")
    print(f"-> {OUT/'mcp_arm.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
