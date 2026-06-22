"""v2 tokenomics — the efficiency axis, for EVERY arm in BOTH model tiers.

Measures REAL total tokens each method burns (actual prompt+completion usage,
summed across every call an arm makes — agentic/MCP arms make several), then
reports tokens/query and tokens per CORRECT answer. Covers:

  Tier 1 (llama): A0 A1 A2 A3 A4 A5 A6 A7w  A_mcp_llama
  Tier 2 (Claude): A0 A1 A2 A3 A4 A5 A6 A7f A_mcp_claude

Tokens are stable, so a small balanced subset suffices. Runs AFTER the factorial.
  .venv/bin/python benchmarks/v2_tokenomics.py [n_per_corpus]
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

from benchmarks import b9_agentic_grep as AG  # noqa: E402
from benchmarks import b9_rag as RAG  # noqa: E402
from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import v2_arms as A  # noqa: E402
from benchmarks import v2_bert_tool_arm as BT  # noqa: E402
from benchmarks import v2_grader as GR  # noqa: E402
from benchmarks import v2_mcp_arm as MCP  # noqa: E402

_MTH = {"A3": "hybrid", "A4": "vector", "A5": "bm25"}
CTX_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]


def usage_reader(max_tokens=600):
    from core import provider as prov

    def _r(prompt):
        for attempt in range(2):
            for pn, m in A.FREE_READER:
                try:
                    r = prov.call(pn, [{"role": "user", "content": prompt}], model=m,
                                  max_tokens=max_tokens, temperature=0.0, timeout=60.0)
                    if r.finish_reason != "error" and not (r.text or "").startswith("[bert]"):
                        return (r.text or "", getattr(r, "usage_prompt_tokens", 0) or 0,
                                getattr(r, "usage_completion_tokens", 0) or 0)
                except Exception:  # noqa: BLE001
                    continue
            if attempt == 0:
                time.sleep(2)
        return ("", 0, 0)
    return _r


def _claude(prompt, model="sonnet", cwd=None, tools=None):
    """claude -p; returns (answer, tokens_in, tokens_out). tools=list enables them."""
    cmd = ["claude", "-p", "--model", model, "--output-format", "json"]
    if tools:
        cmd += ["--permission-mode", "acceptEdits", "--allowedTools", *tools]
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           cwd=cwd, timeout=300)
        out = json.loads(p.stdout)
        ti = to = 0
        for mu in (out.get("modelUsage") or {}).values():
            ti += mu.get("inputTokens", 0) + mu.get("cacheReadInputTokens", 0) + mu.get("cacheCreationInputTokens", 0)
            to += mu.get("outputTokens", 0)
        return (out.get("result", "") or "", ti, to)
    except Exception as e:  # noqa: BLE001
        return (f"[claude err {e}]", 0, 0)


def _ctx_prompt(arm, g, corpus_files, rcache, acache):
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


def main() -> int:
    npc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    MCP._write_mcp_config()
    gold = json.loads((OUT / "gold.json").read_text())
    prog = [g for g in gold if g.get("grade_mode") == "programmatic"]
    by = collections.defaultdict(list)
    for g in prog:
        by[g["corpus"]].append(g)
    subset = [g for gs in by.values() for g in gs[:npc]]
    man = {c["name"]: c for c in json.loads((REPO / "benchmarks/results/v2_corpora_manifest.json").read_text())}
    rcache = json.loads((OUT / "retrieval_cache.json").read_text()) if (OUT / "retrieval_cache.json").exists() else {}
    acache = json.loads((OUT / "aider_cache.json").read_text()) if (OUT / "aider_cache.json").exists() else {}
    cfiles = {c: RR.load_corpus_files(Path(man[c]["root"])) for c in by if c in man}
    reader = usage_reader()

    agg = collections.defaultdict(lambda: {"tin": 0, "tout": 0, "n": 0, "correct": 0})
    print(f"[tokenomics] {len(subset)} Q x both tiers x all arms (parallel; bert on CPU)", flush=True)
    import os as _os
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def measure_q(g):
        """All arms for one question -> list of (tier, arm, correct, tin, tout)."""
        croot = Path(man[g["corpus"]]["root"]) if g["corpus"] in man else None
        lab = f"/tmp/v2_labs/{g['corpus']}"
        gold_a, rgx = g.get("gold_answer", ""), g.get("answer_regex")
        out = []

        def add(tier, arm, ans, ti, to):
            out.append((tier, arm, GR.grade_programmatic(ans, gold_a, rgx), ti, to))
        # Tier 1 — llama
        for arm in CTX_ARMS:
            ans, ti, to = reader(_ctx_prompt(arm, g, cfiles.get(g["corpus"], []), rcache, acache))
            add("llama", arm, ans, ti, to)
        rw = AG.agentic_grep_answer(g["question"], croot, A.FREE_READER, max_steps=8)
        add("llama", "A7w", rw["answer"], rw.get("tokens_in", 0), rw.get("tokens_out", 0))
        rb = BT.run_bert_tool_answer(g["question"], lab)
        add("llama", "A_mcp", rb["answer"], rb.get("tokens_in", 0), rb.get("tokens_out", 0))
        # Tier 2 — Claude
        for arm in CTX_ARMS:
            ans, ti, to = _claude(_ctx_prompt(arm, g, cfiles.get(g["corpus"], []), rcache, acache))
            add("claude", arm, ans, ti, to)
        ga, ti, to = _claude(f"Answer this about the code here by searching with grep/read. "
                             f"Quote the exact value. {g['question']}", cwd=str(croot),
                             tools=["Grep", "Read", "Glob"])
        add("claude", "A7grep", ga, ti, to)
        rm = MCP.ask_via_bert_mcp(g["question"], lab)
        add("claude", "A_mcp", rm["answer"], rm.get("tokens_in", 0), rm.get("tokens_out", 0))
        return out

    workers = int(_os.environ.get("BERT_TOKENOMICS_WORKERS", "3"))
    lock = threading.Lock()
    t0 = time.monotonic(); done_q = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(measure_q, g) for g in subset]):
            for tier, arm, corr, ti, to in fut.result():
                with lock:
                    d = agg[(tier, arm)]
                    d["tin"] += ti; d["tout"] += to; d["n"] += 1; d["correct"] += corr
            with lock:
                done_q += 1
                if done_q % 3 == 0:
                    print(f"  {done_q}/{len(subset)} Q ({round(time.monotonic()-t0)}s)", flush=True)

    summary = {}
    print(f"\n{'tier':7} {'arm':7} {'in/q':>7} {'out/q':>7} {'tot/q':>7} {'acc':>5} {'tok/correct':>12}")
    for (tier, arm), d in sorted(agg.items()):
        if not d["n"]:
            continue
        tin, tout = d["tin"] / d["n"], d["tout"] / d["n"]
        acc = d["correct"] / d["n"]
        tpc = round((d["tin"] + d["tout"]) / d["correct"]) if d["correct"] else None
        summary[f"{tier}/{arm}"] = {"tokens_in_per_q": round(tin), "tokens_out_per_q": round(tout),
                                    "total_per_q": round(tin + tout), "accuracy": round(acc, 3),
                                    "tokens_per_correct": tpc, "n": d["n"]}
        print(f"{tier:7} {arm:7} {tin:7.0f} {tout:7.0f} {tin+tout:7.0f} {acc:5.2f} {(tpc or 0):12}")
    (OUT / "tokenomics.json").write_text(json.dumps({"n_questions": len(subset), "summary": summary}, indent=2))
    print(f"\n-> {OUT/'tokenomics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
