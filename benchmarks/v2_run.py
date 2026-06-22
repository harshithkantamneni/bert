"""v2 SOTA QA-accuracy harness — the orchestrator.

Ties together every control the v1 sweep lacked:
  - closed-book arm (A0) to subtract parametric knowledge
  - deterministic, method-blind gold (v2_gold_ast) graded PROGRAMMATICALLY
  - multiple corpora (httpx+starlette / pydantic / gin-Go) of varied lang+size
  - pinned provider (recorded per call), k repeats for reader variance
  - retrieval precomputed ONCE per (corpus, question) and reused across budgets/k
  - budget sweep for the truncation/RAG arms
  - paired stats: bootstrap CIs + exact McNemar + Holm correction

Phases (run in order; each checkpoints to benchmarks/results/v2/):
  --precompute   ingest+index each corpus, precompute hybrid/vector/bm25 chunks
  --run          run the arm factorial (reader/agentic), grade, checkpoint rows
  --stats        aggregate rows -> per-arm CIs + pairwise significance -> JSON

A6 (graph/Aider) chunks are precomputed separately in the aider venv by
v2_precompute_aider.py and merged here if present.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "benchmarks" / "results" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

REF_BUDGET = 15000          # reference token budget / top_n=10 for the headline comparison
BUDGETS = [5000, 15000, 30000, 60000]
TOP_N = 10
K_REPEATS = 3
SEED = 17

# corpora that get DETERMINISTIC python gold (Go corpus = retrieval/agentic + multi-hop only)
PY_CORPORA = {"c1", "c3", "big"}
MANIFEST = REPO / "benchmarks/results/v2_corpora_manifest.json"


def load_corpora() -> list[dict]:
    """All corpora: the base set (c1/c2/c3) plus the large 'big' corpus.

    NOTE: v2_corpora.build_corpora() rewrites the manifest with only c1/c2/c3,
    which clobbers the 'big' entry that v2_big_corpus.py wrote. So we recover
    'big' from DISK directly (not the manifest), then re-persist the merged
    manifest so the report sees it too."""
    from benchmarks import v2_corpora as C
    base = C.build_corpora()
    names = {c["name"] for c in base}
    big_root = Path("/tmp/v2_corpora/big")
    if "big" not in names and big_root.exists():
        pys = list(big_root.rglob("*.py"))
        if pys:
            toks = sum((f.stat().st_size for f in pys), 0) // 4
            base.append({"name": "big", "lang": "python", "root": str(big_root),
                         "n_files": len(pys), "est_tokens": toks})
            names.add("big")
    try:
        MANIFEST.write_text(json.dumps(base, indent=2))
    except OSError:
        pass
    return base


# ── gold assembly ────────────────────────────────────────────────────
def assemble_gold(corpora: list[dict], per_corpus: int = 80, seed: int = SEED) -> list[dict]:
    """Deterministic AST gold from the python corpora (file-diversity capped) +
    method-blind multi-hop gold if benchmarks/results/v2/multihop_gold.json exists."""
    from benchmarks import v2_gold_ast as G
    rng = random.Random(seed)
    gold: list[dict] = []
    by_corpus = {c["name"]: c for c in corpora}
    for cname in PY_CORPORA:
        c = by_corpus.get(cname)
        if not c:
            continue
        facts = G.extract_gold(c["root"])
        for f in facts:
            f["corpus"] = cname
        # cap ~3 per source_file for spread, then sample per_corpus
        rng.shuffle(facts)
        per_file: dict[str, int] = {}
        picked = []
        for f in facts:
            sf = f.get("source_file", "")
            if per_file.get(sf, 0) >= 3:
                continue
            per_file[sf] = per_file.get(sf, 0) + 1
            picked.append(f)
            if len(picked) >= per_corpus:
                break
        gold.extend(picked)
    # multi-hop (method-blind generated), if present
    mh = OUT / "multihop_gold.json"
    if mh.exists():
        gold.extend(json.loads(mh.read_text()))
    # stable ids
    for i, g in enumerate(gold):
        g.setdefault("id", f"g{i:04d}")
        g["grade_mode"] = g.get("grade_mode", "programmatic" if g.get("answer_regex") or g.get("gold_answer") else "judge")
    return gold


# ── per-corpus lab + retrieval precompute ────────────────────────────
def ensure_lab(corpus: dict) -> Path:
    from core import lab_context, memory
    lab = Path("/tmp/v2_labs") / corpus["name"]
    lab.mkdir(parents=True, exist_ok=True)
    tok = lab_context.set_active_lab_path(lab)
    try:
        n = memory.ingest_corpus(Path(corpus["root"]), eager_index=True)
        print(f"  [{corpus['name']}] ingested {n} files -> {lab}", flush=True)
    finally:
        lab_context.reset_active_lab_path(tok)
    return lab


def precompute_retrieval(corpora: list[dict], gold: list[dict],
                         methods=("hybrid", "vector", "bm25")) -> None:
    """For each (corpus, question) cache the top-N chunk texts per method."""
    from benchmarks import b9_rag_runner as RR
    by_corpus = {c["name"]: c for c in corpora}
    cache: dict[str, dict] = {}
    labs = {}
    for cname in {g["corpus"] for g in gold}:
        if cname in by_corpus:
            labs[cname] = ensure_lab(by_corpus[cname])
    t0 = time.monotonic()
    for i, g in enumerate(gold, 1):
        lab = labs.get(g["corpus"])
        if lab is None:
            continue
        entry = {}
        for m in methods:
            try:
                entry[m] = [c for _id, c in RR.retrieve_for(g["question"], lab, method=m, top_n=TOP_N)]
            except Exception as e:  # noqa: BLE001
                entry[m] = []
                print(f"  [retrieve warn] {m} {g['id']}: {e}", flush=True)
        cache[g["id"]] = entry
        if i % 25 == 0:
            print(f"  precomputed {i}/{len(gold)} ({round(time.monotonic()-t0)}s)", flush=True)
    (OUT / "retrieval_cache.json").write_text(json.dumps(cache))
    print(f"  -> retrieval_cache.json ({len(cache)} questions)", flush=True)


# ── arm factorial ────────────────────────────────────────────────────
def run_factorial(corpora: list[dict], gold: list[dict], *,
                  arms: list[str], budgets: list[int], k: int,
                  frontier_model: str = "sonnet", frontier_subset: int = 60) -> None:
    from benchmarks import v2_arms as A
    from benchmarks import v2_grader as GR
    by_corpus = {c["name"]: c for c in corpora}
    reader = A.make_pinned_reader()
    rcache = json.loads((OUT / "retrieval_cache.json").read_text()) if (OUT / "retrieval_cache.json").exists() else {}
    acache = json.loads((OUT / "aider_cache.json").read_text()) if (OUT / "aider_cache.json").exists() else {}
    labs = {c["name"]: Path("/tmp/v2_labs") / c["name"] for c in corpora}
    cfiles = {c["name"]: __import__("benchmarks.b9_rag_runner", fromlist=["load_corpus_files"]).load_corpus_files(Path(c["root"])) for c in corpora}

    ckpt = OUT / "factorial_rows.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["id"], r["arm"], r["budget"], r["rep"]))
            except Exception:  # noqa: BLE001
                pass
    print(f"  resuming: {len(done)} cells already done", flush=True)

    # frontier arm only on a fixed subset (quota); deterministic-tier preferred
    fsub = {g["id"] for g in sorted(gold, key=lambda x: x["id"])[:frontier_subset]}

    # Build the pending-cell list (skip already-done), then run concurrently —
    # cells are independent and LLM-bound (I/O), so a thread pool gives a big
    # speedup over the serial loop. The checkpoint append is lock-guarded.
    _MTH = {"A3": "hybrid", "A4": "vector", "A5": "bm25"}
    cells = []
    for g in gold:
        cname = g["corpus"]
        corpus_root = Path(by_corpus[cname]["root"]) if cname in by_corpus else None
        for arm in arms:
            arm_budgets = budgets if arm in ("A1", "A2") else [REF_BUDGET]
            arm_k = 1 if arm == "A7f" else k
            if arm == "A7f" and g["id"] not in fsub:
                continue
            for budget in arm_budgets:
                for rep in range(arm_k):
                    if (g["id"], arm, budget, rep) in done:
                        continue
                    pre = None
                    if arm in _MTH:
                        pre = (rcache.get(g["id"], {}) or {}).get(_MTH[arm], [])
                    elif arm == "A6":
                        pre = (acache.get(g["id"], {}) or {}).get("graph")
                        if pre is None:
                            continue
                    cells.append({"g": g, "arm": arm, "budget": budget, "rep": rep,
                                  "pre": pre, "corpus_root": corpus_root, "cname": cname})
    total = len(cells)
    print(f"  {total} pending cells (parallel)", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _worker(cell):
        g, arm = cell["g"], cell["arm"]
        try:
            res = A.run_arm(arm, g["question"], corpus_root=cell["corpus_root"],
                            lab=labs.get(cell["cname"]), corpus_files=cfiles.get(cell["cname"], []),
                            budget_tokens=cell["budget"], reader=reader, top_n=TOP_N,
                            frontier_model=frontier_model, precomputed_chunks=cell["pre"])
        except Exception as e:  # noqa: BLE001
            res = {"arm": arm, "answer": f"[error: {e}]", "provider": "err",
                   "input_tokens": 0, "latency_ms": 0, "cost_usd": 0.0, "steps": None}
        if g.get("grade_mode") == "programmatic":
            correct = GR.grade_programmatic(res["answer"], g.get("gold_answer", ""), g.get("answer_regex"))
            judge_meta = None
        else:
            jr = GR.grade_judges(g["question"], g.get("gold_answer", ""), res["answer"])
            correct = jr["verdict"]; judge_meta = jr
        return {"id": g["id"], "corpus": cell["cname"], "tier": g.get("tier"),
                "arm": arm, "budget": cell["budget"], "rep": cell["rep"], "correct": correct,
                "grade_mode": g.get("grade_mode"), "provider": res["provider"],
                "input_tokens": res["input_tokens"], "latency_ms": res["latency_ms"],
                "cost_usd": res.get("cost_usd", 0.0), "steps": res.get("steps"),
                "answer": res["answer"][:240], "judge": judge_meta}

    import os as _os
    workers = int(_os.environ.get("BERT_FACTORIAL_WORKERS", "6"))
    lock = threading.Lock()
    f = ckpt.open("a")
    t0 = time.monotonic()
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_worker, c) for c in cells]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush()
                n += 1
                if n % 25 == 0:
                    print(f"  {n}/{total} cells ({round(time.monotonic()-t0)}s)", flush=True)
    f.close()
    print(f"  factorial done: {n} new cells", flush=True)


# ── stats aggregation ────────────────────────────────────────────────
def compute_stats() -> None:
    from benchmarks import v2_stats as S
    rows = [json.loads(l) for l in (OUT / "factorial_rows.jsonl").read_text().splitlines() if l.strip()]
    # headline: ref budget, collapse k-repeats by majority (or mean) per (id,arm)
    ref = [r for r in rows if r["budget"] == REF_BUDGET or r["arm"] in ("A0", "A7w", "A7f")]
    # per (arm,id): mean correctness over reps -> binarize at 0.5 for paired tests; keep mean for acc
    from collections import defaultdict
    cell = defaultdict(list)
    for r in ref:
        cell[(r["arm"], r["id"])].append(r["correct"])
    import numpy as np
    arms = sorted({a for a, _ in cell})
    # collapse k-repeats per (arm,id): binary (mean>=0.5) for paired tests, float mean for std.
    # Per-arm accuracy uses each arm's OWN ids; pairwise tests align PER PAIR on shared ids —
    # so A7f (run on a subset) never collapses the headline n for the other arms.
    collapsed: dict[str, dict[str, int]] = {a: {} for a in arms}
    meanmap: dict[str, dict[str, float]] = {a: {} for a in arms}
    for (a, i), vals in cell.items():
        m = sum(vals) / len(vals)
        collapsed[a][i] = 1 if m >= 0.5 else 0
        meanmap[a][i] = m
    arm_stats = {}
    for a in arms:
        vec = list(collapsed[a].values())
        rstd = float(np.std(list(meanmap[a].values()))) if meanmap[a] else 0.0
        d = S.arm_stat(a, vec, repeat_std=rstd).__dict__
        d["n_questions"] = len(vec)
        arm_stats[a] = d
    # pairwise: align on ids BOTH arms answered (per-pair pairing — correct for McNemar)
    pairs = []
    npaired = {}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            shared = sorted(set(collapsed[a]) & set(collapsed[b]))
            if len(shared) < 5:
                continue
            va = [collapsed[a][x] for x in shared]
            vb = [collapsed[b][x] for x in shared]
            pt = S.pair_test(a, va, b, vb)
            pairs.append(pt)
            npaired[(a, b)] = len(shared)
    S.holm_bonferroni(pairs)
    pair_out = [{"a": p.arm_a, "b": p.arm_b, "diff": p.diff, "diff_ci": p.diff_ci,
                 "p": p.p_value, "p_holm": p.p_holm, "sig": p.significant,
                 "b_wins": p.c, "a_wins": p.b, "n_paired": npaired.get((p.arm_a, p.arm_b))}
                for p in pairs]
    # by tier + by corpus accuracy (descriptive)
    def acc_by(keyf):
        d = defaultdict(lambda: defaultdict(list))
        for r in ref:
            d[r["arm"]][keyf(r)].append(r["correct"])
        return {a: {k: round(sum(v) / len(v), 3) for k, v in kv.items()} for a, kv in d.items()}
    # budget sweep (truncation arms)
    sweep = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] in ("A1", "A2"):
            sweep[r["arm"]][r["budget"]].append(r["correct"])
    sweep_out = {a: {b: round(sum(v) / len(v), 3) for b, v in bv.items()} for a, bv in sweep.items()}
    # cost/latency per arm
    cl = defaultdict(lambda: {"lat": [], "cost": []})
    for r in ref:
        cl[r["arm"]]["lat"].append(r["latency_ms"]); cl[r["arm"]]["cost"].append(r["cost_usd"])
    import numpy as np
    costlat = {a: {"p50_latency_ms": float(np.percentile(d["lat"], 50)) if d["lat"] else 0,
                   "p95_latency_ms": float(np.percentile(d["lat"], 95)) if d["lat"] else 0,
                   "mean_cost_usd": float(np.mean(d["cost"])) if d["cost"] else 0} for a, d in cl.items()}
    out = {"n_by_arm": {a: arm_stats[a]["n_questions"] for a in arms},
           "arms": arms, "arm_stats": arm_stats, "pairwise": pair_out,
           "acc_by_tier": acc_by(lambda r: r["tier"]), "acc_by_corpus": acc_by(lambda r: r["corpus"]),
           "budget_sweep": sweep_out, "cost_latency": costlat}
    (OUT / "stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"n_by_arm": out["n_by_arm"], "arms": arms,
                      "accuracy": {a: round(arm_stats[a]["accuracy"], 3) for a in arms}}, indent=2))
    print("\nsignificant pairs (Holm < .05):")
    for p in pair_out:
        if p["sig"]:
            print(f"  {p['a']} vs {p['b']}: diff={p['diff']:+.3f} CI=({p['diff_ci'][0]:+.2f},{p['diff_ci'][1]:+.2f}) p_holm={p['p_holm']:.4f}")
    print("-> stats.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--per-corpus", type=int, default=80)
    ap.add_argument("--regen-gold", action="store_true", help="force-regenerate gold.json (otherwise reuse the frozen one)")
    ap.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7w", "A7f"])
    ap.add_argument("--k", type=int, default=K_REPEATS)
    args = ap.parse_args()

    corpora = load_corpora()  # includes the big corpus (recovered from disk)
    # FREEZE the gold set: assemble_gold is not bit-reproducible across processes
    # (set-iteration + traversal order), so regenerating per phase drifted the
    # question set and broke cache alignment (RAG arms got empty context). Generate
    # ONCE, persist, and reuse on every later phase so all arms run identical Qs.
    gp = OUT / "gold.json"
    if gp.exists() and not args.regen_gold:
        gold = json.loads(gp.read_text())
        print(f"gold: loaded {len(gold)} FROZEN questions from {gp.name}", flush=True)
    else:
        gold = assemble_gold(corpora, per_corpus=args.per_corpus)
        gp.write_text(json.dumps(gold, indent=2))
        print(f"gold: generated {len(gold)} questions "
              f"({sum(1 for g in gold if g.get('grade_mode')=='programmatic')} programmatic / "
              f"{sum(1 for g in gold if g.get('grade_mode')=='judge')} judge)", flush=True)

    if args.precompute:
        precompute_retrieval(corpora, gold)
    if args.run:
        run_factorial(corpora, gold, arms=args.arms, budgets=BUDGETS, k=args.k)
    if args.stats:
        compute_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
