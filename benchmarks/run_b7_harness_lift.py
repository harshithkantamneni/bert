"""B7 harness-lift study (Max-only, no OpenRouter): does bert's harness let a
CHEAPER model (Sonnet) match the frontier (bare Opus)?

Three arms per task, all on the Max claude -p bridge:
  bare-Opus    = claude -p --model opus   (frontier ceiling, raw)
  bare-Sonnet  = claude -p --model sonnet (cheaper tier, raw)         <- the control
  bert-Sonnet  = bert pipeline pinned to Sonnet (BERT_FORCE_MODEL)

Decomposition:
  harness_lift = bert-Sonnet - bare-Sonnet   (orchestration value at fixed tier)
  tier_gap     = bare-Opus  - bare-Sonnet    (raw Opus premium)
  win          = bert-Sonnet >= bare-Opus    (harness closes the tier gap)

Judge = strong NON-Claude, NON-OpenRouter panel (Mistral + Gemini, free tier).
Absolute weighted_score (K=3) + pairwise win-rate (both orders) for bert-Sonnet
vs each baseline. Quota-modest: ~1 Opus + ~3 Sonnet bridge calls per task.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b7_ab_infra as R  # noqa: E402

# Non-Claude AND non-OpenRouter judges (Mistral + Google free tiers). Mistral
# leads (reliable); Gemini-2.5-pro differentiates; gemini-flash + nvidia llama
# as fallbacks. assert_non_claude_cascade guards circularity (arms are Claude).
JUDGE = [("mistral", "mistral-large-latest"), ("gemini", "gemini-2.5-pro"),
         ("gemini", "gemini-2.0-flash"), ("nvidia", "meta/llama-3.3-70b-instruct")]
R.assert_non_claude_cascade(JUDGE)
TMP = REPO / "benchmarks" / ".b7hl_runs"
WORKLOADS = ["t1_i1", "t1_i2", "t2_i1"]   # substantive briefs/analysis (deep tier)


def _contract():
    from core import quality
    return quality.QualityContract(correctness=5, completeness=4, provenance=4,
                                   defensibility=4, usability=3, honesty=4,
                                   reproducibility=3, efficiency=3, pass_threshold=0.7)


def _seed(name: str) -> str:
    from benchmarks.b7_runner_loop import _load_workload
    return _load_workload(REPO / "benchmarks" / "b7_workloads" / f"{name}.md")["seed"]


def _grade(text: str, gaps: str, contract) -> float:
    g = R.neutral_judge_grade(R.scrub_fingerprints(text), R.scrub_fingerprints(gaps),
                              contract, K=3, cascade=JUDGE)
    return g["weighted_score"]


def _run_bare(seed: str, model: str, inst: str, contract) -> dict:
    abs_out = None
    tele = {"failed_to_produce": True}
    for attempt in range(2):   # one retry — the bare bridge call is flaky (no-artifact)
        out = TMP / f"{inst}_{model}_a{attempt}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        abs_out = out / "artifact.md"
        prompt = R.build_arm_prompt(seed, "A", str(abs_out))   # bare: no rubric block
        cmd = R.build_baseline_cmd(str(out), prompt, model=model)
        tele = R.run_baseline_arm(cmd, abs_out)
        if not tele["failed_to_produce"]:
            break
        print(f"    [bare-{model} {inst}] attempt {attempt+1} produced no artifact; "
              f"retrying" if attempt == 0 else "", flush=True)
    if tele["failed_to_produce"]:
        return {"arm": f"bare-{model}", "failed": True, "score": None, "text": ""}
    body, gaps = R.split_on_limitations(abs_out.read_text(encoding="utf-8", errors="replace"))
    return {"arm": f"bare-{model}", "failed": False, "score": _grade(body, gaps, contract),
            "text": body, "tokens": tele.get("tokens_in_net", 0) + tele.get("tokens_out", 0)}


def _run_bert_sonnet(seed: str, inst: str, contract) -> dict:
    lab = TMP / f"{inst}_bertsonnet_lab"
    if lab.exists():
        shutil.rmtree(lab)
    lab.mkdir(parents=True)
    (lab / "seed_brief.md").write_text(seed)
    env = dict(os.environ)
    env["BERT_FORCE_MODEL"] = "anthropic-cli/sonnet"          # pin EVERY role to Sonnet
    env["BERT_LEGACY_RESEARCHER_STRATEGIST"] = "1"
    env["BERT_RUN_SUMMARY_PATH"] = str(lab / "_summary.json")
    cmd = R.build_bert_cmd(str(lab), 1, force_model="anthropic-cli/sonnet")
    subprocess.run(cmd, capture_output=True, text=True, timeout=2400, env=env)
    cands = sorted(lab.glob("findings/bert_run_C*_strategist.md"))
    if not cands:
        return {"arm": "bert-Sonnet", "failed": True, "score": None, "text": ""}
    raw = cands[-1].read_text(encoding="utf-8", errors="replace")
    body, gaps = R.split_on_limitations(raw)
    return {"arm": "bert-Sonnet", "failed": False, "score": _grade(body, gaps, contract),
            "text": body}


def main() -> int:
    contract = _contract()
    TMP.mkdir(parents=True, exist_ok=True)
    rows = []
    for inst in WORKLOADS:
        seed = _seed(inst)
        print(f"\n=== {inst} — bare-Opus / bare-Sonnet / bert-Sonnet ===", flush=True)
        t0 = time.monotonic()
        opus = _run_bare(seed, "opus", inst, contract)
        sonnet = _run_bare(seed, "sonnet", inst, contract)
        bsonnet = _run_bert_sonnet(seed, inst, contract)
        # pairwise (both orders) bert-Sonnet vs each baseline, non-Claude judge
        pw_opus = R.pairwise_compare(bsonnet["text"], opus["text"], cascade=JUDGE) \
            if not (bsonnet["failed"] or opus["failed"]) else {"winner": "n/a"}
        pw_son = R.pairwise_compare(bsonnet["text"], sonnet["text"], cascade=JUDGE) \
            if not (bsonnet["failed"] or sonnet["failed"]) else {"winner": "n/a"}
        row = {"inst": inst, "bare_opus": opus["score"], "bare_sonnet": sonnet["score"],
               "bert_sonnet": bsonnet["score"],
               "pw_bertSonnet_vs_opus": pw_opus["winner"],
               "pw_bertSonnet_vs_sonnet": pw_son["winner"],
               "secs": round(time.monotonic() - t0)}
        rows.append(row)
        print(f"  bare-Opus={opus['score']}  bare-Sonnet={sonnet['score']}  "
              f"bert-Sonnet={bsonnet['score']}  pw(vs Opus)={pw_opus['winner']}  "
              f"({row['secs']}s)", flush=True)

    def _avg(k):
        v = [r[k] for r in rows if r[k] is not None]
        return round(sum(v) / len(v), 3) if v else None
    bo, bs, bts = _avg("bare_opus"), _avg("bare_sonnet"), _avg("bert_sonnet")
    derived = {"bare_opus_mean": bo, "bare_sonnet_mean": bs, "bert_sonnet_mean": bts,
               "harness_lift": round(bts - bs, 3) if (bts is not None and bs is not None) else None,
               "tier_gap": round(bo - bs, 3) if (bo is not None and bs is not None) else None,
               "bert_sonnet_vs_opus": round(bts - bo, 3) if (bts is not None and bo is not None) else None,
               "pw_wins_vs_opus": [r["pw_bertSonnet_vs_opus"] for r in rows]}
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out = {"arms": ["bare-Opus", "bare-Sonnet", "bert-Sonnet"], "judge": "mistral+gemini (non-Claude, non-OR)",
           "workloads": WORKLOADS, "rows": rows, "derived": derived}
    (REPO / "benchmarks" / "results" / f"b7_harness_lift_{ts}.json").write_text(json.dumps(out, indent=2))
    print("\n=== HARNESS-LIFT RESULT ===")
    print(f"  bare-Opus   {bo}")
    print(f"  bare-Sonnet {bs}")
    print(f"  bert-Sonnet {bts}")
    print(f"  harness_lift (bert-Sonnet - bare-Sonnet) = {derived['harness_lift']}")
    print(f"  tier_gap     (bare-Opus  - bare-Sonnet)  = {derived['tier_gap']}")
    print(f"  bert-Sonnet - bare-Opus = {derived['bert_sonnet_vs_opus']}  "
          f"(>=0 -> harness closes the gap)")
    print(f"  pairwise bert-Sonnet vs Opus: {derived['pw_wins_vs_opus']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
