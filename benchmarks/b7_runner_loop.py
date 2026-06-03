"""B7 real-run orchestration loop — executes the 3 arms per workload instance on
real Opus, grades blindly, aggregates, and writes the JSON+MD report. Imported
lazily by b7_ab_infra.main(); validated by the pilot, not the offline unit tests.

The pure, testable helpers live in b7_ab_infra / b7_stats; this file is the glue
that touches subprocess + the grader + the filesystem.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks import b7_ab_infra as R  # noqa: E402
from benchmarks import b7_stats as stats  # noqa: E402

# Pre-registered grading contract (methodology §2/§8). Frozen here; echoed into
# the output config so a reviewer can re-weight and re-derive.
CONTRACT_WEIGHTS = {"correctness": 5, "completeness": 4, "provenance": 4,
                    "defensibility": 4, "usability": 3, "honesty": 4,
                    "reproducibility": 3, "efficiency": 3}
PASS_THRESHOLD = 0.7
TERMINAL_ROLE = "strategist"      # pre-registered deterministic selection rule
BENCH_TMP = REPO / "benchmarks" / ".b7_runs"   # fresh per-arm scaffolds (gitignored)

# STRONG, NON-CLAUDE judge panel (WS1b). The B7 pilot's free-tier llama judges
# compressed every score into 0.85-0.95 — 70B llamas round everything to "looks
# fine" = 4-5/5. These frontier non-Claude models actually use the bottom of the
# scale (the design workflow probed them live: DeepSeek gave a 0, Mistral a 2).
# Arms are all Claude-family, so the judge MUST be non-Claude (assert below).
# Ordered for reliability: Mistral anchors (fast, no 429); Gemini last (free tier
# 429s in hot loops). All 4 judge personas share this cascade; they differ by
# persona prompt, not by model.
STRONG_JUDGE_CASCADE = [
    ("mistral", "mistral-large-latest"),
    ("openrouter", "deepseek/deepseek-v4-pro"),
    ("openrouter", "z-ai/glm-5.1"),
    ("gemini", "gemini-2.5-pro"),
]
R.assert_non_claude_cascade(STRONG_JUDGE_CASCADE)   # structural anti-circularity guard


def _contract():
    from core import quality
    return quality.QualityContract(**CONTRACT_WEIGHTS, pass_threshold=PASS_THRESHOLD)


def _load_workload(path: Path) -> dict:
    """A workload file is markdown; an optional leading HTML comment carries
    metadata (tier/max_cycles/kind). The remaining body is the seed task."""
    text = path.read_text()
    meta = {"tier": path.stem.split("_")[0].upper(), "max_cycles": 1,
            "kind": "generic"}
    # Strip ALL leading HTML comments; the first one carries key:value metadata,
    # later ones (e.g. provenance) are ignored. Without this loop a second comment
    # leaks into the seed sent to the model.
    while text.lstrip().startswith("<!--"):
        head, _, text = text.lstrip().partition("-->")
        for part in head.replace("<!--", "").split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                meta[k.strip()] = v.strip()
    meta["max_cycles"] = int(meta.get("max_cycles", 1))
    meta["seed"] = text.strip()
    return meta


def _grade_artifact_file(artifact_path: Path, contract) -> dict:
    """Read a produced artifact, split off its limitations, scrub fingerprints
    from BOTH halves, and grade blindly with the neutral judge persona."""
    raw = artifact_path.read_text(encoding="utf-8", errors="replace")
    body, gaps = R.split_on_limitations(raw)
    body_s = R.scrub_fingerprints(body)
    gaps_s = R.scrub_fingerprints(gaps)
    graded = R.neutral_judge_grade(body_s, gaps_s, contract, K=3,
                                   cascade=STRONG_JUDGE_CASCADE)
    # Auditability axis, reported separately (not folded into weighted_score).
    try:
        from core import grader
        graded["gaps_audit_score"] = grader.validate_gaps(gaps_s).score
    except Exception:  # noqa: BLE001
        graded["gaps_audit_score"] = None
    return graded


def _run_baseline(arm: str, seed: str, contract, *, master_seed: int,
                  tier: str, instance: str) -> dict:
    out_dir = BENCH_TMP / f"{tier}_{instance}_{arm}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    abs_out = out_dir / "artifact.md"
    prompt = R.build_arm_prompt(seed, arm, str(abs_out))
    cmd = R.build_baseline_cmd(str(out_dir), prompt)
    tele = R.run_baseline_arm(cmd, abs_out)
    row = {"tier": tier, "instance": instance, "arm": arm, **tele}
    if not tele["failed_to_produce"]:
        graded = _grade_artifact_file(abs_out, contract)
        row.update({"weighted_score": graded["weighted_score"],
                    "passes": graded["passes"], "medians": graded["medians"],
                    "variances": graded["variances"],
                    "overall_variance": graded["overall_variance"],
                    "dropped": graded["dropped"],
                    "gaps_audit_score": graded.get("gaps_audit_score")})
    row["tokens_total"] = row.get("tokens_in_net", 0) + row.get("tokens_out", 0)
    return row


def _c_telemetry_from_summary(summary: dict) -> dict:
    disp = [d for c in summary.get("cycles", []) for d in c.get("dispatches", [])]
    tin = sum((d.get("telemetry") or {}).get("tokens_in", 0) for d in disp)
    tout = sum((d.get("telemetry") or {}).get("tokens_out", 0) for d in disp)
    lat_model = sum((d.get("telemetry") or {}).get("latency_secs", 0.0) for d in disp)
    cost = sum((d.get("telemetry") or {}).get("cost_usd", 0.0) for d in disp)
    retry = sum((d.get("telemetry") or {}).get("retry_count", 0) for d in disp)
    wall = summary.get("wall_secs", lat_model)
    return {"model_calls": len(disp), "tokens_in_net": tin, "tokens_out": tout,
            "latency_model_secs": lat_model, "latency_wall_secs": wall,
            "harness_overhead_secs": max(0.0, wall - lat_model),
            "cost_usd_estimate": cost, "retry_count": retry,
            "providers": sorted({(d.get("telemetry") or {}).get("provider")
                                 for d in disp if d.get("telemetry")})}


def _run_bert(seed: str, contract, *, tier: str, instance: str,
              max_cycles: int, force_model: str, legacy: bool) -> dict:
    lab = BENCH_TMP / f"{tier}_{instance}_C_lab"
    if lab.exists():
        shutil.rmtree(lab)
    lab.mkdir(parents=True)
    (lab / "seed_brief.md").write_text(seed)
    summary_path = lab / "_run_summary.json"

    env = dict(os.environ)
    env["BERT_RUN_SUMMARY_PATH"] = str(summary_path)
    if legacy:
        env["BERT_LEGACY_RESEARCHER_STRATEGIST"] = "1"
    cmd = R.build_bert_cmd(str(lab), max_cycles, force_model=force_model)

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    wall = round(time.monotonic() - t0, 1)

    row = {"tier": tier, "instance": instance, "arm": "C",
           "bert_exit_code": proc.returncode}
    # telemetry from the run-summary dump (bridge token/latency live there)
    if summary_path.exists():
        import json
        tele = _c_telemetry_from_summary(json.loads(summary_path.read_text()))
    else:
        tele = {"model_calls": 0, "tokens_in_net": 0, "tokens_out": 0,
                "latency_model_secs": 0.0, "latency_wall_secs": wall,
                "harness_overhead_secs": 0.0, "cost_usd_estimate": 0.0,
                "retry_count": 0, "providers": []}
    tele["latency_wall_secs"] = wall   # authoritative subprocess wall-clock
    row.update(tele)

    # terminal artifact: newest findings/bert_run_C*_{TERMINAL_ROLE}.md (by cycle)
    cands = sorted(lab.glob(f"findings/bert_run_C*_{TERMINAL_ROLE}.md"),
                   key=lambda p: int("".join(ch for ch in p.stem.split("_C")[1].split("_")[0]
                                             if ch.isdigit()) or 0))
    if not cands:
        row.update({"failed_to_produce": True, "fail_reason": "no terminal artifact",
                    "weighted_score": 0.0, "passes": False})
    else:
        row["failed_to_produce"] = False
        row["artifact_path"] = str(cands[-1])
        graded = _grade_artifact_file(cands[-1], contract)
        row.update({"weighted_score": graded["weighted_score"],
                    "passes": graded["passes"], "medians": graded["medians"],
                    "variances": graded["variances"],
                    "overall_variance": graded["overall_variance"],
                    "dropped": graded["dropped"],
                    "gaps_audit_score": graded.get("gaps_audit_score")})
    row["tokens_total"] = row.get("tokens_in_net", 0) + row.get("tokens_out", 0)
    return row


def run(args) -> int:
    contract = _contract()
    workloads_dir = Path(args.workloads_dir)
    if args.pilot:
        tiers = ["T1"]
        instances = 2
    else:
        tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
        instances = args.instances

    force_model = "anthropic-cli/opus"   # C-pure (the model-held-constant control)
    all_rows: list[dict] = []
    stat_rows: list[dict] = []
    failed_instances: list[dict] = []
    ckpt_ts = args.timestamp or _now_stamp()

    for tier in tiers:
        for j in range(1, instances + 1):
            wf = workloads_dir / f"{tier.lower()}_i{j}.md"
            if not wf.exists():
                print(f"[skip] missing workload {wf}")
                continue
            meta = _load_workload(wf)
            seed, max_cycles = meta["seed"], meta["max_cycles"]
            legacy = meta.get("roster", "legacy") == "legacy"
            print(f"\n=== {tier} i{j} ({meta.get('kind')}) — running A,B,C ===")

            a = _run_baseline("A", seed, contract, master_seed=args.master_seed,
                              tier=tier, instance=f"i{j}")
            b = _run_baseline("B", seed, contract, master_seed=args.master_seed,
                              tier=tier, instance=f"i{j}")
            c = _run_bert(seed, contract, tier=tier, instance=f"i{j}",
                          max_cycles=max_cycles, force_model=force_model, legacy=legacy)
            all_rows += [a, b, c]
            print(f"    A={a.get('weighted_score')}  B={b.get('weighted_score')}  "
                  f"C={c.get('weighted_score')}  (C tokens={c.get('tokens_total')}, "
                  f"C wall={c.get('latency_wall_secs')}s)")
            # Crash-resilience for the multi-hour sweep: persist raw per-arm rows
            # after every instance so a late failure loses nothing (metrics can be
            # re-derived offline from these, as the pilot was).
            _checkpoint(all_rows, Path(args.results_dir), ckpt_ts)

            row = R.make_stat_row(a, b, c)
            if row is None:
                failed_instances.append({
                    "tier": tier, "instance": f"i{j}",
                    "failed_arms": [arm for arm, r in (("A", a), ("B", b), ("C", c))
                                    if r.get("failed_to_produce")
                                    or r.get("weighted_score") is None]})
                print(f"    [excluded from quality] failed arms: "
                      f"{failed_instances[-1]['failed_arms']}")
                continue
            stat_rows.append(row)

    per_tier = stats.per_tier_summary(stat_rows, tau_ktoken=args.tau_ktoken,
                                      boot_seed=args.master_seed)
    # enrich per-tier with raw arm medians + c model-calls for the report table
    for tier in per_tier:
        trows = [r for r in stat_rows if r["tier"] == tier]
        per_tier[tier]["A_median"] = stats.median([r["A_score"] for r in trows])
        per_tier[tier]["B_median"] = stats.median([r["B_score"] for r in trows])
        per_tier[tier]["C_median"] = stats.median([r["C_score"] for r in trows])
        ccalls = [r["model_calls"] for r in all_rows
                  if r.get("arm") == "C" and r.get("tier") == tier]
        per_tier[tier]["c_model_calls"] = max(ccalls) if ccalls else 0

    crossover = stats.find_crossover(per_tier, tau_ktoken=args.tau_ktoken)
    derived = {"per_tier": per_tier, "crossover_tier": crossover or "no crossover in range",
               "decision_rule": _decision_rule(per_tier, crossover),
               "failed_instances": failed_instances,
               "valid_pairs": len(stat_rows)}
    config = {**{f"w_{k}": v for k, v in CONTRACT_WEIGHTS.items()},
              "pass_threshold": PASS_THRESHOLD, "terminal_role": TERMINAL_ROLE,
              "bert_config": "pure", "master_seed": args.master_seed,
              "tau_ktoken": args.tau_ktoken, "regrade_K": 3,
              "pilot": bool(args.pilot), "tiers": tiers, "instances": instances,
              "neutral_judge": True, "grader_cascade": "free-tier 4-judge default"}

    jp, mp = R.write_outputs(config, all_rows, derived,
                             results_dir=Path(args.results_dir), timestamp=ckpt_ts)
    print(f"\n[B7] wrote {jp}\n[B7] wrote {mp}")
    return 0


def _decision_rule(per_tier: dict, crossover: str | None) -> str:
    if not crossover:
        return ("No tier cleared the justification threshold with a CI excluding "
                "zero in this run — orchestration's quality gain is not yet "
                "distinguishable from noise at this n. Treat as directional.")
    s = per_tier.get(crossover, {})
    return (f"Use bert from tier {crossover} upward: orchestration gain "
            f"{s.get('quality_gain_orch_median', 0):.3f} "
            f"(CI [{s.get('ci_low', 0):.3f}, {s.get('ci_high', 0):.3f}], "
            f"Cliff's δ {s.get('cliffs_delta', 0):.2f}) at "
            f"{s.get('overhead_token_ratio', 0):.1f}× tokens, "
            f"{s.get('overhead_lat_ratio', 0):.1f}× latency. Below it, the gain's "
            f"CI includes zero — prefer raw Opus.")


def _checkpoint(all_rows: list[dict], results_dir: Path, ts: str) -> None:
    """Persist raw per-arm rows after each instance so a multi-hour-run crash
    loses nothing (metrics re-derivable offline). Best-effort, never raises."""
    try:
        import json
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"ab_infra_{ts}_partial.json").write_text(
            json.dumps({"rows": all_rows, "n_arms": len(all_rows)}, indent=2))
    except OSError as exc:
        print(f"[checkpoint warn] {exc}")


def _now_stamp() -> str:
    # local import; avoids the module-level Date restriction in workflow scripts
    from datetime import datetime
    return datetime.now().strftime("%Y%m%dT%H%M%S")
