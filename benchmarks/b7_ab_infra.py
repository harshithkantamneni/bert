"""B7 — Infrastructure-value A/B benchmark runner (3-arm).

Measures how much bert's ORCHESTRATION improves deliverable quality over raw
Opus, holding the model fixed at Opus in every arm and the grader fixed on the
free-tier 4-judge cascade. Three arms per workload instance:

    A — bare           : one `claude -p --model opus` call, plain task
    B — baseline+rubric: A + the exact verification-requirements block bert injects
    C — full bert      : tools/bert_run.py --model anthropic-cli/opus (roster+verify+memory)

so the effect decomposes as C-A = (C-B) + (B-A): orchestration value + rubric leakage.
See benchmarks/B7_INFRA_VALUE_METHODOLOGY.md for the locked, pre-registered design.

Network and subprocess are injected (`_runner`, `_grade`) so the unit tests run
offline; only main() / run_*_arm touch real Opus.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks import b7_stats as stats  # noqa: E402

PLATFORM = "M3 Pro 18GB unified memory, Python 3.13"
ALLOWED_TOOLS = "Write,Read,Edit,WebSearch,WebFetch,Bash"

# Held constant across ALL arms (the fairness ledger §2-3): the variable under
# test is orchestration, not tool availability or anti-fabrication norms.
CAPABILITY_FLOOR = (
    "You have WebSearch and WebFetch available — use them to ground every claim "
    "in a real, citable source. Cite real URLs / arXiv IDs / DOIs and authors; "
    "never invent placeholder links such as example.com, and never fabricate a "
    "citation. Each section must be substantive content, not a meta-description."
)
LIMITATIONS_ASK = (
    "End the document with a `## Limitations` section containing at least 3 "
    "bullets naming specific things you could not verify, sources you could not "
    "access, or assumptions that limit confidence. Do not write 'no known gaps'."
)

# Headers that mark the start of a gaps/limitations section, for split_on_limitations.
_LIMITATIONS_HDR = re.compile(
    r"(?im)^#{1,6}[ \t]+(?:limitations|gaps|open questions|uncertainties|"
    r"what i (?:could not|couldn'?t) verify)\b.*$"
)

# Blinding scrub: strip bert fingerprints identically from every arm's artifact
# before grading so judges can't recognize the "bert shape" (methodology §5.2).
_SCRUB_REGEXES = [
    re.compile(r"bert_run_C\d+\w*"),                       # artifact filenames
    re.compile(r"(?im)^you are bert.s .*$"),               # role-constitution banners
    re.compile(r"(?im)^#{1,6}\s*cycle\s*\d+.*$"),          # cycle headers
    re.compile(r"(?i)\bbert\b"),                           # the literal brand
    # Model-family tells: arms are bert-Sonnet / bare-Opus / bare-Sonnet, so a
    # self-identifying line or CLI footer would leak which Claude tier wrote it
    # to the (non-Claude) judge. Strip the family + tier names + version tails.
    re.compile(r"(?i)\bclaude(?:[ -]?(?:opus|sonnet|haiku))?(?:[ -]?\d[\d.]*)?\b"),
    re.compile(r"(?i)\banthropic\b"),
    re.compile(r"(?i)\b(?:opus|sonnet|haiku)(?:[ -]?\d[\d.]*)?\b"),
]


# ── arm command + prompt construction ────────────────────────────────

def build_baseline_cmd(out_dir: str, prompt: str, *, model: str = "opus",
                       max_budget: float = 2.0) -> list[str]:
    """The raw-Opus baseline invocation. Mirrors _dispatch_via_claude_cli's flags
    EXACTLY except it carries NO --append-system-prompt — arms A/B get no role
    constitution, only the task + held-constant capability floor in the prompt."""
    return [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--add-dir", out_dir,
        "--permission-mode", "acceptEdits",
        "--allowedTools", ALLOWED_TOOLS,
        "--max-budget-usd", str(max_budget),
        prompt,
    ]


def _arm_base_prompt(seed_text: str, abs_out_path: str) -> str:
    return (
        f"Output absolute path: {abs_out_path}\n\n"
        f"--- task ---\n{seed_text}\n\n"
        f"{CAPABILITY_FLOOR}\n\n"
        f"{LIMITATIONS_ASK}"
    )


def build_arm_prompt(seed_text: str, arm: str, abs_out_path: str) -> str:
    """Arm A = base (task + capability floor + limitations ask). Arm B = A plus
    the EXACT verification-requirements block bert injects, verbatim — so B-A
    isolates rubric disclosure and C-B isolates genuine orchestration."""
    base = _arm_base_prompt(seed_text, abs_out_path)
    if arm == "A":
        return base
    if arm == "B":
        from core import subagent, verify_engine
        block = subagent._render_verification_requirements(verify_engine.DEFAULT_SPEC)
        return base + "\n\n" + block
    raise ValueError(f"unknown arm for a baseline prompt: {arm!r} (A or B)")


def build_bert_cmd(lab_dir: str, max_cycles: int, *,
                   force_model: str = "anthropic-cli/opus") -> list[str]:
    """Arm C: run bert's pipeline pinned to host Opus (C-pure) or a router-default
    model string (C-real). force_model='anthropic-cli/opus' pins every role to Opus."""
    return [
        sys.executable, str(REPO / "tools" / "bert_run.py"),
        "--lab", lab_dir,
        "--max-cycles", str(max_cycles),
        "--model", force_model,
    ]


# ── blinding: split + scrub ──────────────────────────────────────────

def split_on_limitations(text: str) -> tuple[str, str]:
    """Split an artifact into (body, gaps) at the LAST limitations-style header.
    Returns (text, '') when no such header exists. Deterministic; never raises."""
    matches = list(_LIMITATIONS_HDR.finditer(text))
    if not matches:
        return (text, "")
    start = matches[-1].start()
    return (text[:start], text[start:])


def scrub_fingerprints(text: str, regexes: list[re.Pattern] | None = None) -> str:
    """Remove bert fingerprints (filenames, role banners, cycle headers, the brand)
    so grading is blind. Idempotent and a no-op on already-clean baseline text."""
    pats = regexes if regexes is not None else _SCRUB_REGEXES
    out = text
    for pat in pats:
        if pat.pattern == r"(?i)\bbert\b":
            out = pat.sub("the system", out)
        else:
            out = pat.sub("", out)
    return out


# ── baseline (arm A/B) telemetry + guards ────────────────────────────

def parse_baseline_result(returncode: int, stdout: str, abs_out_path: Path) -> dict:
    """Parse one `claude -p` JSON result into a telemetry dict. Applies the same
    guards bert's bridge uses; any guard failure sets failed_to_produce=True with
    weighted_score=0.0 (never a silent drop — that would bias the comparison)."""
    def _failed(reason: str) -> dict:
        return {
            "failed_to_produce": True, "fail_reason": reason,
            "weighted_score": 0.0, "passes": False,
            "tokens_in_gross": 0, "tokens_in_net": 0, "tokens_out": 0,
            "latency_secs": 0.0, "model_calls": 1, "session_id": None,
            "cost_usd_estimate": 0.0, "artifact_path": str(abs_out_path),
        }

    if returncode != 0:
        return _failed(f"returncode={returncode}")
    try:
        cli = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return _failed("stdout not JSON")
    if not isinstance(cli, dict):
        return _failed("stdout JSON not an object")
    if cli.get("is_error"):
        return _failed("cli is_error")
    if not abs_out_path.exists() or abs_out_path.stat().st_size < 100:
        return _failed("artifact missing or <100 bytes")

    usage = cli.get("usage") or {}
    return {
        "failed_to_produce": False, "fail_reason": None,
        "weighted_score": None, "passes": None,
        "tokens_in_gross": (usage.get("input_tokens", 0)
                            + usage.get("cache_creation_input_tokens", 0)
                            + usage.get("cache_read_input_tokens", 0)),
        "tokens_in_net": usage.get("input_tokens", 0),
        "tokens_out": usage.get("output_tokens", 0),
        "latency_secs": cli.get("duration_ms", 0) / 1000.0,
        "model_calls": 1,
        "session_id": cli.get("session_id"),
        "cost_usd_estimate": cli.get("total_cost_usd", 0.0),
        "artifact_path": str(abs_out_path),
    }


def run_baseline_arm(cmd: list[str], abs_out_path: Path, *,
                     timeout: float = 900.0, _runner=subprocess.run) -> dict:
    """Execute one baseline arm and parse its telemetry. _runner is injected for
    tests; in production it is subprocess.run."""
    try:
        proc = _runner(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return parse_baseline_result(124, "", abs_out_path)
    return parse_baseline_result(proc.returncode, proc.stdout, abs_out_path)


# ── bert (arm C) telemetry summation + run isolation ─────────────────

def summarize_bert_telemetry(tail_rows: list[dict], dispatch_costs: list[float],
                             wall_secs: float, expected_dispatches: int,
                             retry_counts: list[int] | None = None) -> dict:
    """Sum the run-isolated model_call.jsonl tail into bert's overhead. row_count_ok
    asserts the captured rows equal the expected dispatch count (no foreign rows
    leaked into the shared append-only log). harness_overhead = wall - model time
    is reported as a named line item so bert's non-model cost (embedding, reranker,
    KG writes, verify passes) is not undercounted."""
    model_calls = len(tail_rows)
    tokens_in_net = sum(r.get("input_tokens", 0) for r in tail_rows)
    tokens_in_gross = sum(
        r.get("input_tokens", 0) + r.get("cache_creation_input_tokens", 0)
        + r.get("cache_read_input_tokens", 0) for r in tail_rows)
    tokens_out = sum(r.get("output_tokens", 0) for r in tail_rows)
    latency_model = sum(r.get("elapsed_ms", 0) for r in tail_rows) / 1000.0
    return {
        "model_calls": model_calls,
        "expected_dispatches": expected_dispatches,
        "row_count_ok": model_calls == expected_dispatches,
        "tokens_in_net": tokens_in_net,
        "tokens_in_gross": tokens_in_gross,
        "tokens_out": tokens_out,
        "latency_model_secs": latency_model,
        "latency_wall_secs": wall_secs,
        "harness_overhead_secs": max(0.0, wall_secs - latency_model),
        "cost_usd_estimate": sum(dispatch_costs),
        "retry_count": sum(retry_counts or []),
    }


# ── neutral grading (reuses core.grader unmodified) ──────────────────

def NEUTRAL_JUDGE_PROMPT_FN(judge: str, rubric: dict) -> str:  # noqa: N802
    """A neutral evaluator persona for the benchmark — removes the production
    'You are bert's {judge} judge' house-framing (methodology §5.6) while keeping
    the same lens + rubric anchors. Passed to grade_artifact(system_prompt_fn=…)."""
    from core import grader, quality
    dims = ", ".join(quality.DIMENSIONS)
    lens = grader._PERSONAS.get(judge, "")
    try:
        rubric_block = grader._rubric_text(rubric)
    except Exception:  # noqa: BLE001 — a partial rubric (e.g. in tests) still yields a prompt
        rubric_block = ""
    return (
        f"You are an expert, impartial evaluator. Lens: {lens}. Grade the artifact "
        f"on ALL 8 dimensions ({dims}) 0-5 using the rubric anchors below — score "
        f"every dimension, including ones outside your lens; an all-identical score "
        f"is almost never real. Return ONLY a JSON object with an int 0-5 per "
        f"dimension plus a short \"rationale\".\n\nRUBRIC:\n{rubric_block}"
    )


def neutral_judge_grade(artifact: str, gaps: str, contract, *, K: int = 3,  # noqa: N803
                        cascade=None, _grade=None) -> dict:
    """Grade an artifact K times with the neutral persona and collapse to a
    median-of-medians (damps the non-deterministic free-tier cascade). Reuses
    core.grader.grade_artifact unmodified via its system_prompt_fn hook."""
    if _grade is None:
        from core import grader
        _grade = grader.grade_artifact
    regrades = []
    for _ in range(K):
        res = _grade(artifact, gaps, contract=contract, cascade=cascade,
                     system_prompt_fn=NEUTRAL_JUDGE_PROMPT_FN)
        regrades.append(res)
    ws = stats.median([r.weighted_score for r in regrades])
    dim_keys = list(regrades[0].medians.keys())
    medians = {d: stats.median([r.medians.get(d, 0) for r in regrades]) for d in dim_keys}
    variances = {d: stats.median([r.variances.get(d, 0.0) for r in regrades]) for d in dim_keys}
    dropped = sorted({j for r in regrades for j in getattr(r, "dropped", [])})
    return {
        "weighted_score": ws,
        "medians": medians,
        "variances": variances,
        "overall_variance": stats.median([getattr(r, "overall_variance", 0.0)
                                          for r in regrades]),
        "passes": ws >= getattr(contract, "pass_threshold", 0.7),
        "dropped": dropped,
        "regrades": [r.to_dict() for r in regrades],
    }


# ── outputs (repo convention) ────────────────────────────────────────

def _summary_markdown(config: dict, derived: dict, timestamp: str) -> str:
    per_tier = derived.get("per_tier", {})
    lines = [
        "# B7 — Infrastructure-Value A/B Benchmark",
        "",
        f"_Generated: {timestamp}_",
        f"_Platform: {PLATFORM}_",
        "",
        "## Methodology",
        "3-arm, model held constant at Opus, grader held constant on the free-tier "
        "4-judge cascade. **A** = bare baseline, **B** = baseline + the exact "
        "verification-requirements block bert injects, **C** = full bert. "
        "Headline = orchestration value **C − B** (rubric leakage **B − A** netted "
        "out). Cost = tokens + wall-clock (Opus $ is an imputed Max-plan estimate). "
        "Pre-registered in `benchmarks/B7_INFRA_VALUE_METHODOLOGY.md`.",
        "",
        "## Quality by tier",
        "| tier | n | A (bare) | B (+rubric) | C (bert) | leakage B−A | orch C−B | CI(C−B) | Cliff's δ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tier, s in sorted(per_tier.items()):
        ci = f"[{s.get('ci_low', 0):.3f}, {s.get('ci_high', 0):.3f}]"
        lines.append(
            f"| {tier} | {s.get('n_pairs', 0)} | "
            f"{s.get('A_median', float('nan')):.3f} | {s.get('B_median', float('nan')):.3f} | "
            f"{s.get('C_median', float('nan')):.3f} | {s.get('leakage_median', 0):.3f} | "
            f"{s.get('quality_gain_orch_median', 0):.3f} | {ci} | "
            f"{s.get('cliffs_delta', 0):.3f} ({s.get('cliffs_band', '?')}) |"
        )
    lines += [
        "",
        "## Overhead by tier (tokens + wall-clock)",
        "| tier | token ratio C/A | latency ratio C/A | C model-calls | gain per extra ktoken |",
        "|---|---|---|---|---|",
    ]
    for tier, s in sorted(per_tier.items()):
        lines.append(
            f"| {tier} | {s.get('overhead_token_ratio', 0):.2f}× | "
            f"{s.get('overhead_lat_ratio', 0):.2f}× | {s.get('c_model_calls', '?')} | "
            f"{s.get('gain_per_extra_ktoken', 0):.4f} |"
        )
    lines += [
        "",
        "## Workload justification",
        f"Crossover tier: **{derived.get('crossover_tier', 'no crossover in range')}** "
        f"(smallest tier where orchestration gain clears τ and its bootstrap CI > 0, "
        f"holding for all higher tiers).",
        "",
        derived.get("decision_rule", ""),
        "",
        "## Honest limitations (POPPER-style)",
        "- Small n: no per-tier comparison reaches p<0.05 (Wilcoxon floor ~0.25); "
        "results are directional/exploratory, effect reported as Cliff's δ + bootstrap CI.",
        "- Grader is non-deterministic free-tier llama/gemini; K=3 median mitigates "
        "but absolute scores carry ±overall_variance.",
        "- Length/structure bias: judges may reward bert's longer artifacts; the "
        "3-arm design length-matches B and we report the length slope.",
        "- Cost $ is imputed Max-plan list price, not marginal spend; tokens+seconds "
        "are the primary overhead axis.",
        "- No claim that bert > raw Opus in general — only that orchestration (C−B) "
        "is justified above the crossover tier for this contract.",
        "",
        "## Reproducing this run",
        "```bash",
        f"cd {REPO}",
        ".venv/bin/python benchmarks/b7_ab_infra.py \\",
        f"  --master-seed {config.get('master_seed', 0)} --bert-config "
        f"{config.get('bert_config', 'pure')}",
        "```",
    ]
    return "\n".join(lines)


def assert_non_claude_cascade(cascade) -> None:
    """Guard: the producer arms are all Claude-family (bert-Sonnet, bare-Opus,
    bare-Sonnet), so the judge must NOT be Claude/Anthropic or same-family
    stylistic affinity would bias the comparison. Fail the run structurally if
    any judge lane is Anthropic."""
    for prov, model in cascade:
        blob = f"{prov} {model}".lower()
        if "anthropic" in blob or "claude" in blob:
            raise ValueError(
                f"judge cascade lane {prov}/{model} is Claude-family; the arms "
                "are Claude-family so the judge must be non-Claude (circularity).")


def _pairwise_verdict(doc1: str, doc2: str, *, cascade, max_tokens: int = 300) -> str:
    """Ask one judge which document is better. Returns '1', '2', or 'tie'.
    doc1/doc2 are presented as DOCUMENT 1 / DOCUMENT 2 (already blinded)."""
    from core import provider as _prov
    sys_p = (
        "You are an impartial expert evaluator comparing two documents that answer "
        "the SAME task. Decide which is genuinely better on correctness, "
        "completeness, evidence, and usefulness — not which is longer or more "
        "confident. If they are truly equal, say tie, but prefer to pick a winner. "
        "Return ONLY JSON: {\"better\": \"1\" | \"2\" | \"tie\", \"reason\": \"...\"}."
    )
    user_p = f"DOCUMENT 1:\n{doc1}\n\nDOCUMENT 2:\n{doc2}\n\nWhich is better? JSON only."
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]
    for prov_name, model in cascade:
        try:
            resp = _prov.call(prov_name, msgs, model=model, max_tokens=max_tokens,
                              temperature=0.0, response_format={"type": "json_object"},
                              timeout=40.0)
            if resp.finish_reason == "error" or resp.text.startswith("[bert]"):
                continue
            better = str(json.loads(resp.text).get("better", "")).strip().lower()
            if better in ("1", "2", "tie"):
                return better
        except Exception:  # noqa: BLE001 — try the next lane
            continue
    return "tie"   # all lanes failed -> no signal -> tie (recorded, not dropped)


def pairwise_compare(doc_a: str, doc_b: str, *, cascade,
                     _verdict=None) -> dict:
    """Blind both docs and compare them in BOTH orders. A win counts only when
    the judge is order-consistent (picks the same doc regardless of position);
    disagreement across orders is a position-bias flip, recorded and scored a
    tie. De-compresses better than absolute 0.85-0.95 scores (methodology WS1a)."""
    vfn = _verdict or _pairwise_verdict
    a, b = scrub_fingerprints(doc_a), scrub_fingerprints(doc_b)
    v1 = vfn(a, b, cascade=cascade)        # order 1: A=DOC1, B=DOC2
    v2 = vfn(b, a, cascade=cascade)        # order 2: B=DOC1, A=DOC2
    a_pref_1 = v1 == "1"                    # A preferred in order 1
    a_pref_2 = v2 == "2"                    # A preferred in order 2 (A is DOC2)
    if a_pref_1 and a_pref_2:
        winner = "a"
    elif (v1 == "2") and (v2 == "1"):      # B preferred in both
        winner = "b"
    else:
        winner = "tie"
    return {"winner": winner,
            "order_consistent": winner != "tie",
            "order_flip": a_pref_1 != a_pref_2,
            "verdicts": [v1, v2]}


def make_stat_row(a: dict, b: dict, c: dict) -> dict | None:
    """Build the per-instance paired stat row, or None when ANY arm failed to
    produce / wasn't graded. A failed arm is a RELIABILITY outcome, not quality —
    letting its 0.0 'didn't run' into the median would corrupt the gain estimate
    (the pilot's C-median collapsed 0.88→0.44 that way). Failures are counted
    separately, never averaged into quality."""
    for arm in (a, b, c):
        if arm.get("failed_to_produce") or arm.get("weighted_score") is None:
            return None
    return {
        "tier": a["tier"], "instance": a["instance"],
        "A_score": a["weighted_score"], "B_score": b["weighted_score"],
        "C_score": c["weighted_score"],
        "A_tokens": max(1, a.get("tokens_total", 0)),
        "C_tokens": max(1, c.get("tokens_total", 0)),
        "A_latency": max(1e-6, a.get("latency_secs", 0.0)),
        "C_latency": max(1e-6, c.get("latency_wall_secs", 0.0)),
    }


def write_outputs(config: dict, results: list[dict], derived: dict, *,
                  results_dir: Path, timestamp: str) -> tuple[Path, Path]:
    """Write ab_infra_<TS>.json + ab_infra_summary_<TS>.md per the bN convention."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    obj = {"config": config, "platform": PLATFORM, "timestamp": timestamp,
           "results": results, "derived": derived}
    json_path = results_dir / f"ab_infra_{timestamp}.json"
    json_path.write_text(json.dumps(obj, indent=2))
    md_path = results_dir / f"ab_infra_summary_{timestamp}.md"
    md_path.write_text(_summary_markdown(config, derived, timestamp))
    return (json_path, md_path)


# ── orchestration (real run; exercised by the pilot, not unit tests) ──

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B7 infra-value A/B benchmark")
    ap.add_argument("--workloads-dir", default=str(REPO / "benchmarks" / "b7_workloads"))
    ap.add_argument("--results-dir", default=str(REPO / "benchmarks" / "results"))
    ap.add_argument("--tiers", default="T0,T1,T2,T3")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--pilot", action="store_true",
                    help="single tier x 2 seeds, C-pure only (harness validation)")
    ap.add_argument("--bert-config", choices=["pure", "real", "both"], default="pure")
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--tau-ktoken", type=float, default=0.0)
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args(argv)
    # The full orchestration loop lives here; it requires real Opus and is
    # validated by the pilot. Kept thin so the unit-tested helpers do the work.
    from benchmarks import b7_runner_loop  # lazy: only needed for real runs
    return b7_runner_loop.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
