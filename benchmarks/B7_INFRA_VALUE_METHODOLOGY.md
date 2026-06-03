<!-- Pre-registration: frozen 2026-06-01 before any Opus call. Designed via a 6-agent adversarial methodology workflow (recon -> 3x design -> critique -> synthesize); the critique's 8 must-fix confounds are folded in. Do not edit to fit results; deviations go in the limitations section. -->

# B7 — Infrastructure-Value A/B Benchmark Methodology (LOCKED, pre-registration)

_Platform: M3 Pro 18GB unified memory, Python 3.13. Repo: `/path/to/Desktop/bert-lab`._

> **Pre-registration discipline.** Everything in this file is fixed *before* any Opus call. Every weight, prompt, selection rule, scrub regex, and decision threshold is frozen in `config` of the output JSON. No post-hoc tuning. If a choice must change after seeing data, it is reported as a deviation in the limitations section, not silently applied.

## 0. The question and the invariant

We measure **whether bert's orchestration infrastructure improves deliverable quality enough to justify its overhead, and on which workloads.** The model is held constant (**Opus in both arms**, via `claude -p --model opus`), and the grader is held constant (the **free-tier 4-judge cascade** in `core/grader.py:48`, which is `llama-3.3-70b` on groq/cerebras/nvidia + `gemini-2.0-flash` — **not Opus**, so there is no Opus-grades-Opus circularity; verified `core/grader.py:48-53`). The **only** variable is bert's infrastructure: role roster, verification spec, memory/retrieval, the gaps mandate, and cycle iteration.

## 1. The fairness problem and why this is a 3-arm design (NOT 2)

The adversarial audit verified a **critical confound**: bert's dispatch injects `_render_verification_requirements(verification_spec)` (`core/subagent.py:225-244`, wired at `tools/bert_run.py:715-721`) as a block literally headed **"## Deliverable requirements (you are GRADED on these — satisfy ALL)"**. Its contents (the DEFAULT_SPEC at `core/verify_engine.py:64-87`: `min_chars=1500`, ≥1 H1 + ≥3 H2 headers, a citation regex `https?://|arxiv:|doi\.org|github\.com|… et al`, forbid `example.com`/`TBD`) are a near-paraphrase of the grader's scored dimensions (`grading_rubric.yaml`: provenance = "every claim traces to a primary source", completeness = "covers what the mission asked", usability = "well-structured"). **bert is handed the answer key; the baseline is not.** A 2-arm result would confound "orchestration value" with "we told one arm the rubric."

**Mitigation (must-fix): run THREE arms per workload instance.**

| Arm | What it is | Isolates |
|---|---|---|
| **A — bare** | one `claude -p --model opus` call, plain task + capability floor (§3) | the honest user baseline |
| **B — baseline+rubric** | identical to A **plus** the exact `_render_verification_requirements(DEFAULT_SPEC)` block bert injects, verbatim | the rubric-disclosure floor |
| **C — full bert** | `tools/bert_run.py … --model anthropic-cli/opus` (roster + verify + memory + gaps + cycles) | full bert |

Report all three. The deltas decompose the effect cleanly:
- **`B − A`** = the rubric-leakage / teaching-to-the-test component (a one-line baseline addition replicates it).
- **`C − B`** = the **genuine orchestration value** (roster, memory, cycle iteration, enforcement) with rubric leakage netted out. **This is the headline number.**
- **`C − A`** = the total bert advantage as a naive 2-arm benchmark would have (mis)reported it; we show it only to demonstrate how much of it is leakage.

The C-vs-B comparison is the contribution; if `C − B` is near zero while `B − A` is large, the honest finding is "bert's quality lift is mostly rubric-disclosure, not orchestration," and we report that.

## 2. Held-constant fairness controls (the fairness ledger)

| Dimension | A (bare) | B (baseline+rubric) | C (full bert) | Held constant? |
|---|---|---|---|---|
| Model | `claude -p --model opus` | same | every dispatch `claude -p --model opus` | **YES** (verified: `_anthropic_cli_model_flag` bare→opus, `tools/bert_run.py:601-612`) |
| Grader + contract | free-tier 4-judge cascade, contract X (built once) | same | same | **YES** |
| Toolset | `Write,Read,Edit,WebSearch,WebFetch,Bash` | same | same | **YES** (`tools/bert_run.py:742`) |
| Budget cap | `--max-budget-usd 2.0`, timeout 900s | same | `2.0` per dispatch | **YES** per call |
| Tool-awareness + anti-fabrication norm | stated (capability floor §3) | stated | stated (role constitution) | **YES** (effect equalized) |
| Graded-requirements block | **none** | **injected verbatim** | injected | **isolated by the 3-arm split** |
| Honest-limitations ask | identical ≥3-bullet ask (§4) | identical | mandated ≥3-bullet, **enforced + iterated** | floor held constant; only ENFORCEMENT is bert-only |
| Task seed | `workload_seed.md` | identical | identical `seed_brief.md` | **YES** (byte-identical) |
| Role roster / memory / cycles | none | none | researcher→strategist→… | **bert-only (under test)** |

## 3. Capability floor (held constant, all arms)

The variable is orchestration, not tool availability. Withholding tool-knowledge would strawman A/B on provenance (a scored dimension). So all three arms get the identical neutral capability statement in the prompt: the tools exist and claims should cite real, non-fabricated sources. What stays bert-only is everything role-shaped and rubric-shaped (the "you are bert's {role}" constitution, falsifiers, success criteria, memory) — except the rubric block itself, which is deliberately given to B to isolate its effect.

## 4. The gaps/honesty asymmetry (must-fix mitigation)

`grade_artifact(artifact, gaps)` feeds the gaps text to the `honesty` and `gap_finder` judges (`core/grader.py:131-137`). bert mandates a ≥3-bullet `_gaps.md` ("empty fails the cycle", `tools/bert_run.py:216-224`); `validate_gaps` floors "no known gaps" at 0-1. If A/B are graded with `gaps=""`, bert wins two of eight weighted dimensions for free — that is the gaps *checklist*, not orchestration.

**Resolution (held constant, not asymmetric):** All three arms receive the **identical** instruction to produce a `## Limitations` section with **≥3 bullets** of what they could not verify. The harness extracts that section as the `gaps` input for A/B and reads bert's `_gaps.md` for C. The honesty *floor is held constant*; what stays bert-only is **enforcement + cross-cycle iteration** (bert re-prompts when gaps are thin; A/B get one shot). If an A/B run omits the section, pass `gaps=""` honestly — never synthesize one — and flag it.

**Secondary auditability axis (pulled OUT of the headline score):** We additionally report `grader.validate_gaps(gaps_text).score` per arm as a *separate* "auditability" axis. bert is expected to dominate it regardless of cost; reporting it separately keeps it from silently inflating the weighted quality comparison.

## 5. Blind, paired grading protocol

1. Produce `(artifact, gaps)` for all three arms of an instance.
2. **Scrub arm fingerprints** identically from all three artifacts before grading: regex-strip `bert_run_C\d+`, role banners (`^You are bert's …`, `^# Researcher`/`Strategist` headers), cycle-id lines, and any literal "bert". The scrub set is frozen in `config.scrub_regexes` and is **audited before runs** (§9 TDD) on 2-3 real `findings/*.md` pairs: a human (or a separate model) must fail to guess the arm above chance; if guessable, expand the scrub set (likely the gaps-bullet skeleton and H1+3×H2 structure). **Note:** the scrub cannot neutralize length/structure — that is handled by the 3-arm design (B matches bert's length/structure floor) and the length-control probe (§6.4).
3. Grade each scrubbed artifact **independently** (one `grade_artifact` call per artifact → no judge-context bleed) with the shared contract.
4. **Re-grade `K=3` times** and take the per-dimension median-of-medians to damp the non-deterministic cascade. Store all K raw `GradeResult.to_dict()`.
5. **Grade all arms of a pair back-to-back in the same time window** so free-tier judge availability is matched within a pair (mitigates asymmetric dropped-judge degradation across time).
6. Use a **neutral judge system prompt** for the benchmark ("You are an expert evaluator …"), overriding the default `f"You are bert's {judge} judge"` (`core/grader.py:120`) so the house-judge framing is removed; confirm on a held-out set that calibration is unchanged. This is done by passing a benchmark-local rubric/persona wrapper, not by editing `core/grader.py` for production.
7. Arm labels are hidden behind a seeded RNG (`random.Random(master_seed)`) until aggregation.

**Dropped-judge handling:** record which judges answered per grade. `≥2` dropped → flag low-confidence and re-run that grade. All-4-dropped (`aggregate` returns all-zeros) → **grader outage, exclude from quality aggregation but log** (it is not an arm defect). Consider BYO **metered keys for the judges** (judges ≠ model under test, so this does not break the invariant) to remove quota-driven drops; record judge-availability stats per tier.

## 6. Overhead accounting (cost is tokens + wall-clock, NOT dollars)

**Critical correction (must-fix): the Opus bridge runs on the user's Max-plan OAuth session** (`tools/bert_run.py:687-688`), so `total_cost_usd` is **imputed list price, not marginal spend**, and bert's deliberate prompt-cache warmup (`~75%` input-token deflation, `tools/bert_run.py:698-732`) biases the imputed-dollar ratio in bert's favor while the single-call baseline gets no cache benefit. Therefore:

### 6.1 Primary overhead metrics = TOKENS + WALL-CLOCK
- `tokens_in` (report **gross** and **cache-net** both), `tokens_out`, `model_calls`, `latency_wall_secs`.
- Dollars are reported **only** as a clearly-labeled "list-price estimate" with the cache caveat, never as the crossover axis. The justification threshold `τ` is defined **in tokens and seconds**, not dollars (§7). For any dollar claim, compute cost from token counts × a stated public price table, **identically for both arms, with cache-read priced explicitly** (or re-run a cost arm against a metered key); never use the imputed Max-plan number for comparison.

### 6.2 Arm A/B (one call): parse `cli_out = json.loads(result.stdout)`
- `tokens_in = usage.input_tokens + cache_creation_input_tokens + cache_read_input_tokens`; also record net (`input_tokens` only).
- `tokens_out = usage.output_tokens`; `latency_secs = duration_ms/1000`; `model_calls = 1`; `session_id`.
- Guards (same as bert, `tools/bert_run.py:754-775`): `returncode==0`; JSON parses; `is_error` False; artifact exists with `size>=100`. Any guard fail → record `failed_to_produce=True`, `weighted_score=0.0`, `passes=False` (do **not** drop — silent drops bias the comparison).

### 6.3 Arm C (bert, many calls) — run-isolated telemetry (must-fix)
`run()` returns an **int exit code, not a summary dict** (verified `tools/bert_run.py:965`), and `model_call.jsonl` is a **shared append-only file with a free-running `cycle` id and no run-id/cost field** (verified: sample row `cycle:123`, `core/observability.py:46`). So:
- **Invoke bert as a subprocess** (`.venv/bin/python tools/bert_run.py --lab <fresh_lab> --max-cycles N --model anthropic-cli/opus`), capture stdout/stderr, and parse the per-dispatch telemetry blocks bert prints (each carries `telemetry.cost_usd` for the Opus list-price estimate).
- **Run-isolate the telemetry** rather than trusting cycle filtering: set a **per-run private `OBS_DIR`** via env (`BERT_OBS_DIR`/equivalent) OR snapshot the **byte-offset of `model_call.jsonl` immediately before the run and read only the appended tail**. Then **assert** the tail row-count equals the known dispatch count (roster × cycles, + judge/memory calls bert itself makes) — no foreign rows. **Serialize** the benchmark: no other bert process / nightly job during the window, and run the *grading* in a **separate process with its own OBS_DIR** so the benchmark's judge calls never land in bert's runtime tail.
- Tokens / model-call-count / model-time from the isolated tail; the Opus list-price estimate (clearly labeled) from summed per-dispatch `telemetry.cost_usd`. **Never mix sources for the same metric** (tokens from JSONL, $ from per-dispatch CLI).
- **Two latency numbers, report both:** `latency_wall_secs` = full subprocess wall-clock (the honest user-facing number, includes orchestration + free-tier judge + reranker + KG writes) and `latency_model_secs` = Σ row `elapsed_ms/1000`. **Report `latency_wall − latency_model` as a named "harness/memory/reranker overhead" line item** so bert's non-model cost (embedding, bge-reranker +12×, KG writes, verify passes) is not undercounted. **Count retries explicitly** from the dispatch telemetry `retry_count` rather than assuming 0.

### 6.4 Length-bias control probe (must-fix)
LLM judges reward longer/structured text. The 3-arm design already length-matches B to bert. Additionally: (a) regress `weighted_score` on `tokens_out` across all graded artifacts and report the length slope; (b) run a one-time **calibration probe** grading the SAME content short-vs-padded and short-vs-headered to quantify the judges' length/structure sensitivity; (c) include an **adversarial padding control** (verbose-but-vacuous artifact) and confirm it does NOT outscore a terse substantive one — if it does, the grader is not measuring quality and we say so.

### 6.5 Two bert configurations (report both)
`BERT_FORCE_MODEL=anthropic-cli/opus` / `--model anthropic-cli/opus` pins **every** role to Opus (verified bare→opus). Report:
- **C-pure** (all-Opus): the scientific control for the model-held-constant infra delta. **This is the headline for the quality claim.**
- **C-real** (router-default tiering): the realistic product's cost/quality (light roles tier down). **This is the headline for the overhead/crossover-for-a-real-user claim.**
One number cannot do both jobs; state which claim rests on which.

## 7. Workload-justification analysis (the deliverable)

Per instance, derive (using **C-pure** for quality, **token/second overhead** for cost):
```
quality_gain_orch = C.weighted_score − B.weighted_score   # headline: orchestration value, leakage netted out
quality_gain_total = C.weighted_score − A.weighted_score  # naive 2-arm number (for contrast only)
leakage           = B.weighted_score − A.weighted_score   # rubric-disclosure component
overhead_token_ratio = C.tokens_total / A.tokens_total
overhead_lat_ratio   = C.latency_wall / A.latency_secs
gain_per_extra_ktoken = quality_gain_orch / ((C.tokens_total − A.tokens_total)/1000)
gain_per_extra_second = quality_gain_orch / (C.latency_wall − A.latency_secs)
```

**Justification crossover (token/second thresholds, not dollars):**
```
justified(tier) := gain_per_extra_ktoken(tier) >= tau_ktoken
                   AND lower_bootstrap_CI(quality_gain_orch(tier)) > 0   # gain must be real, not noise
crossover_tier  := smallest tier where justified holds AND holds for all higher tiers
```
`tau_ktoken`/`tau_seconds` are **logged policy knobs**, not constants. If no tier qualifies, report **"no crossover within tested range"** — an honest null, not a forced answer.

**Decision rule (numbers filled from the run, not asserted):**
> Use bert when the task (a) requires synthesis across ≥2 sources/sub-analyses, OR (b) is one where overclaiming is costly (correctness/provenance/honesty weighted high), AND you tolerate ~`{measured overhead_lat_ratio}`× latency and ~`{measured overhead_token_ratio}`× tokens. Use raw Opus for single-shot lookups (T0/T1 class) where the orchestration gain's CI includes zero. Separately: bert wins the **auditability** axis (`validate_gaps`) at all tiers — choose it whenever a checkable gaps trail matters regardless of overhead.

## 8. Statistics (honest about small n)

- **Paired design:** each instance yields a matched A/B/C triple → use paired statistics (controls for workload difficulty; far more powerful at small n).
- **Primary report = median + IQR** per tier per arm (means are fragile at n=3-9; weighted_score is bounded/skewed). Report min/max too.
- **Quality gain:** median of paired differences `Δ_i` + **bootstrap percentile CI** (B=10,000). At n=3 the CI is wide — that width is the honest signal.
- **Effect size:** **Cliff's δ** (primary, robust at small n; |δ|<0.147 negligible … >0.474 large) on the per-tier `C` vs `B` scores; Cohen's d_z secondary with the explicit caveat that sd(Δ) is unstable at n<10. Report the δ **trend across tiers** (does orchestration value grow T0→T3?).
- **Significance honesty:** with n=3, the two-sided Wilcoxon p-floor is ~0.25 — **no per-tier comparison can reach p<0.05**. State this; report the sign test / Cliff's δ trend as the primary evidence, **never** a misleading p. The entire result is labeled **directional / exploratory** in the headline, not just the appendix.
- **Variance is a confidence channel, not a quality signal:** surface mean `overall_variance` and dropped-judge counts per arm per tier; if bert's richer artifacts draw systematically higher judge disagreement, note it as a confound on the gain CI.

## 9. Workload selection bias mitigation (must-fix)

Prompts must NOT ask for bert's native artifact shapes by name. **Use generic framings** ("research X and recommend", "analyze Y") and let each arm choose structure. **Do not** put "pre-registered falsifiers", "ranked matrix", "lineage" in the ask unless all arms get it identically. Include at least one **anti-bert** workload per low tier (a task wanting a single terse answer, where decomposition should *hurt*) to test the predicted T0 null honestly. **Disclose workload authorship** in the methodology section; ideally a disinterested party authors them blind to bert's roles. The full prompt set is frozen in `config` before runs.

## 10. Terminal-artifact selection (must-fix, pre-registered rule)
bert produces multiple gradeable artifacts; grading all would multiply its samples. **Pre-register the rule: the bert deliverable = the last-cycle terminal role's artifact, selected deterministically by name (`findings/bert_run_C{max_cycle}_{terminal_role}.md`), regardless of its score.** `terminal_role` is frozen in `config` (default `strategist`; for organic rosters, the last dispatch in cycle order). **Never select by grade.** Report what the other bert artifacts scored, for transparency, so a reviewer can confirm no cherry-picking.

## 11. T3 baseline-fairness construction (pre-registered)
A single A/B call cannot "do 3 cycles". **Pre-register and run both variants:** (a) **information-matched** — A/B get a *neutral human-written summary* of prior stages (NOT bert's own artifacts — that would be circular), isolating the lineage/contradiction machinery; (b) **cold single-shot** — A/B re-derive everything, isolating full memory value. Report them as two distinct claims. **Never paste bert's own artifacts into A/B.**

## 12. Independence hygiene
Scaffold a **fresh, empty `BASELINE_OUT_DIR` per (tier, instance, arm, repeat)** and a fresh lab dir per bert run; assert empty before each call so the Read/Edit tools cannot see a prior artifact. `split_on_limitations` must be deterministic (final-header match with documented fallback) and unit-tested on real artifacts.

## 13. Outputs (repo convention)
`benchmarks/results/ab_infra_<TIMESTAMP>.json` (`config` with all frozen knobs + `platform` + `timestamp` + `results` per-row list + `derived` per-tier analysis) paired with `ab_infra_summary_<TIMESTAMP>.md` (leads with title, `_Generated:_`, `_Platform: M3 Pro 18GB …_`, then `## Methodology` stating the 3-arm definition verbatim, `## Quality by tier`, `## Overhead by tier`, `## Workload justification`, mandatory `## Honest limitations (POPPER-style)`, `## Reproducing this run` fenced bash). Extend `benchmarks/b6_compile_report.py:main()` to `latest("ab_infra_*.json")` into `REPORT.md`.

## 14. Honest limitations (POPPER-style) — mandatory in every summary
1. **Power:** n=3-9/tier; no per-tier comparison reaches p<0.05 (Wilcoxon floor ~0.25). Directional/exploratory, not confirmatory.
2. **Grader noise:** free-tier llama/gemini, non-deterministic cascades; K=3 + median mitigates, but absolute scores carry ±`overall_variance`.
3. **Length/structure bias:** quantified by the §6.4 probe and the regression slope; residual bias is disclosed.
4. **Single contract:** one weighting drives the 0-1 collapse; we report a **weight-sensitivity analysis** (2-3 alternative defensible contracts incl. usability/efficiency-weighted and equal-weight) and state whether the crossover survives. Cross-tier comparison assumes one contract.
5. **Cost is plan-dependent:** Opus `$` is imputed Max-plan list price; we report tokens+seconds as primary and label any `$` as estimate with the cache caveat.
6. **No claim bert > raw Opus in general** — only "orchestration (C − B) is justified above tier X under threshold τ for this contract." Negative/zero gain at low tiers is reported as-is.
