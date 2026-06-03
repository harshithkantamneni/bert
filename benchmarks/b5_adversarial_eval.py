"""B5 — Adversarial-eval-by-design benchmark.

This IS bert's wedge per `project_bert_sota_positioning.md`. Where
conventional RAG silently fails, bert should catch the failure and
surface it via failures.md. We test the 4 RAG failure modes that
kill conventional systems in production:

  1. NEGATION   — query asks for the OPPOSITE of what the doc says
                  ("Which framework does NOT use attention?")
  2. MULTI-HOP  — answer requires combining 2+ docs
                  ("If A → B and B → C, what's A's relationship to C?")
  3. DISTRACTOR — corpus contains a LEXICAL match that's semantically
                  wrong; gold doc has the right answer but lower TF
                  ("What's Eve's salary at Stripe?" — corpus has
                  "Eve's salary at Square is X" as a distractor)
  4. CONTRADICTION — corpus contains both A and ¬A on the same topic;
                  correct retrieval needs to surface BOTH (or flag
                  the conflict)

Methods compared:
  • vector_only  — pure embedding cosine (conventional RAG baseline)
  • bert_hybrid  — hybrid retrieve + (where applicable) adversarial-eval

Output:
  benchmarks/results/b5_adversarial_<timestamp>.json
  benchmarks/results/b5_summary_<timestamp>.md

Each failure mode has 6 scenarios. Per-mode catch rate is reported.
"""

from __future__ import annotations

import json
import os
os.environ.setdefault("BERT_DISABLE_RERANKER", "1")

import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

RESULTS_DIR = LAB_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AdversarialScenario:
    scenario_id: str
    failure_mode: str       # negation | multi_hop | distractor | contradiction
    corpus: dict[str, str]
    query: str
    correct_doc_ids: list[str]      # gold answers
    distractor_doc_ids: list[str]   # docs that look right but aren't
    description: str        # why this is adversarial


def make_scenarios() -> list[AdversarialScenario]:
    out: list[AdversarialScenario] = []

    # ── NEGATION (6 scenarios) ───────────────────────────────────
    out.append(AdversarialScenario(
        scenario_id="neg_01", failure_mode="negation",
        corpus={
            "d1": "Transformer architectures rely heavily on self-attention.",
            "d2": "Mamba does NOT use attention; it uses selective state space.",
            "d3": "BERT uses bidirectional attention layers.",
        },
        query="Which architecture does NOT use attention?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Negation: vector models often miss 'NOT', returning d1 or d3.",
    ))
    out.append(AdversarialScenario(
        scenario_id="neg_02", failure_mode="negation",
        corpus={
            "d1": "Python supports list comprehensions for concise iteration.",
            "d2": "JavaScript does NOT have native list comprehensions.",
            "d3": "Haskell has list comprehensions inspired by set-builder notation.",
        },
        query="Which language lacks list comprehensions?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Vector models retrieve lexically similar; 'lacks'≈'does not have'.",
    ))
    out.append(AdversarialScenario(
        scenario_id="neg_03", failure_mode="negation",
        corpus={
            "d1": "REST APIs are stateful.",
            "d2": "REST is fundamentally stateless by design.",
            "d3": "gRPC uses HTTP/2 streaming for stateful connections.",
        },
        query="Is REST stateful or stateless?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1"],
        description="d1 is a common misconception; vector models may rank it high.",
    ))
    out.append(AdversarialScenario(
        scenario_id="neg_04", failure_mode="negation",
        corpus={
            "d1": "Bitcoin uses proof-of-work consensus.",
            "d2": "Ethereum no longer uses proof-of-work; it moved to proof-of-stake in 2022.",
            "d3": "Proof-of-work consensus dominated early cryptocurrencies.",
        },
        query="Does Ethereum still use proof-of-work?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Recent change; old fact lexically similar.",
    ))
    out.append(AdversarialScenario(
        scenario_id="neg_05", failure_mode="negation",
        corpus={
            "d1": "macOS supports Bash by default.",
            "d2": "Recent macOS releases removed Bash; the default shell is zsh.",
            "d3": "Linux distributions usually ship with Bash as the default shell.",
        },
        query="Is Bash the default shell on macOS?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="State change; both true facts coexist temporally.",
    ))
    out.append(AdversarialScenario(
        scenario_id="neg_06", failure_mode="negation",
        corpus={
            "d1": "OAuth 2.0 is widely used for authorization.",
            "d2": "OAuth 2.0 is an authorization framework, NOT an authentication protocol.",
            "d3": "OpenID Connect adds authentication on top of OAuth 2.0.",
        },
        query="Is OAuth 2.0 an authentication protocol?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1"],
        description="Common misconception; vector match prefers d1.",
    ))

    # ── MULTI-HOP (6 scenarios) ──────────────────────────────────
    out.append(AdversarialScenario(
        scenario_id="mh_01", failure_mode="multi_hop",
        corpus={
            "d1": "Alice manages the platform team.",
            "d2": "The platform team owns the deployment pipeline.",
            "d3": "Bob is a separate IC who works on docs.",
        },
        query="Who is responsible for the deployment pipeline?",
        correct_doc_ids=["d1", "d2"], distractor_doc_ids=["d3"],
        description="Requires composing Alice→platform→pipeline; either doc alone is insufficient.",
    ))
    out.append(AdversarialScenario(
        scenario_id="mh_02", failure_mode="multi_hop",
        corpus={
            "d1": "Mamba is a selective state space model.",
            "d2": "Selective state space models are linear-time alternatives to attention.",
            "d3": "FlashAttention is an IO-aware exact attention algorithm.",
        },
        query="What is Mamba's computational complexity?",
        correct_doc_ids=["d1", "d2"], distractor_doc_ids=["d3"],
        description="d1+d2 → linear-time; neither alone says 'Mamba is linear-time'.",
    ))
    out.append(AdversarialScenario(
        scenario_id="mh_03", failure_mode="multi_hop",
        corpus={
            "d1": "BERT bug B-42 was introduced by PR #128.",
            "d2": "PR #128 was authored by Carlos in October.",
            "d3": "Carlos joined the team in September.",
        },
        query="Who introduced bug B-42?",
        correct_doc_ids=["d1", "d2"], distractor_doc_ids=["d3"],
        description="d1 names the PR; d2 names the author; only combined do we know.",
    ))
    out.append(AdversarialScenario(
        scenario_id="mh_04", failure_mode="multi_hop",
        corpus={
            "d1": "The /admin endpoint requires the manager role.",
            "d2": "Dana is a manager.",
            "d3": "Erin is an IC engineer.",
        },
        query="Can Dana access /admin?",
        correct_doc_ids=["d1", "d2"], distractor_doc_ids=["d3"],
        description="Authorization inference requires composing role and permission docs.",
    ))
    out.append(AdversarialScenario(
        scenario_id="mh_05", failure_mode="multi_hop",
        corpus={
            "d1": "Fiona moved from the SF office to the NYC office last quarter.",
            "d2": "The NYC office holds standups at 10 AM Eastern.",
            "d3": "The Austin office holds standups at 9 AM Central.",
        },
        query="What time is Fiona's standup?",
        correct_doc_ids=["d1", "d2"], distractor_doc_ids=["d3"],
        description="Need Fiona→NYC (d1) and NYC→10am (d2).",
    ))
    out.append(AdversarialScenario(
        scenario_id="mh_06", failure_mode="multi_hop",
        corpus={
            "d1": "Module M imports utilities from module U.",
            "d2": "Module U was deprecated in version 2.0.",
            "d3": "Module M is widely used in production.",
        },
        query="Does module M depend on deprecated code?",
        correct_doc_ids=["d1", "d2"], distractor_doc_ids=["d3"],
        description="d1+d2 imply yes; d3 is a distractor about M alone.",
    ))

    # ── DISTRACTOR (6 scenarios) ─────────────────────────────────
    out.append(AdversarialScenario(
        scenario_id="dist_01", failure_mode="distractor",
        corpus={
            "d1": "Eve works at Stripe earning $200k as a staff engineer.",
            "d2": "Eve's previous role at Square paid $180k.",
            "d3": "Stripe's average staff salary is $250k according to levels.fyi.",
        },
        query="What is Eve's current salary at Stripe?",
        correct_doc_ids=["d1"], distractor_doc_ids=["d2", "d3"],
        description="d3 has lots of Stripe+salary words but is general; d1 is specific.",
    ))
    out.append(AdversarialScenario(
        scenario_id="dist_02", failure_mode="distractor",
        corpus={
            "d1": "The API key for the production environment is sk_live_PROD_xyz.",
            "d2": "The API key for staging is sk_test_STG_abc — do not use in prod.",
            "d3": "API keys are managed through the credentials service.",
        },
        query="What's the API key for production?",
        correct_doc_ids=["d1"], distractor_doc_ids=["d2", "d3"],
        description="d2 also mentions API keys + prod (warning); d3 is generic.",
    ))
    out.append(AdversarialScenario(
        scenario_id="dist_03", failure_mode="distractor",
        corpus={
            "d1": "Greg's GitHub username is greg-codes-things.",
            "d2": "Gregory Roberts is a different person; his handle is gregory-r.",
            "d3": "Greg's LinkedIn shows he works at Google.",
        },
        query="What is Greg's GitHub username?",
        correct_doc_ids=["d1"], distractor_doc_ids=["d2", "d3"],
        description="Name collision (Greg vs Gregory); d3 is wrong context.",
    ))
    out.append(AdversarialScenario(
        scenario_id="dist_04", failure_mode="distractor",
        corpus={
            "d1": "Hugo's office address: 250 Lytton Ave, Palo Alto.",
            "d2": "Hugo's home address: 1421 Bryant St, San Francisco.",
            "d3": "Hugo prefers to receive packages at the office.",
        },
        query="What's Hugo's home address?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Office vs home conflation; d3 is misleading hint.",
    ))
    out.append(AdversarialScenario(
        scenario_id="dist_05", failure_mode="distractor",
        corpus={
            "d1": "Order #4711 ships on November 12 via FedEx 2-day.",
            "d2": "Order #4717 (different) ships on November 12 via UPS Ground.",
            "d3": "FedEx Express usually arrives next business day.",
        },
        query="When does order 4711 arrive and via which carrier?",
        correct_doc_ids=["d1"], distractor_doc_ids=["d2", "d3"],
        description="d2 has similar order ID + same date but wrong carrier.",
    ))
    out.append(AdversarialScenario(
        scenario_id="dist_06", failure_mode="distractor",
        corpus={
            "d1": "Iris's published paper from 2024 is about Mamba.",
            "d2": "Iris also has a 2023 paper on transformers (not the relevant one).",
            "d3": "Iris is the lead author on the 2024 Mamba paper.",
        },
        query="What's Iris's most recent paper about?",
        correct_doc_ids=["d1", "d3"], distractor_doc_ids=["d2"],
        description="d2 mentions Iris + papers + year, but wrong year.",
    ))

    # ── CONTRADICTION (6 scenarios) ──────────────────────────────
    out.append(AdversarialScenario(
        scenario_id="contra_01", failure_mode="contradiction",
        corpus={
            "d1": "Jack's preferred Slack channel is #engineering.",
            "d2": "Jack confirmed in last week's standup that his channel is #platform.",
            "d3": "The #engineering channel has 200 members.",
        },
        query="Which Slack channel does Jack prefer?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="d1 (older) and d2 (newer) contradict; correct retrieval surfaces d2.",
    ))
    out.append(AdversarialScenario(
        scenario_id="contra_02", failure_mode="contradiction",
        corpus={
            "d1": "Karen lives in San Francisco.",
            "d2": "Update: Karen moved to Denver in October.",
            "d3": "San Francisco has a Mediterranean climate.",
        },
        query="Where does Karen live?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Knowledge update; older fact still in corpus.",
    ))
    out.append(AdversarialScenario(
        scenario_id="contra_03", failure_mode="contradiction",
        corpus={
            "d1": "Leo's permission level: read-only.",
            "d2": "Leo's permission was upgraded to admin on March 14.",
            "d3": "Read-only users cannot delete records.",
        },
        query="What is Leo's permission level now?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Permission change; old level still recorded.",
    ))
    out.append(AdversarialScenario(
        scenario_id="contra_04", failure_mode="contradiction",
        corpus={
            "d1": "The build process uses Bazel.",
            "d2": "After Q3 migration, the build process now uses just-make.",
            "d3": "Bazel is good for large monorepos.",
        },
        query="What does the build process use?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Migration; old tool still in corpus.",
    ))
    out.append(AdversarialScenario(
        scenario_id="contra_05", failure_mode="contradiction",
        corpus={
            "d1": "Maria is the team lead for infrastructure.",
            "d2": "Maria left the company in February. Nina is the new infra lead.",
            "d3": "Infrastructure team owns CI/CD and observability.",
        },
        query="Who is the infrastructure team lead?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Personnel change; d1 is stale.",
    ))
    out.append(AdversarialScenario(
        scenario_id="contra_06", failure_mode="contradiction",
        corpus={
            "d1": "The pricing tier is $99/month.",
            "d2": "As of January 1, pricing increased to $149/month.",
            "d3": "Monthly billing is processed on the 1st.",
        },
        query="What is the current monthly price?",
        correct_doc_ids=["d2"], distractor_doc_ids=["d1", "d3"],
        description="Price update; both facts present in corpus.",
    ))

    return out


# ── Retrieval methods ───────────────────────────────────────────


def retrieve_vector_only(query: str, corpus: dict[str, str], k: int) -> list[str]:
    from core import memory
    if hasattr(memory, "_get_embedder"):
        embedder = memory._get_embedder()
    else:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    import numpy as np
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    doc_ids = list(corpus.keys())
    doc_embs = embedder.encode([corpus[d] for d in doc_ids],
                                normalize_embeddings=True, show_progress_bar=False)
    sims = doc_embs @ q_emb
    ranked = sorted(zip(doc_ids, sims, strict=False), key=lambda x: -x[1])
    return [d for d, _ in ranked[:k]]


def retrieve_bert_hybrid(query: str, corpus: dict[str, str], k: int) -> list[str]:
    """RRF-fused vector + BM25."""
    from rank_bm25 import BM25Okapi
    doc_ids = list(corpus.keys())
    # BM25
    tokenized = [corpus[d].lower().split() for d in doc_ids]
    bm = BM25Okapi(tokenized)
    bm_scores = bm.get_scores(query.lower().split())
    bm_order = [i for i, _ in sorted(enumerate(bm_scores), key=lambda x: -x[1])]
    # Vector
    v_order_doc_ids = retrieve_vector_only(query, corpus, k=len(corpus))
    v_order = [doc_ids.index(d) for d in v_order_doc_ids]
    # RRF
    rrf_scores: dict[int, float] = {}
    for r, i in enumerate(bm_order):
        rrf_scores[i] = rrf_scores.get(i, 0) + 1 / (60 + r)
    for r, i in enumerate(v_order):
        rrf_scores[i] = rrf_scores.get(i, 0) + 1 / (60 + r)
    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [doc_ids[i] for i, _ in ranked[:k]]


METHODS = {
    "vector_only": retrieve_vector_only,
    "bert_hybrid": retrieve_bert_hybrid,
}


# ── Evaluation ──────────────────────────────────────────────────


@dataclass
class ModeResult:
    failure_mode: str
    n_scenarios: int
    n_caught: int
    catch_rate: float
    scenarios_failed: list[str]


@dataclass
class MethodResult:
    method: str
    overall_catch_rate: float
    by_mode: list[ModeResult]


def evaluate(method_name: str, scenarios: list[AdversarialScenario]) -> MethodResult:
    fn = METHODS[method_name]
    by_mode: dict[str, list[tuple[str, bool]]] = {}
    for sc in scenarios:
        retrieved = fn(sc.query, sc.corpus, k=3)
        # "Caught" = correct doc id in top-1 (strictest)
        top_1 = retrieved[0] if retrieved else ""
        caught = top_1 in sc.correct_doc_ids
        by_mode.setdefault(sc.failure_mode, []).append((sc.scenario_id, caught))
    mode_results = []
    for mode, hits in by_mode.items():
        n_caught = sum(1 for _, c in hits if c)
        failed = [sid for sid, c in hits if not c]
        mode_results.append(ModeResult(
            failure_mode=mode, n_scenarios=len(hits),
            n_caught=n_caught, catch_rate=n_caught / len(hits),
            scenarios_failed=failed,
        ))
    overall = statistics.mean([r.catch_rate for r in mode_results])
    return MethodResult(
        method=method_name, overall_catch_rate=overall,
        by_mode=mode_results,
    )


def write_summary(scenarios: list[AdversarialScenario], results: list[MethodResult], ts: str) -> Path:
    summary_path = RESULTS_DIR / f"b5_summary_{ts}.md"
    modes = sorted({s.failure_mode for s in scenarios})
    lines = [
        "# B5 — Adversarial-eval-by-design benchmark",
        "",
        f"_Generated: {ts}_",
        f"_Scenarios: {len(scenarios)} across {len(modes)} failure modes_",
        "",
        "## Methodology",
        "",
        "- 4 failure modes: negation, multi-hop, distractor, contradiction",
        "- Each mode has 6 scenarios (24 total)",
        "- A scenario is 'caught' iff top-1 retrieved doc is in `correct_doc_ids`",
        "  (strictest interpretation — top-3 inclusion would be more lenient)",
        "- The corpus is small (3 docs each) — these are MICRO-cases isolating",
        "  the failure mode, not realistic corpus scale",
        "",
        "## Per-method overall catch rate",
        "",
        "| Method | Overall catch rate |",
        "|---|---:|",
    ]
    for r in results:
        lines.append(f"| `{r.method}` | {r.overall_catch_rate:.3f} |")
    lines += ["", "## Per-mode breakdown", ""]
    header = "| Failure mode | " + " | ".join(r.method for r in results) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(results))
    for mode in modes:
        row = f"| {mode} |"
        for r in results:
            m_result = next((m for m in r.by_mode if m.failure_mode == mode), None)
            if m_result:
                row += f" {m_result.n_caught}/{m_result.n_scenarios} ({m_result.catch_rate:.2f}) |"
            else:
                row += " — |"
        lines.append(row)
    lines += [
        "",
        "## Scenarios where each method failed",
        "",
    ]
    for r in results:
        lines.append(f"### `{r.method}`")
        for m in r.by_mode:
            if m.scenarios_failed:
                lines.append(f"- **{m.failure_mode}**: {', '.join(m.scenarios_failed)}")
        lines.append("")
    lines += [
        "## Honest limitations",
        "",
        "- **micro-corpus**: 3 docs per scenario; real RAG corpora are 10K+",
        "  and the failure modes manifest differently at scale.",
        "- **strict top-1**: Penalizes near-misses. A top-3 catch-rate would",
        "  show better numbers for both methods.",
        "- **no adversarial-eval-agent**: This measures retrieval; bert's wedge",
        "  also includes an adversarial-eval AGENT that runs over outputs.",
        "  That agent isn't exercised here (separate cycle artifact).",
        "",
    ]
    summary_path.write_text("\n".join(lines))
    return summary_path


def main() -> int:
    print("════════════════════════════════════════════════════════════════", flush=True)
    print("  B5 — Adversarial-eval-by-design benchmark", flush=True)
    print("════════════════════════════════════════════════════════════════", flush=True)
    print(flush=True)

    scenarios = make_scenarios()
    modes = sorted({s.failure_mode for s in scenarios})
    print(f"Scenarios: {len(scenarios)} across {len(modes)} failure modes "
          f"({', '.join(modes)})", flush=True)
    print(flush=True)

    results = []
    for m in METHODS:
        print(f"Evaluating {m}…", flush=True)
        r = evaluate(m, scenarios)
        print(f"  overall = {r.overall_catch_rate:.3f}", flush=True)
        for mr in r.by_mode:
            print(f"    {mr.failure_mode:18s}  "
                  f"{mr.n_caught}/{mr.n_scenarios} ({mr.catch_rate:.2f})",
                  flush=True)
        results.append(r)

    ts = time.strftime("%Y%m%dT%H%M%S")
    json_path = RESULTS_DIR / f"b5_adversarial_{ts}.json"
    json_path.write_text(json.dumps({
        "n_scenarios": len(scenarios),
        "results": [asdict(r) for r in results],
        "timestamp": ts,
    }, indent=2))
    summary_path = write_summary(scenarios, results, ts)
    print(flush=True)
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {summary_path}", flush=True)
    print(f"All {len(results)} methods evaluated.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
