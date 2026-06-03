"""bert run — autonomous cycle loop.

Reads a lab's seed_brief.md, then runs N research cycles end-to-end.
Each cycle:
  1. Researcher dispatch — gather information on the seed topic.
  2. Strategist dispatch — evaluate the research, declare a position.
  3. Threshing pass — assess quality + emit a verdict event.

Each dispatch hits the real router (Groq + NVIDIA + Mistral fallback).
Events flow into the lab's `sor/events.jsonl` automatically via
core.subagent's observability wiring. After the loop, the daily report
+ director letter pick up the new events on the next compile.

Usage:
  .venv/bin/python tools/bert_run.py                # 3 cycles on default lab
  .venv/bin/python tools/bert_run.py --max-cycles 5
  .venv/bin/python tools/bert_run.py --lab ~/.bert/labs/my-lab
  .venv/bin/python tools/bert_run.py --dry-run      # plumbing only, no dispatches

Stage-safety: missing API keys exit 2 with a clear message; Ctrl-C
exits cleanly; max-cycles cap prevents runaway loops; --watch loops
forever with a tick interval (Ctrl-C to stop).

Exit codes:
  0  all cycles completed successfully
  1  partial success (some dispatches failed)
  2  hard failure (missing keys, no seed brief, etc.)
130  Ctrl-C
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def _warmup_jsonschema(attempts: int = 3, sleep_s: float = 0.3) -> None:
    """Force jsonschema_specifications to load its bundled schemas at
    process start, retrying on transient I/O failures.

    On disk-pressured macOS we've seen the package's eager schema
    loader (`REGISTRY = (_schemas() @ _EMPTY_REGISTRY).crawl()` runs at
    module import) hit JSONDecodeError on a file that re-reads cleanly
    seconds later — APFS/Spotlight contention on first read of a
    freshly pulled-in resource. Doing it once up front (before
    director or subagent fires) avoids a mid-cycle crash. If the
    module body raised, sys.modules holds a partial — evict before
    retry."""
    import importlib
    for attempt in range(attempts):
        sys.modules.pop("jsonschema_specifications", None)
        try:
            jss = importlib.import_module("jsonschema_specifications")
            _ = jss.REGISTRY
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(sleep_s)


_warmup_jsonschema()

DEFAULT_MODEL = "nvidia/meta/llama-3.3-70b-instruct"
DEFAULT_MAX_CYCLES = 3
DEFAULT_WATCH_INTERVAL_SECS = 60

CANDLE = "\033[38;5;215m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _print(msg: str, color: str = "") -> None:
    print(f"{color}{msg}{RESET}", flush=True)


# ── Pre-flight ──────────────────────────────────────────────────────

def _check_provider_keys() -> tuple[bool, list[str]]:
    """At least one provider key required. Returns (ok, present_keys).

    GG-A.0 — reads via core.config.load() so persisted keys at
    ~/.bert-lab/credentials.json count, not just env vars. Pre-GG
    this only checked os.environ which forced the operator to
    export keys every shell session even after onboarding had
    saved them.
    """
    candidates = ["GROQ_API_KEY", "NVIDIA_API_KEY", "MISTRAL_API_KEY",
                  "CEREBRAS_API_KEY", "GOOGLE_AI_API_KEY", "GOOGLE_API_KEY",
                  "OPENROUTER_API_KEY", "HF_TOKEN"]
    try:
        from core import config as _cfg
        cfg = _cfg.load()
        present = [k for k in candidates if cfg.credentials.get(k)]
    except Exception:  # noqa: BLE001
        # Fall back to env-only check if config load fails
        present = [k for k in candidates if os.environ.get(k)]
    return bool(present), present


def _resolve_lab(lab_arg: str | None) -> Path:
    """Resolve --lab to an absolute path. Defaults to the repo's own
    lab/ directory if no arg given."""
    if lab_arg is None:
        return LAB_ROOT / "lab"
    p = Path(lab_arg).expanduser()
    if not p.is_absolute():
        # Try interpreting as a name first: ~/.bert/labs/<name>
        candidate = Path.home() / ".bert" / "labs" / lab_arg
        if candidate.exists():
            return candidate
        # Fall back to relative path
        p = (Path.cwd() / lab_arg).resolve()
    return p


def _read_seed_brief(lab_path: Path) -> str:
    seed_path = lab_path / "seed_brief.md"
    if not seed_path.exists():
        raise FileNotFoundError(
            f"no seed_brief.md at {seed_path}. Scaffold the lab first: "
            f"`bert init --name <lab-name>`."
        )
    return seed_path.read_text()


def _next_cycle_id(lab_path: Path, fallback_start: int = 1) -> int:
    """Walk recent events.jsonl tail for the largest seen cycle id;
    return max+1. Falls back to `fallback_start` if no events."""
    events_path = lab_path / "sor" / "events.jsonl"
    if not events_path.exists():
        return fallback_start
    try:
        stat = events_path.stat()
        with events_path.open("rb") as f:
            f.seek(max(0, stat.st_size - 256 * 1024))
            tail = f.read().decode("utf-8", errors="replace")
        max_cycle = -1
        for line in tail.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                c = ev.get("cycle")
                if isinstance(c, int) and c > max_cycle:
                    max_cycle = c
            except json.JSONDecodeError:
                continue
        if max_cycle >= 0:
            return max_cycle + 1
    except OSError:
        pass
    return fallback_start


# ── Dispatch specs ──────────────────────────────────────────────────

def _seed_to_research_task(seed: str) -> str:
    """Compress the seed_brief into a researcher task. The full brief
    is the context; the task is the action verb.

    Earlier prompt versions allowed the model to summarize the task
    instead of doing it ("This is a research brief with actual
    findings."). The current wording demands ≥1500 chars of concrete
    domain content, named entities, and reasoning chains. The
    verification gate enforces 1500 chars minimum."""
    context = seed[:800].strip()
    return (
        "You are a domain researcher. The seed brief below is your "
        "research subject. Produce a SUBSTANTIVE research brief "
        "GROUNDED IN REAL EXTERNAL SOURCES, not just model memory.\n\n"
        "WORKFLOW (do all of these in order):\n"
        "1. Call WebSearch with 2-3 targeted queries to find recent "
        "papers, products, or experts in this space. Pull URLs.\n"
        "2. Call WebFetch on the 2-3 most-promising results to extract "
        "concrete claims with paper IDs, author names, or product "
        "specs.\n"
        "3. Write your brief — each signal MUST cite a real URL or "
        "paper ID you actually fetched. No invented citations. If "
        "WebSearch returns nothing useful for a query, say so and "
        "fall back to your model knowledge with a `[WEAK]` tag.\n\n"
        "MANDATORY output format (verification gate enforces all of "
        "these — failures will be recorded as BUILD_FAIL):\n"
        "1. Markdown H1 (# ...) naming the specific topic\n"
        "2. ## Summary — 2-3 sentence overview (≥80 chars)\n"
        "3. ## Top signals — at least 3 numbered signals, each with "
        "a URL or paper ID you actually fetched, the author/source, "
        "the specific claim, and your relevance assessment (≥150 "
        "chars each)\n"
        "4. ## Candidate hypotheses — at least 2 named hypotheses "
        "with confidence (0-1) and the chain of reasoning that "
        "links signals to hypothesis (≥120 chars each)\n"
        "5. ## Open questions — at least 2 concrete questions with "
        "what evidence (specific experiment, paper, benchmark) would "
        "resolve them (≥100 chars each)\n"
        "6. TOTAL file length ≥ 1500 characters\n\n"
        "FORBIDDEN: meta-descriptions ('this is a research brief'), "
        "vague claims ('X has better characteristics'), invented "
        "citations, and signals that just paraphrase the question.\n\n"
        "Use the Write tool to write your output to the path in "
        "output_path.\n\n"
        "MANDATORY gaps file: also write a companion file at the same "
        "path with `_gaps.md` appended before the extension (e.g., if "
        "output_path is `findings/foo.md`, also write "
        "`findings/foo_gaps.md`). The gaps file MUST contain ≥3 "
        "markdown bullets:\n"
        "  - Sources you couldn't access (paywalls, dead links)\n"
        "  - Claims you couldn't verify (evidence uncertain)\n"
        "  - Open questions a knowledgeable reader would still ask\n"
        "An EMPTY gaps file fails the cycle. Honest disclosure non-negotiable.\n\n"
        "Then return a ResultPacket with verdict=APPROVE and "
        "calibration_reasoning ≥80 chars explaining your confidence + "
        "key uncertainties.\n\n"
        f"--- seed brief ---\n{context}"
    )


def _skill_plan_section_for_role(role: str) -> str:
    """Sprint 2: read the role's frontmatter skill_plan and produce a
    prompt fragment listing the recommended skills for this role.

    The agent isn't forced to use them — this is an availability
    announcement. Each line: `- <skill_name>: <one-line description>`.

    Returns empty string when:
      - role isn't in the registry
      - role has no skill_plan declared
      - the named skills don't exist in the skill registry
    """
    try:
        from core import role_registry as _rr
        from core import skill_registry as _sr
    except ImportError:
        return ""
    tmpl = _rr.load(role)
    if tmpl is None or not tmpl.skill_plan:
        return ""
    lines: list[str] = []
    for skill_name in tmpl.skill_plan:
        skill = _sr.get(skill_name)
        if skill is None:
            continue
        first_line = (skill.description or "").strip().split("\n", 1)[0]
        lines.append(f"  - `{skill_name}`: {first_line}")
    if not lines:
        return ""
    return (
        "\n\nSKILLS available for this role (use as building blocks "
        "if helpful; not mandatory):\n" + "\n".join(lines)
    )


def _seed_to_role_task(role: str, seed: str,
                       prior_findings: list[str] | None = None) -> str:
    """Generic per-role task generator for non-researcher / non-strategist roles.

    Sprint 2: skill_plan from the role's frontmatter is injected as a
    SKILLS section so the agent knows which seed skills are available
    for this role and what each is for. Agents can choose to invoke them
    via their own reasoning — the dispatch isn't forced.

    Specialized helpers exist for researcher (_seed_to_research_task)
    and strategist (_seed_to_strategy_task) — those are preserved.
    Other roles (analyst, option_scorer, red_team, writer,
    code_reader, test_author, etc.) use this generic generator.
    """
    context = seed[:800].strip()
    role_instruction = _ROLE_INSTRUCTIONS.get(role, _ROLE_INSTRUCTIONS["_default"])
    prior_section = ""
    if prior_findings:
        prior_section = (
            "\n\nPRIOR FINDINGS IN THIS CYCLE (read each before producing yours):\n"
            + "\n".join(f"  - {p}" for p in prior_findings)
        )
    skill_section = _skill_plan_section_for_role(role)
    # output_path is `findings/bert_run_C{cycle}_{role}.md`; the
    # gaps file lives alongside it as `..._gaps.md`. We can't include
    # the exact path in this prompt template (it's per-cycle), so the
    # convention is "write {output_path}_gaps.md" — the agent knows
    # output_path at dispatch time.
    return (
        f"You are bert's {role}. {role_instruction}\n\n"
        f"--- mission brief ---\n{context}{prior_section}{skill_section}\n\n"
        "Write your main deliverable to the path in output_path. The "
        "verification gate requires:\n"
        "  - Markdown H1 (# Title)\n"
        "  - At least 3 H2 sections (## ...)\n"
        "  - ≥1500 characters total\n"
        "  - At least one real citation (URL / arxiv / paper id / "
        "GitHub link / Author-et-al)\n"
        "  - NO placeholder URLs (no example.com / .org / .net)\n\n"
        "MANDATORY: also write a companion gaps file at the same path "
        "with `_gaps.md` appended before the extension. For example, "
        "if output_path is `findings/foo.md`, also write "
        "`findings/foo_gaps.md`. The gaps file MUST contain ≥3 "
        "markdown bullets covering:\n"
        "  - Sources you couldn't access (paywalls, dead links, missing tools)\n"
        "  - Claims you couldn't verify (evidence uncertain or absent)\n"
        "  - Open questions a knowledgeable reader would still ask\n"
        "An EMPTY gaps file fails the cycle. High-stakes work always has "
        "gaps; pretending otherwise is dishonest.\n\n"
        "Return verdict APPROVE if you have a defensible deliverable, "
        "SCOPE_STOP if the task is genuinely insufficient (don't dodge)."
    )


_ROLE_INSTRUCTIONS: dict[str, str] = {
    "analyst": (
        "Read the mission carefully and produce a structured analysis with "
        "options, scoring criteria, and a recommended option backed by data."
    ),
    "option_scorer": (
        "Identify the discrete options implied by the mission, score each "
        "on the relevant dimensions, surface the top 2-3 with rationale."
    ),
    "red_team": (
        "Adversarially review the proposed direction. Surface ≥1 substantive "
        "issue (factual error, unsupported claim, missing scope, hidden "
        "assumption). If you genuinely can't find any, explain why."
    ),
    "writer": (
        "Synthesize the prior findings into a polished, defensible "
        "deliverable. Comprehensive coverage. Every claim cites a source."
    ),
    "literature_hunter": (
        "Search for recent, paper-shaped sources (papers, technical reports, "
        "established blog posts with citations). Produce a structured list "
        "with full citations."
    ),
    "change_detector": (
        "Identify what has CHANGED in this domain in the last 18 months. "
        "Each change must cite a source dated within that window."
    ),
    "methodology_critic": (
        "Examine the methodological choices in cited sources. Flag "
        "weaknesses, missing controls, sample-size concerns."
    ),
    "code_reader": (
        "Read the relevant code (use Grep / Glob / Read tools). Produce a "
        "structured map of the codebase or subsystem the mission addresses."
    ),
    "refactor_specialist": (
        "Identify refactor candidates with concrete before/after snippets. "
        "Justify each by tests-still-pass + measurable improvement."
    ),
    "test_author": (
        "Write tests for the target code. Tests must actually pass when "
        "run via pytest. Cover happy path + at least 2 edge cases."
    ),
    "reviewer": (
        "Review the change for correctness, style, security, maintainability. "
        "Surface specific line-level issues."
    ),
    "security_auditor": (
        "Audit the code for security issues with file:line evidence. "
        "Score severity. Suggest concrete fixes."
    ),
    "performance_tuner": (
        "Identify performance hotspots with measurement. Propose specific "
        "optimizations with expected impact."
    ),
    "evaluator": (
        "Independently grade the prior work product on its own merits. "
        "Score on correctness, completeness, defensibility, honesty."
    ),
    "consolidator": (
        "Combine multi-cycle findings into a coherent synthesis. Resolve "
        "contradictions explicitly."
    ),
    "_default": (
        "Read the mission, produce a substantive deliverable in the shape "
        "implied by the role name."
    ),
}


def _seed_to_strategy_task(seed: str, research_path: str) -> str:
    context = seed[:600].strip()
    return (
        "You are a strategist. Read the research findings at the path "
        "below, combine with the seed brief, and recommend the *single* "
        "best next action. The PI needs specifics, not abstract advice.\n\n"
        "MANDATORY output format (verification gate enforces all of "
        "these — failures will be recorded as BUILD_FAIL):\n"
        "1. Markdown H1 (# ...) naming the recommended action concretely\n"
        "2. ## Recommendation — specific, testable next step (what to "
        "build / measure / read, in what order), ≥200 chars\n"
        "3. ## Why — your reasoning chain referencing the research "
        "findings, ≥250 chars\n"
        "4. ## How to falsify — what observation would prove the "
        "recommendation wrong, ≥150 chars\n"
        "5. TOTAL file length ≥ 1500 characters\n\n"
        "FORBIDDEN: vague recommendations like 'build a better model' "
        "or 'do more research'. Be specific: which architecture, "
        "which benchmark, which timeframe, which dataset.\n\n"
        "Use the Write tool to write your output to the path in "
        "output_path.\n\n"
        "MANDATORY gaps file: also write a companion file at the same "
        "path with `_gaps.md` appended before the extension (e.g., if "
        "output_path is `findings/foo.md`, also write "
        "`findings/foo_gaps.md`). The gaps file MUST contain ≥3 "
        "markdown bullets:\n"
        "  - Open questions your recommendation doesn't resolve\n"
        "  - Risks / failure modes of the recommended path\n"
        "  - What would make you change your mind (the falsifier)\n"
        "An EMPTY gaps file fails the cycle.\n\n"
        "ResultPacket verdict=APPROVE if you have a clear "
        "recommendation, SCOPE_STOP if the research is genuinely "
        "insufficient (don't dodge — give your best call based on the "
        "research you have).\n\n"
        f"--- seed brief ---\n{context}\n\n"
        f"--- research output ---\nPath: {research_path}"
    )


def _resolve_dispatch_model(role: str, task_text: str, default_model: str) -> str:
    """Sprint 1 commit 11: call multi-source router to pick the best
    (provider, model) for this role+task. Returns 'provider/model'
    string in the format bert_run + subagent expect.

    Best-effort — on any failure, returns the default_model passed in
    (preserves legacy behavior for callers that haven't migrated).

    Operator escape hatch: BERT_FORCE_MODEL='provider/model' pins every dispatch
    to one known-good lane, bypassing the router. Useful when a routed provider's
    free-tier quota is exhausted (the per-dispatch retry doesn't cross-provider
    fall back), and for deterministic data-gen / test runs.
    """
    import os as _os
    forced = (_os.environ.get("BERT_FORCE_MODEL") or "").strip()
    if forced:
        return forced
    try:
        from core import host_detector, router
        ctx = host_detector.detect()
        byo_keys = set(ctx.byo_keys_present)
        provider, model = router.resolve_model_for_dispatch(
            role=role,
            task_text=task_text,
            host_ctx=ctx,
            byo_keys=byo_keys,
        )
        return f"{provider}/{model}"
    except Exception as e:  # noqa: BLE001
        # Defensive fallback — don't crash dispatch on a routing miss
        import logging
        logging.getLogger("bert.dispatch_routing").debug(
            "model resolve fallback (%s): using default %s",
            e, default_model,
        )
        return default_model


def _verification_spec_for_role(role: str) -> dict:
    """Sprint 1 commit 10: pull per-role verification spec from the
    role template's frontmatter if it declares one; else use the
    default spec.

    Most role templates today don't declare custom verification — they
    inherit the default. Sprint 2 will add per-role specs for build /
    audit / strategist roles.
    """
    try:
        from core import role_registry, verify_engine
        custom = role_registry.get_verification_spec(role)
        if custom:
            return custom
        # Default spec lives in verify_engine; copy to avoid shared-state mutations
        return dict(verify_engine.DEFAULT_SPEC)
    except Exception:  # noqa: BLE001
        # If role_registry/verify_engine can't load, fall back to the
        # spec that's hardcoded directly in _build_spec below.
        return {}


def _build_spec(*, role: str, cycle: int, task: str, output_path: str,
                model: str, falsifier_text: str) -> dict:
    """Wire-format dispatch spec for core.subagent.run_subagent.

    schemas/dispatch_spec.json (referenced via validate_dispatch_spec)
    requires:
      - process_hygiene minLength 20  — was "" pre-fix, silently
        rejected every researcher/strategist dispatch
      - confidence_required boolean   — was missing entirely
      - success_criterion minLength 20
      - falsifier_text   minLength 30
    The previous spec only met some of those, so run_subagent's first
    line (validate_dispatch_spec) returned False, the function returned
    early without firing the LLM, and _run_one_cycle saw result_valid=
    False → cycle stopped before any work happened. Bug surfaced when
    the user's first real test01 mission terminated via three-strike
    without ever spawning a researcher.
    """
    # Sprint 1 commit 11: per-role multi-source model resolution.
    # Falls back to the passed-in `model` if router can't pick.
    resolved_model = _resolve_dispatch_model(role, task_text=task, default_model=model)

    # Sprint 1 commit 10: per-role verification spec from role template
    # frontmatter. Most roles inherit DEFAULT_SPEC today; Sprint 2 will
    # add role-specific specs (test_author runs pytest, auditor checks
    # ledger rows, etc.).
    role_verification_spec = _verification_spec_for_role(role)

    return {
        "dispatch_altitude": "IMPL",
        "role": role,
        "cycle": cycle,
        "task": task,
        "success_criterion": (
            f"Sub-agent writes a schema-valid ResultPacket to "
            f"{output_path} that satisfies the falsifier_text criterion. "
            f"On failure, the verdict must be honest (REJECT or OTHER, "
            f"not synthesized BUILD_PASS from external check)."
        ),
        "output_path": output_path,
        "model": resolved_model,           # router-resolved, not caller's default
        "falsifier_text": falsifier_text,
        # Verification gate enforces actual content + grounding.
        # Earlier runs produced ungrounded prose then fabricated
        # `example.com` URLs to pass the citation gate. Now we require:
        # (a) ≥1500 chars, (b) H1 + ≥3 H2, (c) at least one real-
        # looking citation (URL/arxiv/paper id), (d) NO obvious
        # placeholder URLs like example.com/.org/.net.
        #
        # Sprint 1 commit 2 (v1.0): Python-native verification_spec
        # eliminates shell injection. The legacy shell `verification_command`
        # remains as a backward-compat fallback for any caller that
        # hasn't migrated yet; subagent prefers verification_spec.
        # Sprint 1 commit 10: per-role spec from role template overrides
        # this default. _verification_spec_for_role returns DEFAULT_SPEC
        # (or empty if registry unavailable) when no per-role spec exists.
        "verification_spec": role_verification_spec or {
            "output_required": True,
            "min_chars": 1500,
            "required_headers": [
                {"level": 1, "count": 1},
                {"level": 2, "count": 3},
            ],
            "required_patterns": [
                {
                    "description": "at least one citation",
                    "pattern": r"https?://|arxiv:|arXiv:|doi\.org|github\.com|[A-Z][a-z]+ et al",
                },
            ],
            "forbidden_patterns": [
                {
                    "description": "no placeholder URLs",
                    "pattern": r"example\.(com|org|net)",
                },
                {
                    "description": "no placeholder markers",
                    "pattern": r"\bTBD\b|\bXXX\b|\bplaceholder\b",
                },
            ],
            # Sprint 1 commit 3: gaps.md honest disclosure required.
            # Memory rule: "Honest disclosure non-negotiable —
            # failures.md mandatory, signed separately." Without this
            # gate, agents over-claim confidence and gaps stay hidden.
            "gaps_required": {
                "enabled": True,
                "min_bullets": 3,
            },
        },
        # Legacy shell command (back-compat — subagent uses verification_spec when present)
        "verification_command": (
            f"test -s {output_path} "
            f"&& grep -q '^# ' {output_path} "
            f"&& grep -cE '^## ' {output_path} | grep -qE '^[3-9]|^[1-9][0-9]' "
            f"&& [ $(wc -c < {output_path}) -ge 1500 ] "
            f"&& grep -qE 'https?://|arxiv:|arXiv:|doi.org|github.com|[A-Z][a-z]+ et al' {output_path} "
            f"&& ! grep -qE 'example\\.(com|org|net)|placeholder|TBD\\b|XXX' {output_path}"
        ),
        "process_hygiene": (
            f"Write a ResultPacket JSON + a markdown brief at "
            f"{output_path}. If the task can't be completed, return an "
            f"honest REJECT verdict rather than fabricating findings."
        ),
        "confidence_required": True,
        "forbidden_actions": [],
        "caveats_embedded": [],
    }


# ── Cycle loop ──────────────────────────────────────────────────────


def _anthropic_cli_model_flag(resolved_model: str) -> str:
    """Map a router-resolved anthropic-cli model id to the `claude -p
    --model` short alias. The router does cost-tier routing (opus for the
    deep roles, sonnet/haiku for lighter ones); the bridge must honor it
    rather than always burning opus. Unknown/bare ids default to opus
    (highest tier — never silently downgrade a role the router wanted high)."""
    m = (resolved_model or "").lower()
    if "haiku" in m:
        return "haiku"
    if "sonnet" in m:
        return "sonnet"
    return "opus"


def _grade_bridge_artifact(spec: dict, label: str, abs_output: Path,
                           output_path: str, cli_out: dict, t0: float) -> dict:
    """Grade a host-Opus bridge artifact with the SAME verification_spec the
    standard subagent loop enforces, then build the summary dict.

    Host output is not exempt from grading: if the dispatch carries a
    verification_spec, run verify_engine against the written file. On pass →
    APPROVE. On fail → CHANGES_REQUESTED but result_valid stays True (Opus
    wrote a real artifact; the cycle keeps it and stays honest about the
    miss rather than downgrading to free-tier llama, which won't do better).
    No spec → fall back to the existence guarantee the caller already made."""
    role = spec["role"]
    cycle = int(spec["cycle"])
    elapsed = round(time.monotonic() - t0, 1)
    vspec = spec.get("verification_spec")

    verdict = "APPROVE"
    errors: list[str] = []
    if vspec:
        from core import verify_engine
        vr = verify_engine.verify_artifact(vspec, abs_output)
        if not vr.ok:
            verdict = "CHANGES_REQUESTED"
            errors = list(vr.checks_failed)

    usage = cli_out.get("usage") or {}
    return {
        "label": label,
        "role": role,
        "cycle": cycle,
        "verdict": verdict,
        "output_path": output_path,
        "result_path": "",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 8 if verdict == "APPROVE" else 4,
        "calibration_reasoning": (
            f"Routed via claude CLI (host tier); session "
            f"{cli_out.get('session_id', '?')[:8]} wrote {abs_output.stat().st_size} bytes "
            f"in {cli_out.get('duration_ms', 0)//1000}s; cost "
            f"${cli_out.get('total_cost_usd', 0):.3f}; verify={verdict}"
        ),
        "telemetry": {
            "tokens_in": usage.get("input_tokens", 0)
                        + usage.get("cache_creation_input_tokens", 0)
                        + usage.get("cache_read_input_tokens", 0),
            # Split so the real cost is visible: fresh = actually-processed input
            # (priced full), cache_read = re-reads of prior context across agentic
            # turns (priced ~10%). Gross tokens_in counts both and overstates cost.
            "tokens_in_fresh": usage.get("input_tokens", 0)
                              + usage.get("cache_creation_input_tokens", 0),
            "tokens_cache_read": usage.get("cache_read_input_tokens", 0),
            "tokens_out": usage.get("output_tokens", 0),
            "latency_secs": elapsed,
            "model_used": (spec.get("model", "") or "").split("/", 1)[-1]
                          or _anthropic_cli_model_flag(spec.get("model", "")),
            "provider": "anthropic-cli",
            "retry_count": 0,
            "fallback_chain": [],
            "cost_usd": cli_out.get("total_cost_usd", 0),
        },
        "spec_valid": True,
        "result_valid": True,
        "errors": errors,
        "elapsed_secs": elapsed,
    }


def _dispatch_via_claude_cli(spec: dict, label: str, lab_path: Path) -> dict:
    """Bridge: route a dispatch through the local Claude Code CLI
    (`claude -p --model opus`) instead of bert's standard subagent loop.

    Why: NVIDIA llama-3.3-70b-instruct hits a depth + honesty ceiling
    on research tasks (fabricates citations, copies prompt examples).
    For the researcher role specifically, we can route through Opus
    via the user's existing Claude Code OAuth session. Claude Code
    has WebSearch/WebFetch/Write/Read tools built in; we hand it the
    task and read the resulting file.

    Auth: uses the user's logged-in Claude Code session (no
    ANTHROPIC_API_KEY needed). Counts against the Max plan budget.

    Returns a summary dict in the same shape as subagent.run_subagent.
    """
    import subprocess as _sub
    t0 = time.monotonic()
    role = spec["role"]
    output_path = spec["output_path"]
    abs_output = lab_path / output_path

    # A5 — Split prompt into STABLE system prefix + per-cycle user task
    # so Anthropic's prompt cache can fire. The system prefix (role
    # constitution + invariants) is identical across cycles for a given
    # role, so it caches and the next dispatch only pays for the
    # per-cycle task. Measured savings: ~75% input tokens after warmup.
    system_prefix = (
        f"You are bert's {role}. The lab path is {lab_path}.\n"
        "Write your output to the ABSOLUTE PATH specified in the task.\n\n"
        "Use WebSearch and WebFetch to ground claims in real sources. "
        "Cite real URLs and authors — no fabricated example.com links. "
        "Each section must be substantive content (not meta-descriptions)."
    )
    # Tell host Opus exactly what it's graded on (min_chars / headers /
    # required+forbidden patterns) — _grade_bridge_artifact runs that same
    # spec after. Without this the agent is graded on rules it was never
    # shown (the min_chars / missing-header miss the standard loop already
    # fixed via the identical renderer).
    from core.subagent import _render_verification_requirements
    _reqs = _render_verification_requirements(spec.get("verification_spec"))
    per_cycle_task = (
        f"Output absolute path: {abs_output}\n\n"
        f"--- task ---\n{spec['task']}"
        + (f"\n\n{_reqs}" if _reqs else "")
    )

    # Ensure output dir exists so Claude's Write call doesn't fail
    abs_output.parent.mkdir(parents=True, exist_ok=True)

    # Write system prefix to a stable per-role file so Claude CLI can
    # pass it via --append-system-prompt and the prompt cache hits on
    # repeat calls for the same role.
    cache_dir = lab_path / "state" / "claude_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sys_prompt_path = cache_dir / f"system_{role}.md"
    sys_prompt_path.write_text(system_prefix)

    try:
        result = _sub.run(
            [
                "claude", "-p",
                "--model", _anthropic_cli_model_flag(spec.get("model", "")),
                "--output-format", "json",
                "--add-dir", str(lab_path),
                "--permission-mode", "acceptEdits",
                "--allowedTools", "Write,Read,Edit,WebSearch,WebFetch,Bash",
                "--append-system-prompt", system_prefix,
                "--max-budget-usd", "2.0",
                per_cycle_task,
            ],
            capture_output=True, text=True, timeout=900,
        )
    except _sub.TimeoutExpired:
        return _claude_bridge_failure(
            spec, label, t0,
            "claude CLI timed out at 900s")

    if result.returncode != 0:
        return _claude_bridge_failure(
            spec, label, t0,
            f"claude CLI exit {result.returncode}: {result.stderr[:400]}")

    try:
        cli_out = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return _claude_bridge_failure(
            spec, label, t0,
            f"claude CLI returned non-JSON: {e} (first 400 chars: {result.stdout[:400]})")

    if cli_out.get("is_error"):
        return _claude_bridge_failure(
            spec, label, t0,
            f"claude CLI is_error: {cli_out.get('result', '')[:400]}")

    # Verify the agent actually wrote the expected file
    if not abs_output.exists() or abs_output.stat().st_size < 100:
        return _claude_bridge_failure(
            spec, label, t0,
            f"claude CLI ran but did not write {output_path}")

    # Grade the host-Opus artifact with the same verification_spec the
    # standard loop uses — host output is not exempt from the gate.
    return _grade_bridge_artifact(spec, label, abs_output, output_path, cli_out, t0)


def _claude_bridge_failure(spec: dict, label: str, t0: float, msg: str) -> dict:
    """Shape a failure-summary that the cycle loop can handle."""
    return {
        "label": label,
        "role": spec["role"],
        "cycle": spec["cycle"],
        "verdict": "OTHER",
        "output_path": spec["output_path"],
        "result_path": "",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 1,
        "calibration_reasoning": f"claude CLI bridge failed: {msg}",
        "telemetry": {
            "tokens_in": 0, "tokens_out": 0,
            "latency_secs": round(time.monotonic() - t0, 1),
            "model_used": "claude-opus-4-7",
            "provider": "anthropic-cli",
            "retry_count": 0,
            "fallback_chain": [],
        },
        "spec_valid": True,
        "result_valid": False,
        "errors": [msg],
        "elapsed_secs": round(time.monotonic() - t0, 1),
    }


def _dump_run_summary(summaries: list[dict], path: str | None, *,
                      lab_name: str | None, wall_secs: float) -> None:
    """Opt-in: when BERT_RUN_SUMMARY_PATH is set, write the per-cycle/per-dispatch
    summary (incl. telemetry) as JSON so an external harness (the B7 benchmark)
    gets clean, run-isolated telemetry. The claude -p bridge's real token/latency/
    cost numbers live in the dispatch summary; its model_call.jsonl rows are
    all-zero placeholders. No path -> no-op (production default unchanged)."""
    if not path:
        return
    payload = {
        "lab": lab_name,
        "wall_secs": wall_secs,
        "cycles": [
            {
                "cycle": c.get("cycle"),
                "success": c.get("success"),
                "elapsed_secs": c.get("elapsed_secs"),
                "dispatches": [
                    {"role": d.get("role"), "verdict": d.get("verdict"),
                     "result_valid": d.get("result_valid"),
                     "output_path": d.get("output_path"),
                     "telemetry": d.get("telemetry", {})}
                    for d in (c.get("dispatches") or [])
                ],
            }
            for c in summaries
        ],
    }
    try:
        Path(path).write_text(json.dumps(payload, indent=2))
    except OSError as exc:  # never let telemetry I/O crash a run
        logging.getLogger("bert.dispatch_routing").warning(
            "could not write run summary to %s: %s", path, exc)


def _safe_dispatch(spec: dict, label: str, lab_path: Path | None = None) -> dict:
    """Run one dispatch; on crash, return an OTHER-verdict summary so
    the cycle loop can decide whether to continue.

    Host-Opus tier1 (the MCP-first pivot): when the router resolved this
    dispatch to the host's Claude (model starts with "anthropic-cli/"),
    run it on the user's Opus via `claude -p` for ALL roles — that's the
    intelligence bert's infrastructure is meant to serve, not free-tier.
    Free-tier subagent execution is the LAST resort: we fall through to
    the standard loop only if the host CLI bridge fails (so a flaky host
    session still produces work via llama/etc.).

    Legacy: BERT_RESEARCHER_VIA_CLAUDE=1 + role==researcher forces the
    CLI bridge even when the router didn't resolve to anthropic-cli (kept
    for the older researcher-quality override path)."""
    resolved_model = spec.get("model", "") or ""
    host_opus = resolved_model.startswith("anthropic-cli") and lab_path is not None
    legacy_researcher = (
        os.environ.get("BERT_RESEARCHER_VIA_CLAUDE") == "1"
        and spec.get("role") == "researcher"
        and lab_path is not None
    )
    if host_opus or legacy_researcher:
        summary = _dispatch_via_claude_cli(spec, label, lab_path)
        if summary.get("result_valid"):
            return summary
        # Host CLI bridge failed (timeout / non-zero exit / no artifact).
        # Fall through to the standard subagent loop so free-tier failover
        # still produces work rather than the whole cycle going dark.
        logging.getLogger("bert.dispatch_routing").warning(
            "host-Opus CLI bridge failed for %s (model=%s); falling back to "
            "standard subagent loop (free-tier)", label, resolved_model,
        )

    from core import subagent  # lazy import — keeps --dry-run quick
    t0 = time.monotonic()
    try:
        summary = subagent.run_subagent(spec)
    except Exception as exc:  # noqa: BLE001
        return {
            "label": label,
            "role": spec["role"],
            "cycle": spec["cycle"],
            "verdict": "OTHER",
            "result_valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "elapsed_secs": round(time.monotonic() - t0, 1),
        }
    summary["label"] = label
    summary["elapsed_secs"] = round(time.monotonic() - t0, 1)
    return summary


_LEGACY_ROSTER = ("researcher", "strategist")
_BAD_VERDICTS = {"BUILD_FAIL", "REJECT", "OTHER", "CHANGES_REQUESTED"}

# Effort-triage: a trivial lookup gets ONE direct cheap-tier answer instead of a
# 1500-char multi-role research ritual (see core/effort_triage + B8 plan WS0c).
_LIGHT_VERIFICATION_SPEC = {"output_required": True, "min_chars": 40}
_TRIVIAL_TIER_MODEL = "anthropic-cli/claude-haiku-4-5"   # cheapest host tier


def _seed_to_direct_answer(seed: str) -> str:
    """Lightweight task for a trivial lookup: a correct, terse answer with no
    research ritual. Explicitly discourages web tools unless the answer is
    genuinely time-sensitive, which is the main token sink on trivia."""
    return (
        f"Answer this question directly and correctly in 1-3 sentences:\n\n{seed}\n\n"
        "Do NOT use WebSearch/WebFetch unless the answer is genuinely "
        "time-sensitive or you are not sure — most factual lookups you already "
        "know. If you are not certain, say so and name the one source to check. "
        "Write only the answer to the output path; no headers, no report format."
    )


def _run_one_cycle(*, seed: str, cycle: int, model: str,
                   lab_path: Path | None = None,
                   roster: tuple[str, ...] | None = None) -> dict:
    """Dispatch each role in `roster` sequentially. Stop early on first
    invalid result. Returns dict summary with success flag + per-dispatch records.

    Sprint 1: roster comes from `LabSchema.roster_initial` via
    `core/lab_schema_io`. When roster is None, falls back to the legacy
    pair `(researcher, strategist)` for backward compat with any caller
    that doesn't pass it (and for the `BERT_LEGACY_RESEARCHER_STRATEGIST=1`
    safety net flag honored by the `run()` outer wrapper).

    lab_path is threaded through so _safe_dispatch can route the
    researcher dispatch through the Claude CLI bridge when
    BERT_RESEARCHER_VIA_CLAUDE=1 (the bridge needs an absolute
    output path to hand to claude -p).

    Both stages write to findings/ (not drafts/) so the artifact gets
    picked up by Manuscript / Atlas immediately via core.tools._write's
    finding-event auto-emit. Previously each role wrote to drafts/
    only and the cycle's work never surfaced in the UI.
    """
    cycle_start = time.monotonic()
    dispatches: list[dict] = []
    if roster is None:
        roster = _LEGACY_ROSTER
    if not roster:
        # Defensive: an empty roster from a malformed schema. Don't crash;
        # fall back to legacy so the cycle still does useful work.
        roster = _LEGACY_ROSTER

    # Effort-triage (WS0c): decide how much machinery this seed warrants BEFORE
    # committing to the roster. A trivial lookup ("what's the default port?")
    # short-circuits to ONE direct cheap-tier answer — no ritual, no 2nd role —
    # which is where the 253K-token-on-trivia waste came from. Disabled with
    # BERT_EFFORT_TRIAGE=off (the un-gamed baseline / ablation).
    effort = "deep"
    if os.environ.get("BERT_EFFORT_TRIAGE", "on") != "off":
        try:
            from core import effort_triage
            effort, needs_grounding, _conf = effort_triage.classify(seed)
        except Exception:  # noqa: BLE001 — triage must never break a cycle
            effort = "deep"
        try:
            from core import observability as _obs
            _obs.emit("effort_triage", {"cycle": cycle, "effort": effort,
                                        "lab": lab_path.name if lab_path else None})
        except Exception:  # noqa: BLE001
            pass

    if effort == "trivial":
        spec = _build_spec(
            role="researcher", cycle=cycle, model=model,
            task=_seed_to_direct_answer(seed),
            output_path=f"findings/bert_run_C{cycle}_answer.md",
            falsifier_text="A correct, terse direct answer at output_path.",
        )
        spec["model"] = _TRIVIAL_TIER_MODEL        # cheap tier, skip the resolver
        spec["verification_spec"] = dict(_LIGHT_VERIFICATION_SPEC)
        result = _safe_dispatch(spec, "responder", lab_path=lab_path)
        return {
            "cycle": cycle, "effort": "trivial",
            "success": bool(result.get("result_valid")
                            and result.get("verdict") not in _BAD_VERDICTS),
            "elapsed_secs": round(time.monotonic() - cycle_start, 1),
            "dispatches": [result],
        }
    if effort == "standard":
        # Drop the redundant evaluator/critic chain; keep the primary role only.
        roster = roster[:1]

    # Sprint 4 C1 — emit a cycle_started event before any dispatch (pairs with
    # the cycle_outcome rollup so /now + audits get per-cycle wall-clock).
    try:
        from core import observability as _obs
        _obs.emit("cycle_started", {
            "cycle": cycle,
            "lab": lab_path.name if lab_path else None,
            "roster": list(roster),
        })
    except Exception:  # noqa: BLE001
        pass

    prior_findings: list[str] = []
    for idx, role in enumerate(roster):
        output_path = f"findings/bert_run_C{cycle}_{role}.md"

        # Task composition: use the specialized researcher/strategist
        # task generators when those roles appear (they have the
        # citation-grounded, falsifier-aware prompts already tuned for
        # the verification gate). Other roles use _seed_to_role_task.
        if role == "researcher" and idx == 0:
            task = _seed_to_research_task(seed)
        elif role == "strategist" and prior_findings:
            # Strategist's specialized task wants a single research_path.
            # Pass the first prior finding (typically the researcher's).
            task = _seed_to_strategy_task(seed, prior_findings[0])
        else:
            task = _seed_to_role_task(role, seed, prior_findings=prior_findings)

        spec = _build_spec(
            role=role, cycle=cycle, model=model, task=task,
            output_path=output_path,
            falsifier_text=(
                f"Schema-valid ResultPacket + {role} deliverable at output_path."
            ),
        )
        result = _safe_dispatch(spec, role, lab_path=lab_path)
        dispatches.append(result)

        if not result.get("result_valid"):
            # Stop the cycle short: subsequent roles read prior findings.
            return {
                "cycle": cycle,
                "success": False,
                "elapsed_secs": round(time.monotonic() - cycle_start, 1),
                "dispatches": dispatches,
                "stopped_early": True,
                "stop_reason": f"{role}_invalid",
            }
        prior_findings.append(output_path)

    # Honest success: BOTH result_valid AND the verdict is actually positive.
    # Previously success=True whenever a packet existed, which made
    # BUILD_FAIL+synthesized cycles look indistinguishable from real wins.
    success = all(
        d.get("result_valid") and d.get("verdict") not in _BAD_VERDICTS
        for d in dispatches
    )
    return {
        "cycle": cycle,
        "effort": effort,
        "success": success,
        "elapsed_secs": round(time.monotonic() - cycle_start, 1),
        "dispatches": dispatches,
    }


# ── Public API ──────────────────────────────────────────────────────

def run(*, lab_path: Path, max_cycles: int = DEFAULT_MAX_CYCLES,
        model: str = DEFAULT_MODEL, dry_run: bool = False,
        watch: bool = False,
        watch_interval_secs: int = DEFAULT_WATCH_INTERVAL_SECS,
        autonomous: bool = False) -> int:
    """End-to-end autonomous loop.

    Returns shell exit code:
      0 = all cycles succeeded
      1 = partial (some dispatches failed)
      2 = hard fail (missing keys, missing seed brief, etc.)

    CC.3 — When `autonomous=True`, each iteration starts with a
    director dispatch that reads recent state and picks the next
    cycle's shape + focus area. The director's decision composes the
    researcher's prompt. Termination guardrails (3-strike rule,
    failure cascade, pending-approval threshold) prevent runaway loops.
    """
    _run_start = time.monotonic()
    _print("─" * 60, CANDLE)
    if autonomous:
        _print(" bert run — AUTONOMOUS cycle loop (director-led)",
               BOLD + CANDLE)
    else:
        _print(" bert run — autonomous cycle loop", BOLD + CANDLE)
    _print(f" lab: {lab_path}", DIM)
    _print(f" max cycles: {max_cycles}  model: {model}  "
           f"dry_run: {dry_run}  autonomous: {autonomous}", DIM)
    _print("─" * 60, CANDLE)

    # Per-lab observability routing — set the active-lab ContextVar so
    # canvas_emit can route SoR mirrors to <lab>/sor/events.jsonl
    # instead of always bert-lab/lab/sor/events.jsonl. Without this, no
    # researcher / verdict / model_call events ever land in a user
    # lab's events.jsonl, so Atlas roster + strata + Manuscript
    # findings stay empty for any non-default lab.
    from core.lab_context import set_active_lab_path
    set_active_lab_path(lab_path)

    # Sprint 1: load or synthesize the lab's LabSchema. The schema's
    # `roster_initial` field replaces the hardcoded `(researcher,
    # strategist)` pair as the source-of-truth for which roles to
    # dispatch per cycle. Persisted to lab/lab_schema.json so
    # subsequent runs skip re-classification.
    #
    # Safety net: BERT_LEGACY_RESEARCHER_STRATEGIST=1 forces the legacy
    # pair regardless of schema (used during the v1.0 transition; will
    # be removed in Sprint 3 once organic dispatch is solid).
    from core import lab_schema_io
    legacy_roster = (
        os.environ.get("BERT_LEGACY_RESEARCHER_STRATEGIST", "").strip()
        in {"1", "true", "yes"}
    )
    schema_roster: tuple[str, ...] | None = None
    if not legacy_roster:
        try:
            # dry-run is "plumbing only, no dispatches" — the LLM
            # classifier (`claude -p` subprocess, up to 60s + 90s) is a
            # model call, so skip it and use the heuristic classifier for
            # a fast, deterministic, network-free roster preview. Real
            # runs keep the LLM classifier for accuracy (cached after the
            # first synthesis).
            lab_schema = lab_schema_io.load_or_synthesize(
                lab_path, use_llm_classifier=not dry_run,
            )
            schema_roster = lab_schema.roster_initial
            _print(
                f"[schema] rule={lab_schema.rule_id} "
                f"roster={list(schema_roster)} workflow={lab_schema.workflow}",
                DIM,
            )
        except lab_schema_io.SchemaLoadError as exc:
            _print(f"[ABORT] {exc}", YELLOW)
            return 2
    else:
        _print(
            "[schema] BERT_LEGACY_RESEARCHER_STRATEGIST=1 — using legacy roster",
            YELLOW,
        )

    try:
        seed = _read_seed_brief(lab_path)
    except FileNotFoundError as exc:
        _print(f"[ABORT] {exc}", YELLOW)
        return 2

    _print(f"[seed] {len(seed)} bytes loaded from seed_brief.md", DIM)

    if dry_run:
        _print(f"[dry-run] would run {max_cycles} cycle(s):", YELLOW)
        start = _next_cycle_id(lab_path)
        roster_for_preview = schema_roster or _LEGACY_ROSTER
        roster_str = " → ".join(roster_for_preview)
        for i in range(max_cycles):
            if autonomous:
                _print(f"  iter {i+1}: director → cycle C{start+i}: "
                       f"{roster_str}", DIM)
            else:
                _print(f"  cycle C{start+i}: {roster_str}", DIM)
        _print("[dry-run] exiting before any model call", YELLOW)
        return 0

    keys_ok, present = _check_provider_keys()
    if not keys_ok:
        _print("[ABORT] no provider keys in env. Set at least GROQ_API_KEY.",
               YELLOW)
        return 2
    _print(f"[keys] providers present: {', '.join(present)}", DIM)

    # Watch mode: run cycles indefinitely with sleep between
    interrupted = {"caught": False}

    def _on_sigint(signum, frame):
        interrupted["caught"] = True
        _print("\n[interrupt] caught Ctrl-C — finishing current cycle", YELLOW)

    signal.signal(signal.SIGINT, _on_sigint)

    cycle_id = _next_cycle_id(lab_path)
    summaries: list[dict] = []
    completed = 0
    failures = 0
    director_decisions: list[Any] = []  # CC.4 — 3-strike check

    while completed < max_cycles:
        _print("", "")

        # GG-A-prep — honor the per-lab pause flag between iterations.
        # `/api/pause?lab=X` writes <lab>/state/paused; the loop polls
        # the flag with a 5s cadence so resume is responsive without
        # busy-waiting. Bug #2 fix: pre-GG the autonomous loop never
        # checked this flag, so the UI's pause button was decorative.
        paused_flag = lab_path / "state" / "paused"
        if paused_flag.exists():
            _print(f"[pause] lab paused (flag at {paused_flag.relative_to(LAB_ROOT)})",
                   YELLOW)
            while paused_flag.exists():
                if interrupted["caught"]:
                    break
                time.sleep(5)
            if interrupted["caught"]:
                break
            _print("[resume] lab unpaused, continuing", GREEN)

        # CC.3 — autonomous director dispatch per iteration
        decision = None
        cycle_prompt_focus = seed  # default: researcher reads full seed
        if autonomous:
            from core import director as dir_mod
            _print(f"[iter {completed+1}] director deciding…", BOLD)
            obs = dir_mod.gather_observation(
                lab_path, iteration=completed + 1
            )
            decision = dir_mod.decide_next_cycle(
                lab_path, iteration=completed + 1, observation=obs
            )
            dir_mod.emit_decision_event(lab_path, decision)
            _print(f"  decision: {decision.cycle_shape} / "
                   f"{decision.focus_area}  conf={decision.confidence_1to10}/10",
                   DIM)
            _print(f"  rationale: {decision.rationale[:120]}…", DIM)

            # CC.4 — termination guardrails
            if decision.is_terminal():
                if decision.is_complete():
                    _print("[stop] director declared MISSION COMPLETE — "
                           "terminating loop", GREEN + BOLD)
                    _print(f"  rationale: {decision.rationale[:160]}", DIM)
                    dir_mod.emit_mission_complete_event(
                        lab_path,
                        iteration=completed + 1,
                        decision=decision,
                    )
                    dir_mod.emit_termination_event(
                        lab_path,
                        iteration=completed + 1,
                        reason=dir_mod.TerminationReason.MISSION_COMPLETE,
                        detail=decision.rationale,
                    )
                else:
                    _print("[stop] director chose IDLE — terminating loop",
                           YELLOW)
                    dir_mod.emit_termination_event(
                        lab_path,
                        iteration=completed + 1,
                        reason=dir_mod.TerminationReason.DIRECTOR_IDLE,
                        detail=decision.rationale,
                    )
                break

            director_decisions.append(decision)
            if dir_mod.check_three_strike(director_decisions):
                _print("[stop] 3-strike identical decisions — terminating",
                       YELLOW)
                dir_mod.emit_termination_event(
                    lab_path,
                    iteration=completed + 1,
                    reason=dir_mod.TerminationReason.THREE_STRIKE,
                    detail=(f"last 3 decisions all chose "
                            f"{decision.cycle_shape}/{decision.focus_area}"),
                )
                break

            recent = dir_mod._read_recent_events(lab_path, n=10)
            if dir_mod.check_failure_cascade(recent):
                _print("[stop] 2 consecutive invalid dispatches — terminating",
                       YELLOW)
                dir_mod.emit_termination_event(
                    lab_path,
                    iteration=completed + 1,
                    reason=dir_mod.TerminationReason.FAILURE_CASCADE,
                )
                break

            if dir_mod.check_pending_threshold(obs.pending_count):
                _print(f"[stop] {obs.pending_count} pending approvals — "
                       f"loop pauses for operator", YELLOW)
                dir_mod.emit_termination_event(
                    lab_path,
                    iteration=completed + 1,
                    reason=dir_mod.TerminationReason.PENDING_BACKLOG,
                    detail=f"{obs.pending_count} pending blessings",
                )
                break

            # Compose the researcher's prompt from the decision
            cycle_prompt_focus = dir_mod.compose_researcher_prompt_from_decision(
                decision, seed
            )

        roster_to_dispatch = schema_roster if schema_roster else _LEGACY_ROSTER
        roster_str = " → ".join(roster_to_dispatch)
        _print(f"[cycle {cycle_id}] starting ({roster_str})", BOLD)
        result = _run_one_cycle(seed=cycle_prompt_focus, cycle=cycle_id,
                                model=model, lab_path=lab_path,
                                roster=roster_to_dispatch)
        summaries.append(result)

        for d in result["dispatches"]:
            ok = "✓" if d.get("result_valid") else "✗"
            col = GREEN if d.get("result_valid") else YELLOW
            verdict = d.get("verdict", "—")
            _print(f"  {col}{ok}{RESET}  {d['label']:12}  "
                   f"verdict={verdict}  {d['elapsed_secs']}s", "")

        if result["success"]:
            _print(f"[cycle {cycle_id}] ✓ success in {result['elapsed_secs']}s", GREEN)
        else:
            _print(f"[cycle {cycle_id}] ✗ partial in {result['elapsed_secs']}s", YELLOW)
            failures += 1

        # EE.2 — autonomous mode: grade the director's decision against the
        # cycle's actual outcome + emit a director_decision_outcome event.
        # The next iteration's director sees these via gather_observation,
        # closing the CoALA episodic-feedback loop.
        if autonomous and decision is not None:
            from core import outcome as out_mod
            graded = out_mod.grade_immediate(
                decision, result, iteration=completed + 1)
            out_mod.emit_outcome_event(lab_path, graded)
            _print(f"  outcome: {graded.label.value} ({graded.reasoning[:60]}...)",
                   DIM)

        # v3+ Phase 1d — emit a cycle_outcome rollup REGARDLESS of mode.
        # Captures per-cycle bottom-line data for retrieval-utility analysis
        # (did the cycle succeed? how many dispatches? what verdicts? how
        # many findings? concerns raised?). Best-effort, never blocks.
        try:
            from core import observability as _obs
            dispatches = result.get("dispatches", []) or []
            verdicts = [d.get("verdict") for d in dispatches if d.get("verdict")]
            _obs.emit_cycle_outcome(
                cycle_id=cycle_id,
                lab=lab_path.name if lab_path else None,
                success=bool(result.get("success")),
                elapsed_secs=float(result.get("elapsed_secs") or 0.0),
                dispatches_total=len(dispatches),
                dispatches_valid=sum(1 for d in dispatches if d.get("result_valid")),
                verdicts=verdicts,
                findings_produced=len(result.get("findings", []) or []),
            )
        except Exception:  # noqa: BLE001
            pass

        completed += 1
        cycle_id += 1

        if interrupted["caught"]:
            _print(f"[stop] honoring Ctrl-C — {completed}/{max_cycles} cycles done",
                   YELLOW)
            break

        if watch and completed >= max_cycles:
            _print(f"[watch] sleeping {watch_interval_secs}s before next cycle "
                   f"(Ctrl-C to stop)", DIM)
            max_cycles += 1  # extend the cap for the next iteration
            slept = 0
            while slept < watch_interval_secs and not interrupted["caught"]:
                time.sleep(min(1, watch_interval_secs - slept))
                slept += 1
            if interrupted["caught"]:
                break

    _print("─" * 60, CANDLE)
    if failures == 0:
        _print(f" {completed} cycle(s) completed — all green", BOLD + GREEN)
    elif failures < completed:
        _print(f" {completed} cycle(s) completed, {failures} partial",
               BOLD + YELLOW)
    else:
        _print(f" {completed} cycle(s) attempted, ALL failed", BOLD + YELLOW)
    _print("─" * 60, CANDLE)
    _print("\n  next: tools/daily_quality_report.py --date today", DIM)
    _print("        tools/director_letter.py", DIM)

    _dump_run_summary(
        summaries, os.environ.get("BERT_RUN_SUMMARY_PATH"),
        lab_name=lab_path.name if lab_path else None,
        wall_secs=round(time.monotonic() - _run_start, 1),
    )

    if failures == 0:
        return 0
    if failures < completed:
        return 1
    return 2


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", default=None,
                    help="Lab path or ~/.bert/labs/ name (default: repo's own lab/)")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help=f"Cap autonomous cycles (default: {DEFAULT_MAX_CYCLES}; "
                         f"25 with --autonomous).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Routing target (default: {DEFAULT_MODEL}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate plumbing without firing dispatches.")
    ap.add_argument("--watch", action="store_true",
                    help="Loop forever (Ctrl-C to stop), sleeping between rounds.")
    ap.add_argument("--watch-interval", type=int,
                    default=DEFAULT_WATCH_INTERVAL_SECS,
                    help=f"Seconds between rounds in --watch (default: {DEFAULT_WATCH_INTERVAL_SECS}).")
    ap.add_argument("--autonomous", action="store_true",
                    help="CC.3 — Loop-autonomous mode. Each iteration starts "
                         "with a director dispatch that picks the cycle's "
                         "shape + focus area. Termination guardrails "
                         "(3-strike, failure cascade, pending-approval "
                         "threshold) prevent runaway. Pairs naturally with "
                         "--watch + a higher --max-cycles.")
    args = ap.parse_args()

    # CC.3 — autonomous mode raises the default max_cycles ceiling.
    # `--max-cycles` default is None so user-supplied values (incl. matching
    # DEFAULT_MAX_CYCLES) are NOT clobbered by the autonomous bump.
    if args.max_cycles is None:
        args.max_cycles = 25 if args.autonomous else DEFAULT_MAX_CYCLES

    lab_path = _resolve_lab(args.lab)

    try:
        return run(
            lab_path=lab_path,
            max_cycles=args.max_cycles,
            model=args.model,
            dry_run=args.dry_run,
            watch=args.watch,
            watch_interval_secs=args.watch_interval,
            autonomous=args.autonomous,
        )
    except KeyboardInterrupt:
        _print("\n[interrupt] cancelled", YELLOW)
        return 130


if __name__ == "__main__":
    sys.exit(main())
