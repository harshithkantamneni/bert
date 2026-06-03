"""Mission profile — what kind of lab is this, derived from the seed brief.

Phase A2 of the v3 plan. Every aspect of a lab (roster, memory schema,
knowledge files, graph schema, workflow shape, output format, routing)
is derived from the mission's feature vector. This module produces that
feature vector.

Three-stage classification (quality-first per P-8):

  Stage 0  — regex/keyword precheck for explicit hints
              (e.g. "repo at /path/to/code" → data_shape=code_repo).
              Free, deterministic, ~5ms.
  Stage 1  — Haiku-4.5 single JSON call to classify into the
              MissionProfile schema. ~$0.01 per lab_start; ~3s.
              Quality-first default — beats regex-only on ambiguity.
  Stage 2  — Sonnet-4.6 retry if Stage 1 returns invalid JSON or
              low-confidence. ~$0.05; ~5% of cases.
  Stage 3  — needs_user_input clarification envelope if Stage 2
              still ambiguous. Surfaces to the host via MCP for the
              user to disambiguate. Phase C wires the resume side.

The profile is mutable (per L-7): consolidator may detect drift and
propose a reshape; user confirms via MCP `lab_reshape` (Phase C).

Profile schema is the source of truth for:
  - core.cycle_budget.estimate_budget(profile, ...)
  - core.schema_synthesizer.synthesize(profile)  (A3)
  - core.brief_assembler — uses profile.data_shape for session typing
  - tools/mcp/bert_lab.py lab_start — invokes classify_mission(text)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

LOG = logging.getLogger("bert.mission_profile")

CLASSIFIER_VERSION = "v1-haiku-first"

# ── Enumerations ──────────────────────────────────────────────────────


WORK_TYPES = (
    "discover",     # gather information, lit review, scan
    "compare",      # X vs Y analysis
    "monitor",      # ongoing watch with change detection
    "synthesize",   # write a unified view from sources
    "decide",       # recommend one course of action
    "build",        # produce code or other artifacts
    "audit",        # check for compliance / violations
    "refute",       # adversarial — disprove a claim
    "defend",       # defend a claim against attack
)

DATA_SHAPES = (
    "document_corpus",     # papers, articles, web pages, PDFs
    "code_repo",           # source code with version history
    "time_series",         # timestamped events / logs / metrics
    "tabular",             # rows-and-columns data (CSV, parquet, SQL)
    "conversational",      # multi-turn dialog transcripts
    "knowledge_graph",     # pre-existing graph data
    "multimodal",          # mixed text + numeric + images
    "numeric_simulation",  # experiment runs, parameter sweeps
)

HORIZONS = ("one_shot", "short", "medium", "ongoing")
RIGOR_LEVELS = ("informal", "cited", "falsifiable", "peer_reviewable")
OUTPUT_KINDS = (
    "report", "decision_memo", "dashboard", "pr", "paper", "feed",
)
AUDIENCES = ("self", "team", "publication", "regulator")


# ── Profile schema ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MissionProfile:
    """Structured features extracted from the mission text.

    Source of truth for downstream scaffolding. All fields have safe
    fallbacks so a low-confidence classification still produces a
    usable profile (P-8 quality-first means the scaffolding may be
    less specific, but never broken).
    """
    # Topic
    domain: str                          # 'ml_research' | 'corporate_strategy' | ...
    domain_confidence: float             # 0..1

    # Verbs
    work_types: tuple[str, ...]          # subset of WORK_TYPES
    primary_work: str                    # one of WORK_TYPES

    # When + how often
    horizon: str                         # HORIZONS
    cadence: str | None                  # 'daily'|'weekly'|'on_event'|None

    # What we produce
    output_kind: str                     # OUTPUT_KINDS
    rigor: str                           # RIGOR_LEVELS

    # Data
    data_shape: str                      # DATA_SHAPES
    expected_volume: str                 # 'small'|'medium'|'large'
    input_surfaces: tuple[str, ...]      # 'web_search'|'arxiv'|'git_repo'|'sql_db'|...

    # Audience
    audience: str                        # AUDIENCES

    # Stop condition (in natural language; director consumes)
    success_criteria: tuple[str, ...]

    # Provenance
    classified_at_cycle: int = 0
    classifier_version: str = CLASSIFIER_VERSION
    classifier_confidence: float = 0.0
    stage_used: str = "stage0"           # 'stage0'|'stage1'|'stage2'|'stage3'

    # Catch-all for adapter-specific hints (e.g., code_repo path)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml_block(self) -> str:
        """Pretty YAML for lab.yaml. Tuples serialized as YAML lists."""
        d = self.to_dict()
        lines = []
        for k, v in d.items():
            if isinstance(v, tuple):
                if not v:
                    lines.append(f"  {k}: []")
                else:
                    items = ", ".join(json.dumps(item) for item in v)
                    lines.append(f"  {k}: [{items}]")
            elif isinstance(v, dict):
                if not v:
                    lines.append(f"  {k}: {{}}")
                else:
                    lines.append(f"  {k}:")
                    for sk, sv in v.items():
                        lines.append(f"    {sk}: {json.dumps(sv)}")
            elif v is None:
                lines.append(f"  {k}: null")
            else:
                lines.append(f"  {k}: {json.dumps(v)}")
        return "\n".join(lines)


# Keyword sets for heuristic-only classification when the LLM stage
# is unavailable. Empirically lifts default_profile() accuracy from
# 17-25% (constant fallback) to ~70% on a 12-mission held-out probe.
# Order matters: first cluster to match wins for primary_work; for
# data_shape we take highest-scoring cluster.
_DATA_SHAPE_KEYWORDS: dict[str, frozenset[str]] = {
    "code_repo": frozenset({
        "code", "codebase", "repository", "repo", "source", "refactor",
        "function", "module", "class", "library", "package", "API",
        "git", "github", "PR", "pull request", "import", "linux kernel",
        "framework", "compile", "build system",
    }),
    "time_series": frozenset({
        "time series", "timestamp", "sensor", "telemetry", "metrics",
        "logs", "iot", "stream", "events over time", "monitoring data",
        "trace", "tick", "sample rate",
    }),
    "tabular": frozenset({
        "table", "tabular", "csv", "parquet", "spreadsheet", "rows",
        "columns", "database", "sql", "dataframe", "analyze", "funnel",
        "conversion", "metrics", "report", "kpi", "dashboard",
        "nps", "comparison", "compare", "compare api", "latencies",
        "benchmark numbers",
    }),
    "conversational": frozenset({
        "chat", "chatbot", "conversation", "dialogue", "dialog", "support",
        "customer service", "transcript", "turn-by-turn", "messaging",
    }),
    "knowledge_graph": frozenset({
        "knowledge graph", "graph", "relationships", "entities", "ontology",
        "network of", "connections between", "map relationships",
    }),
    "multimodal": frozenset({
        "image", "video", "audio", "multimodal", "vision", "OCR",
        "visual", "screenshot", "thumbnail", "diagram",
    }),
    "numeric_simulation": frozenset({
        "simulate", "simulation", "monte carlo", "parameter sweep",
        "scenario", "model dynamics", "ensemble", "what-if",
    }),
    "document_corpus": frozenset({
        "paper", "papers", "arxiv", "pdf", "article", "literature",
        "research", "publication", "abstract", "extract", "summarize",
        "review", "findings", "documentation",
    }),
}

_WORK_TYPE_KEYWORDS: dict[str, frozenset[str]] = {
    "monitor":    frozenset({"monitor", "watch", "track", "alert",
                              "weekly", "daily", "continuously", "ongoing"}),
    "build":      frozenset({"build", "create", "implement", "ship",
                              "develop", "make", "construct", "generate"}),
    "compare":    frozenset({"compare", "vs", "versus", "benchmark",
                              "evaluate against", "head-to-head"}),
    "audit":      frozenset({"audit", "review", "check", "compliance",
                              "verify", "inspect", "validate"}),
    "decide":     frozenset({"decide", "recommend", "choose", "select",
                              "should we", "decision"}),
    "synthesize": frozenset({"synthesize", "combine", "unify",
                              "integrate", "merge", "consolidate"}),
    "refute":     frozenset({"refute", "disprove", "challenge",
                              "falsify", "counter-argue"}),
    "defend":     frozenset({"defend", "justify", "argue for",
                              "support the claim"}),
    "discover":   frozenset({"discover", "find", "research", "explore",
                              "investigate", "scan", "survey", "lit review",
                              "literature"}),
}


def _score_categories(text: str,
                       keyword_map: dict[str, frozenset[str]]) -> dict[str, int]:
    """Count keyword matches per category. Returns {category: score}."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for category, words in keyword_map.items():
        n = sum(1 for w in words if w in lower)
        if n > 0:
            scores[category] = n
    return scores


def default_profile(mission_text: str) -> MissionProfile:
    """Heuristic classifier used when the LLM stage is unavailable.

    Previously returned a constant (document_corpus / discover) for any
    mission, giving ~25% data_shape accuracy on a held-out probe. Now
    uses keyword matching against per-category indicator sets plus the
    stage-0 regex hints (paths / URLs / SQL markers). Honest precision
    target: ~70%. The LLM classifier (Haiku, Sonnet fallback) remains
    the primary path; this is the safe fallback when both fail.
    """
    text = mission_text or ""
    hints = stage0_precheck(text)

    # Data shape — stage-0 regex hint wins over keyword match
    shape = hints.get("data_shape_hint")
    if not shape:
        shape_scores = _score_categories(text, _DATA_SHAPE_KEYWORDS)
        if shape_scores:
            shape = max(shape_scores.items(), key=lambda kv: kv[1])[0]
        else:
            shape = "document_corpus"  # remains the safe default

    # Primary work — stage-0 hint wins over keyword match (Sprint 1).
    # Pre-Sprint-1, `default_profile` always scored from keywords, even
    # when stage0_precheck had a strong signal (e.g. "build a CLI" →
    # primary_work_hint=build). That made build missions misclassify
    # when the keyword scoring tied at "discover".
    work_scores = _score_categories(text, _WORK_TYPE_KEYWORDS)
    hint_work = hints.get("primary_work_hint")
    if hint_work:
        primary_work = hint_work
        # Ensure the hint shows up in work_types even if keyword scoring missed it
        if hint_work not in work_scores:
            work_scores[hint_work] = 1
    elif work_scores:
        primary_work = max(work_scores.items(), key=lambda kv: kv[1])[0]
    else:
        primary_work = "discover"
    # Build the work_types tuple from anything with non-zero score, with
    # primary first
    work_types: tuple[str, ...] = tuple(
        [primary_work] + [w for w in work_scores if w != primary_work]
    ) or ("discover",)

    # Heuristic horizon — keyword-driven
    lower = text.lower()
    if any(w in lower for w in ("weekly", "daily", "monitor",
                                  "continuously", "ongoing")):
        horizon = "ongoing"
    elif any(w in lower for w in ("one-shot", "single",
                                    "just answer", "quick")):
        horizon = "one_shot"
    else:
        horizon = "short"

    # Cadence
    cadence: str | None = None
    if "daily" in lower:
        cadence = "daily"
    elif "weekly" in lower:
        cadence = "weekly"

    # Input surfaces from hints
    input_surfaces: tuple[str, ...] = ("web_search",)
    if "arxiv" in lower or hints.get("input_surfaces_hint") == ["arxiv"]:
        input_surfaces = ("arxiv", "web_search")
    if shape == "code_repo":
        input_surfaces = ("git_repo",)
    if shape == "tabular":
        input_surfaces = ("sql_db", "csv_files")

    return MissionProfile(
        domain="general",
        domain_confidence=0.3,                 # mid — heuristic but not blind
        work_types=work_types,
        primary_work=primary_work,
        horizon=horizon,
        cadence=cadence,
        output_kind="report",
        rigor="cited",
        data_shape=shape,
        expected_volume="medium",
        input_surfaces=input_surfaces,
        audience="self",
        success_criteria=(
            f"Produce a defensible answer to: {text[:200]}",
        ),
        classifier_confidence=0.4,             # heuristic, not LLM
        stage_used="default_heuristic",
    )


# ── Stage 0: regex precheck ───────────────────────────────────────────


_PATH_HINT = re.compile(r"(?:repo|codebase|repository|directory|folder)\s*"
                        r"(?:at|in|under)?\s*([/~][\w./-]+)", re.IGNORECASE)
_URL_HINT = re.compile(r"https?://[\w./?#&=%-]+")
_SQL_HINT = re.compile(r"(?:database|sql|table)\s*[\w_]+|\.sqlite\b|\.duckdb\b",
                       re.IGNORECASE)
_PARQUET_HINT = re.compile(r"\.parquet\b|\.csv\b|\.jsonl?\b", re.IGNORECASE)
# Sprint 1: detect build-shape missions. When code signals are present,
# they take precedence over .jsonl-style "tabular" matches (the file
# is being PROCESSED by code, not classified as a data source).
_CODE_BUILD_HINT = re.compile(
    r"\b(?:build|implement|refactor|develop)\s+(?:a\s+)?"
    r"(?:python|cli|utility|tool|module|class|function|script)\b"
    r"|\.py\b|\.ts\b|\.js\b|\.go\b|\.rs\b|\bpytest\b|\bunittest\b"
    r"|\btools/[\w_]+\.py\b|\btests/test_[\w_]+\.py\b|\bsrc/[\w_/]+\b",
    re.IGNORECASE,
)
# Sprint 1: detect audit/analysis-shape missions. Walking files +
# producing a ledger/audit isn't tabular data work — it's document
# corpus work. Tight signals only — generic "review" is too loose
# (research missions say "review the researcher's brief" all the time).
_AUDIT_HINT = re.compile(
    r"\baudit\s+(?:every|each|all|the|my|our|this|files?\b|findings?\b|"
    r"codebase|corpus|claims?\b|ledger|directory|folder)\b"
    r"|\bstale[ -]claim\b|\bstale[ -]?claims?\s+ledger\b"
    r"|\bchain[ -]of[ -]custody\b"
    r"|\bevidence\s+ledger\b|\bcompliance\s+audit\b",
    re.IGNORECASE,
)


def stage0_precheck(mission_text: str) -> dict[str, Any]:
    """Cheap regex extraction of explicit hints. Returns a partial dict
    used to bias Stage 1 and to detect when Stage 1 can be skipped.
    Never raises.

    Sprint 1 (v1.0) — added code_build + audit detection so the
    heuristic differentiates research/build/audit even when the LLM
    classifier is unavailable. Pre-Sprint-1, .jsonl mentions hijacked
    everything to data_shape=tabular.
    """
    hints: dict[str, Any] = {}
    text = mission_text or ""

    # Code build signals are HIGH-PRIORITY (set first; not overridden by
    # tabular-style hints later in this function). Build missions
    # frequently mention .jsonl/.csv files as INPUTS to code without
    # the mission itself being tabular data work.
    if _CODE_BUILD_HINT.search(text):
        hints["data_shape_hint"] = "code_repo"
        hints["primary_work_hint"] = "build"

    # Audit / analysis signals — also high-priority, sets document_corpus.
    elif _AUDIT_HINT.search(text):
        hints["data_shape_hint"] = "document_corpus"
        hints["primary_work_hint"] = "audit"

    # Code repo path? (overrides if explicit path given)
    m = _PATH_HINT.search(text)
    if m:
        hints["repo_path"] = m.group(1)
        hints["data_shape_hint"] = "code_repo"

    # URLs?
    urls = _URL_HINT.findall(text)
    if urls:
        hints["urls"] = urls[:5]
        if any("arxiv" in u for u in urls):
            hints["input_surfaces_hint"] = ["arxiv"]
        elif any("github" in u for u in urls):
            hints["data_shape_hint"] = "code_repo"

    # SQL / tabular hints (LOWER priority than code_build / audit;
    # only fire if no higher-signal hint was set above).
    if "data_shape_hint" not in hints:
        if _SQL_HINT.search(text) or _PARQUET_HINT.search(text):
            hints["data_shape_hint"] = "tabular"
    return hints


# ── Stage 1: Haiku JSON classifier ────────────────────────────────────


_CLASSIFIER_PROMPT_TEMPLATE = """You are a classification assistant for bert,
an autonomous lab framework. Your job is to extract STRUCTURED FEATURES
from a research/work mission so bert can scaffold the right lab shape.

Read the mission below and emit a single JSON object matching this exact
schema. No prose, no commentary, no markdown — JUST the JSON object.

Schema:
{{
  "domain": "<short slug like 'ml_research' | 'corporate_strategy' | 'mobile_dev' | 'medical_finance' | 'legal_due_diligence' | 'code_quality' | 'general' | ...>",
  "domain_confidence": <float 0..1>,
  "work_types": [<one or more of: "discover", "compare", "monitor", "synthesize", "decide", "build", "audit", "refute", "defend">],
  "primary_work": "<one of the work_types>",
  "horizon": "<one of: 'one_shot' | 'short' | 'medium' | 'ongoing'>",
  "cadence": <"daily"|"weekly"|"on_event"|null>,
  "output_kind": "<one of: 'report' | 'decision_memo' | 'dashboard' | 'pr' | 'paper' | 'feed'>",
  "rigor": "<one of: 'informal' | 'cited' | 'falsifiable' | 'peer_reviewable'>",
  "data_shape": "<one of: 'document_corpus' | 'code_repo' | 'time_series' | 'tabular' | 'conversational' | 'knowledge_graph' | 'multimodal' | 'numeric_simulation'>",
  "expected_volume": "<'small' (<100 items) | 'medium' (100-10k) | 'large' (10k+)>",
  "input_surfaces": [<list of: "web_search", "arxiv", "git_repo", "sql_db", "pdf_dir", "internal_api", "live_stream", "user_input", ...>],
  "audience": "<one of: 'self' | 'team' | 'publication' | 'regulator'>",
  "success_criteria": [<one to four natural-language strings stating what 'done' looks like>],
  "classifier_confidence": <float 0..1 — your overall confidence in this classification>
}}

Rules:
- Use 'ongoing' horizon ONLY when the lab itself runs continuously
  (e.g. "monitor X weekly"), not when the SUBJECT is ongoing.
- Use 'code_repo' data_shape ONLY when the lab actually works on a
  codebase, not when the subject mentions code.
- If the mission is ambiguous, set classifier_confidence < 0.6.

Hints extracted from the mission (use these to bias your output):
{hints_json}

Mission:
\"\"\"
{mission}
\"\"\"

JSON:"""


def _haiku_classify(mission_text: str, hints: dict) -> dict | None:
    """Call Claude Haiku via the CLI bridge to classify.
    Returns parsed dict on success, None on failure (caller falls back)."""
    prompt = _CLASSIFIER_PROMPT_TEMPLATE.format(
        mission=mission_text,
        hints_json=json.dumps(hints, indent=2),
    )
    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--model", "haiku",
                "--output-format", "json",
                "--permission-mode", "default",
                prompt,
            ],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        LOG.warning("haiku classifier subprocess failed: %s", e)
        return None
    if result.returncode != 0:
        LOG.warning("haiku classifier exit %d: %s",
                    result.returncode, result.stderr[:200])
        return None
    try:
        cli_out = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if cli_out.get("is_error"):
        LOG.warning("haiku classifier is_error: %s",
                    cli_out.get("result", "")[:200])
        return None
    body = cli_out.get("result", "")
    # The model's reply is the JSON object; extract it
    parsed = _extract_first_json(body)
    return parsed


def _extract_first_json(text: str) -> dict | None:
    """Extract first top-level JSON object from a text blob."""
    if not text:
        return None
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Look for a {...} block
    start = text.find("{")
    if start < 0:
        return None
    # Find matching closing brace; tolerate nested braces
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Stage 2: Sonnet retry ─────────────────────────────────────────────


def _sonnet_classify(mission_text: str, hints: dict,
                     prior_attempt: dict | None) -> dict | None:
    """Retry classification with Sonnet when Haiku produced invalid /
    low-confidence output. Same prompt; better model."""
    extra = ""
    if prior_attempt:
        extra = (
            f"\nA prior classifier returned this — review and improve "
            f"if needed:\n{json.dumps(prior_attempt, indent=2)}\n"
        )
    prompt = _CLASSIFIER_PROMPT_TEMPLATE.format(
        mission=mission_text,
        hints_json=json.dumps(hints, indent=2),
    ) + extra
    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--model", "sonnet",
                "--output-format", "json",
                "--permission-mode", "default",
                prompt,
            ],
            capture_output=True, text=True, timeout=90,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        LOG.warning("sonnet classifier subprocess failed: %s", e)
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        cli_out = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if cli_out.get("is_error"):
        return None
    return _extract_first_json(cli_out.get("result", ""))


# ── Validation + coercion ─────────────────────────────────────────────


def _coerce_to_profile(raw: dict, mission_text: str,
                       *, stage: str,
                       extras: dict | None = None) -> MissionProfile | None:
    """Validate a raw classifier dict into a MissionProfile.
    Returns None if required fields are missing/invalid."""
    if not isinstance(raw, dict):
        return None
    try:
        work_types_raw = raw.get("work_types") or []
        if isinstance(work_types_raw, str):
            work_types_raw = [work_types_raw]
        work_types = tuple(
            w for w in work_types_raw if w in WORK_TYPES
        )
        if not work_types:
            work_types = ("discover",)

        primary_work = raw.get("primary_work")
        if primary_work not in WORK_TYPES:
            primary_work = work_types[0]

        horizon = raw.get("horizon")
        if horizon not in HORIZONS:
            horizon = "short"

        data_shape = raw.get("data_shape")
        if data_shape not in DATA_SHAPES:
            data_shape = "document_corpus"

        output_kind = raw.get("output_kind")
        if output_kind not in OUTPUT_KINDS:
            output_kind = "report"

        rigor = raw.get("rigor")
        if rigor not in RIGOR_LEVELS:
            rigor = "cited"

        audience = raw.get("audience")
        if audience not in AUDIENCES:
            audience = "self"

        volume = raw.get("expected_volume", "medium")
        if volume not in ("small", "medium", "large"):
            volume = "medium"

        cadence = raw.get("cadence")
        if cadence not in ("daily", "weekly", "on_event", None):
            cadence = None

        input_surfaces_raw = raw.get("input_surfaces") or []
        if isinstance(input_surfaces_raw, str):
            input_surfaces_raw = [input_surfaces_raw]
        input_surfaces = tuple(
            s for s in input_surfaces_raw
            if isinstance(s, str) and len(s) <= 50
        )

        success_criteria_raw = raw.get("success_criteria") or []
        if isinstance(success_criteria_raw, str):
            success_criteria_raw = [success_criteria_raw]
        success_criteria = tuple(
            str(s)[:400] for s in success_criteria_raw if s
        )
        if not success_criteria:
            success_criteria = (
                f"Produce a defensible answer to: {mission_text[:200]}",
            )

        domain = raw.get("domain") or "general"
        domain_conf = _clamp01(raw.get("domain_confidence", 0.5))
        confidence = _clamp01(raw.get("classifier_confidence", 0.5))

        return MissionProfile(
            domain=str(domain)[:60],
            domain_confidence=domain_conf,
            work_types=work_types,
            primary_work=primary_work,
            horizon=horizon,
            cadence=cadence,
            output_kind=output_kind,
            rigor=rigor,
            data_shape=data_shape,
            expected_volume=volume,
            input_surfaces=input_surfaces,
            audience=audience,
            success_criteria=success_criteria,
            classifier_confidence=confidence,
            stage_used=stage,
            extras=dict(extras or {}),
        )
    except Exception as e:  # noqa: BLE001
        LOG.warning("profile coercion failed: %s", e)
        return None


def _clamp01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.5


# ── Public API ────────────────────────────────────────────────────────


CONFIDENCE_FLOOR_STAGE1 = 0.6   # below this, Stage 2 retries
CONFIDENCE_FLOOR_STAGE2 = 0.5   # below this, Stage 3 clarifies


def classify_mission(
    mission_text: str,
    *,
    use_llm: bool = True,
    cycle: int = 0,
) -> MissionProfile:
    """Top-level classifier. Stage 0 → Stage 1 → Stage 2 → safe default.

    Returns a MissionProfile. If `use_llm` is False, falls back to
    Stage-0 hints + default_profile (useful for tests + offline runs).
    """
    if not mission_text or not mission_text.strip():
        return default_profile("")._replace_or_clone(classified_at_cycle=cycle) \
            if hasattr(default_profile(""), "_replace_or_clone") \
            else default_profile("")

    # Stage 0: regex precheck (always)
    hints = stage0_precheck(mission_text)

    # Stage 1: Haiku (if LLM enabled)
    profile = None
    if use_llm:
        raw = _haiku_classify(mission_text, hints)
        if raw:
            profile = _coerce_to_profile(
                raw, mission_text, stage="stage1_haiku",
                extras={"hints": hints},
            )
        if profile and profile.classifier_confidence < CONFIDENCE_FLOOR_STAGE1:
            LOG.info("haiku confidence %.2f below floor; escalating to sonnet",
                     profile.classifier_confidence)
            raw2 = _sonnet_classify(mission_text, hints, raw)
            if raw2:
                profile = _coerce_to_profile(
                    raw2, mission_text, stage="stage2_sonnet",
                    extras={"hints": hints, "prior_haiku": raw},
                ) or profile

    # Final fallback
    if profile is None:
        profile = default_profile(mission_text)
        # Apply stage 0 hints if any
        if hints.get("data_shape_hint") in DATA_SHAPES:
            profile = MissionProfile(
                **{**profile.to_dict(),
                   "data_shape": hints["data_shape_hint"],
                   "stage_used": "stage0_hints_only",
                   "extras": {"hints": hints}}
            )

    # Stamp cycle
    if cycle and profile.classified_at_cycle == 0:
        profile = MissionProfile(
            **{**profile.to_dict(), "classified_at_cycle": cycle}
        )

    return profile


def is_ambiguous(profile: MissionProfile) -> bool:
    """True if profile's confidence is low enough to warrant
    needs_user_input clarification (Stage 3)."""
    return profile.classifier_confidence < CONFIDENCE_FLOOR_STAGE2


# ── CLI for smoke testing ─────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """  python -m core.mission_profile classify "<mission text>"
        python -m core.mission_profile classify-offline "<mission text>"
    """
    if len(argv) < 3 or argv[1] not in ("classify", "classify-offline"):
        print('usage: mission_profile classify "<mission>" '
              '[or] classify-offline "<mission>"', file=sys.stderr)
        return 2
    use_llm = argv[1] == "classify"
    mission = argv[2]
    profile = classify_mission(mission, use_llm=use_llm)
    print(json.dumps(profile.to_dict(), indent=2))
    print("\n---\nis_ambiguous:", is_ambiguous(profile))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
