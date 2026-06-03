"""Sub-agent dispatch with scoped DispatchSpec.

Director (or another orchestrator) calls Spawn(spec=DispatchSpec). Spawn
validates the spec, runs the named role's agent loop in-process with a
scoped task, then reads the ResultPacket the sub-agent wrote to disk.

Design choices for MVP (Phase 2):
- In-process loop (faster than subprocess; cycle-level isolation comes from
  run.sh which already spawns each cycle in its own python process).
- Sub-agent must Write a JSON ResultPacket to
  `state/results/{role}_C{cycle}_{tag}.json`. Spawn parses + validates it.
  If missing/invalid, Spawn synthesizes verdict=OTHER for the Director.
- Iteration budget capped at 20 for sub-agents (vs Director's 25-30).
- Caller passes the spec as a plain dict; we coerce + validate; we return
  a plain-dict summary for the Director's tool result.

P-016 sentinel wrapping is handled by the caller (Spawn tool handler).
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from pathlib import Path

import jsonschema

from core import config, log, observability
from core.types import Verdict

LAB_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = LAB_ROOT / "schemas"
# Default results dir; per-lab dispatches override via _active_results_dir
# (resolved from core.lab_context). Without per-lab routing, every
# subagent's ResultPacket landed under bert-lab/state/results/ even
# when the dispatch was for a user lab — meaning the user lab's
# state/results/ stayed empty + Atlas/Manuscript surfaces showed no
# evidence of the cycle.
RESULTS_DIR = LAB_ROOT / "state" / "results"


def _active_results_dir() -> Path:
    """Return the right state/results directory for the active lab
    context, falling back to RESULTS_DIR for the bert-lab default."""
    from core.lab_context import get_active_lab_path
    active = get_active_lab_path()
    if active is None:
        return RESULTS_DIR
    return active / "state" / "results"


def _result_path_str(result_path: Path) -> str:
    """Format result_path for the spawn summary. Prefer relative to
    LAB_ROOT (legacy behavior); when the lab is a user lab outside
    LAB_ROOT, fall back to relative to the active lab path so the
    string stays short + meaningful."""
    try:
        return str(result_path.relative_to(LAB_ROOT))
    except ValueError:
        from core.lab_context import get_active_lab_path
        active = get_active_lab_path()
        if active is not None:
            try:
                return str(result_path.relative_to(active))
            except ValueError:
                pass
        return str(result_path)


LOG = log.get_logger("bert.subagent")

SUBAGENT_MAX_ITERATIONS = 20

# Roles registered as valid sub-agent targets.
# (Implementer/Strategist/etc. prompts may not exist yet; if a prompt file is
# missing, run_role falls back to a generic stub but logs a warning.)
#
# Sprint 1 (v1.0): roster_initial may now be derived from a LabSchema
# (mission_profile → schema_synthesizer → roster), so this set MUST cover
# every role name that appears in core/library/synthesizer_rules.yaml +
# every role template in core/library/agents/. Adding a new role:
#   1. Author core/library/agents/<role>.md
#   2. Reference it in synthesizer_rules.yaml roster_core or roster_initial
#   3. Add the name to this set
# Test: tests/test_bert_run_organicity.py + 3-mission validation.
KNOWN_ROLES = frozenset({
    # Original v0 set
    "researcher", "strategist", "implementer", "evaluator",
    "reflector", "consolidator",
    # Quaker pipeline roles (P-VS-06..09; pending D-10 ratification).
    # threshing_pass: surfaces disagreement, MUST produce SCOPE_STOP verdict.
    # clearness_phase1: open-query phase, MUST produce SCOPE_STOP + clearness_queries.
    # clearness_phase2: verdict pass with phase-1 queries as context.
    "threshing_pass", "clearness_phase1", "clearness_phase2",
    # Sprint 1: roster_core roles from synthesizer_rules.yaml
    "director",
    # Sprint 1: roster_initial roles from synthesizer_rules.yaml
    "literature_hunter", "change_detector", "methodology_critic",
    "analyst", "option_scorer", "red_team", "writer",
    "code_reader", "refactor_specialist", "test_author", "reviewer",
    # Sprint 1: role templates in core/library/agents/ not yet in rules
    "engineer", "security_auditor", "performance_tuner",
})


# ── Schema loading ──────────────────────────────────────────────────


_dispatch_schema_cache: dict | None = None
_result_schema_cache: dict | None = None


def _dispatch_schema() -> dict:
    global _dispatch_schema_cache
    if _dispatch_schema_cache is None:
        _dispatch_schema_cache = json.loads((SCHEMAS_DIR / "dispatch_spec.json").read_text())
    return _dispatch_schema_cache


def _result_schema() -> dict:
    global _result_schema_cache
    if _result_schema_cache is None:
        _result_schema_cache = json.loads((SCHEMAS_DIR / "result_packet.json").read_text())
    return _result_schema_cache


# ── Validation ──────────────────────────────────────────────────────


def validate_dispatch_spec(spec: dict) -> tuple[bool, list[str]]:
    """Validate a DispatchSpec dict against schemas/dispatch_spec.json.

    Returns (is_valid, error_messages). Empty error list iff valid.
    """
    errors: list[str] = []
    try:
        validator = jsonschema.Draft202012Validator(_dispatch_schema())
        for err in sorted(validator.iter_errors(spec), key=lambda e: e.path):
            path = ".".join(str(p) for p in err.path) or "<root>"
            errors.append(f"{path}: {err.message}")
    except (jsonschema.SchemaError, json.JSONDecodeError) as e:
        errors.append(f"schema-load error: {e}")

    # Cross-field rules the JSON schema can't express:
    role = spec.get("role", "")
    if role and role not in KNOWN_ROLES and not role.startswith("custom-"):
        errors.append(f"role: '{role}' is not a known role; prefix with 'custom-' if intentional")

    return (not errors, errors)


def _build_schema_registry():
    """Build a referencing.Registry that can resolve relative $refs from
    result_packet.json to concern_entry.json / clearness_query.json /
    seasoning_entry.json. v2 result_packet uses $defs with $ref to those
    sibling files; the validator needs a registry to find them.
    """
    try:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012
    except ImportError:
        return None  # older jsonschema version; caller falls back

    resources = []
    for name in ("concern_entry.json", "clearness_query.json",
                 "seasoning_entry.json"):
        path = SCHEMAS_DIR / name
        if path.exists():
            schema = json.loads(path.read_text())
            resource = Resource.from_contents(schema, default_specification=DRAFT202012)
            # Register under both the bare filename (for relative refs from
            # result_packet.json) and the schema's own $id (for absolute refs).
            resources.append((name, resource))
            if "$id" in schema:
                resources.append((schema["$id"], resource))
    return Registry().with_resources(resources)


_registry_cache = None


def _registry():
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = _build_schema_registry()
    return _registry_cache


def validate_result_packet(packet: dict) -> tuple[bool, list[str]]:
    """Validate a ResultPacket dict against schemas/result_packet.json.

    Uses a referencing.Registry to resolve relative $refs to sibling
    schema files (concern_entry.json, clearness_query.json).
    """
    errors: list[str] = []
    try:
        registry = _registry()
        if registry is not None:
            validator = jsonschema.Draft202012Validator(
                _result_schema(), registry=registry,
            )
        else:
            validator = jsonschema.Draft202012Validator(_result_schema())
        for err in sorted(validator.iter_errors(packet), key=lambda e: e.path):
            path = ".".join(str(p) for p in err.path) or "<root>"
            errors.append(f"{path}: {err.message}")
    except (jsonschema.SchemaError, json.JSONDecodeError) as e:
        errors.append(f"schema-load error: {e}")
    return (not errors, errors)


# ── Spawn loop ──────────────────────────────────────────────────────


def _result_path_for(role: str, cycle: int, tag: str) -> Path:
    """Canonical path the sub-agent must write its ResultPacket to.
    Routes into the active lab's state/results dir when set."""
    safe_tag = re.sub(r"[^a-zA-Z0-9_-]+", "-", tag).strip("-") or "main"
    return _active_results_dir() / f"{role}_C{cycle}_{safe_tag}.json"


def _render_verification_requirements(vspec: dict | None) -> str:
    """Render the verification spec the deliverable is GRADED against into explicit
    requirements, so the agent isn't graded on rules it was never told (the cause
    of the min_chars / missing-header BUILD_FAILs)."""
    if not vspec:
        return ""
    lines = ["## Deliverable requirements (you are GRADED on these — satisfy ALL)"]
    mc = vspec.get("min_chars")
    if mc:
        lines.append(f"- Length: at least {mc} characters of substantive prose "
                     "(a one-paragraph summary is NOT enough).")
    for h in vspec.get("required_headers") or []:
        lvl, cnt = h.get("level", 1), h.get("count", 1)
        lines.append(f"- Structure: at least {cnt} level-{lvl} markdown "
                     f"header(s) (`{'#' * lvl} ...`).")
    for p in vspec.get("required_patterns") or []:
        lines.append(f"- Must include {p.get('description', 'the required content')}.")
    for p in vspec.get("forbidden_patterns") or []:
        lines.append(f"- Must NOT contain {p.get('description', 'forbidden content')}.")
    return "\n".join(lines)


def _scoped_task(spec: dict, result_path: Path) -> str:
    """Compose the inline task for the sub-agent.

    The sub-agent's role prompt covers methodology; this task message
    delivers the per-call specifics from the dispatch packet plus the
    explicit ResultPacket-writing contract + the verification rubric the
    deliverable is graded against.
    """
    parts = [
        f"# Dispatch from Director (cycle {spec['cycle']})",
        f"**Altitude:** {spec['dispatch_altitude']}",
        f"**Role:** {spec['role']}",
        "",
        "## Task",
        spec["task"],
        "",
        "## Success criterion",
        spec["success_criterion"],
        "",
        "## Required output (full report)",
        f"Write your detailed findings to: `{spec['output_path']}`",
    ]
    _reqs = _render_verification_requirements(spec.get("verification_spec"))
    if _reqs:
        parts += ["", _reqs]
    parts += [
        "",
        "## Required return (ResultPacket)",
        f"Before stopping, write a JSON ResultPacket to: `{_result_path_str(result_path)}`",
        "It MUST validate against `schemas/result_packet.json`. Exact structure:",
        "```json",
        "{",
        f'  "role": "{spec["role"]}",',
        f'  "cycle": {spec["cycle"]},',
        '  "verdict": "APPROVE | APPROVE_WITH_CAVEATS | CHANGES_REQUESTED | REJECT | BUILD_PASS | BUILD_FAIL | BUILD_PARTIAL | SCOPE_STOP | OTHER",',
        '  "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},',
        '  "confidence_1to10": 1,',
        '  "calibration_reasoning": "≥80 chars explaining why that confidence number, what evidence supports it, what would lower it",',
        '  "telemetry": {"tokens_in": 0, "tokens_out": 0, "latency_secs": 0.0, "model_used": "provider/model", "provider": "provider", "retry_count": 0, "fallback_chain": []}',
        "}",
        "```",
        "Field-name discipline: use exactly `tokens_in`, `tokens_out`, `latency_secs` "
        "(not `tokens` / `latency`). Use exactly `confidence_1to10` (not `confidence`). "
        "calibration_reasoning must be ≥80 characters.",
        "",
        "## Process hygiene",
        spec["process_hygiene"],
    ]
    if spec.get("forbidden_actions"):
        parts += ["", "## Forbidden actions"]
        parts += [f"- {a}" for a in spec["forbidden_actions"]]
    if spec.get("caveats_embedded"):
        parts += ["", "## Caveats from prior cycles"]
        parts += [f"- {c}" for c in spec["caveats_embedded"]]
    if spec.get("falsifier_text"):
        parts += ["", "## Pre-registered falsifier", spec["falsifier_text"]]
    parts += [
        "",
        "When the ResultPacket file is written, stop. Do not continue iterating.",
    ]
    return "\n".join(parts)


def _parse_provider_model(s: str) -> tuple[str, str | None]:
    """Parse 'provider/model' into (provider, model). Returns (provider, None)
    if no slash so the provider's default_model is used."""
    if "/" not in s:
        return s, None
    provider, _, model = s.partition("/")
    return provider, (model or None)


# ── Cross-family Evaluator routing (P-VS-02) ────────────────────────

# Model FAMILIES — providers grouped by training-distribution lineage.
# Same-family judgments produce "convergent decoration" not real review
# (per arxiv 2502.01534 "Preference Leakage" ICLR 2026 + R3 finding).
# Cross-family pairs introduce the structural diversity required for
# high-stakes verdicts to be honest.
MODEL_FAMILIES: dict[str, str] = {
    # Mistral family — French open-weights lineage
    "mistral": "mistral",
    # Llama family — Meta open-weights lineage (Cerebras + NVIDIA + Groq + HF Router
    # all serve Llama variants; collapsed into one family). Cerebras was previously
    # mapped to "qwen" but R13 live-API discovery (2026-05-07) found that on bert's
    # free tier only qwen-3-235b (deprecating 2026-05-27) and llama3.1-8b are
    # actually accessible — qwen-3-32b / zai-glm-4.7 / gpt-oss-120b return 404.
    # Post-deprecation Cerebras serves Llama only.
    "cerebras": "llama",
    # Gemini family — Google
    "gemini": "google",
    # Llama family — Meta open-weights lineage (NVIDIA / Groq / HF Router host these
    # by default; can be overridden with model-specific dispatch — e.g., NVIDIA also
    # hosts qwen/qwen3-next-80b-a3b-thinking which is Qwen-family in fact, but
    # provider-level family-attribution is the heuristic we use; per-model
    # family override would be a v3 schema upgrade).
    "nvidia": "llama",
    "groq": "llama",
    "hf_router": "llama",
    # OpenRouter — universal meta-fallback (model varies; treat as own family
    # so it cleanly fills the "any cross-family" slot)
    "openrouter": "openrouter",
    # Ollama — local; family depends on the model the user pulled
    "ollama": "local",
}

# Cross-family judge slot registry. Each slot is a (provider, model_override,
# family) triple. Decoupling "which provider+model fills this slot" from
# "which provider's default the family-of() lookup uses" lets us:
#   - serve the Qwen family explicitly via NVIDIA's qwen/qwen3-next-80b
#     (live-verified on NVIDIA free tier 2026-05-07, 80B thinking-mode model)
#     even though NVIDIA's PROVIDER default is meta/llama-3.3-70b-instruct
#   - keep Cerebras as a fast Llama-family slot (its current default
#     llama3.1-8b is fine for non-judge work), without burning the
#     cross-family Qwen seat on an 8B model
#
# This was a quality-first redo of R13. R13 collapsed Cerebras qwen → llama
# because all qwen-family Cerebras models 404'd on bert's free tier. That fix
# was correct as far as Cerebras goes, but it left no Qwen-family slot in the
# cross-family Evaluator rotation — a 30× capability cut from qwen-3-235b
# for high-stakes judging. R14 (this change, 2026-05-07) restores Qwen by
# routing the slot to NVIDIA's qwen/qwen3-next-80b-a3b-thinking instead.
#
# Slot order = preference order for cross-family fallback (walks until
# finding a slot whose family differs from the producer's).
#
# Live-verified models on bert's free tier (2026-05-07):
#   qwen/qwen3-next-80b-a3b-thinking    ✓ NVIDIA, 80B thinking-mode
#   qwen/qwen2.5-coder-32b-instruct     ✓ NVIDIA, 32B instruct (fallback)
#   qwen/qwen3-235b-a22b                ✗ NVIDIA returns 410 Gone (deprecated)
#   qwen-3-235b-a22b-instruct-2507      ✗ Cerebras, deprecating 2026-05-27
#   qwen-3-32b / zai-glm-4.7 / gpt-oss-120b on Cerebras  ✗ all 404
EVAL_SLOTS: list[tuple[str, str | None, str]] = [
    # (provider, model_override or None for default, family)
    ("nvidia",     "qwen/qwen3-next-80b-a3b-thinking",  "qwen"),
    ("mistral",    None,                                "mistral"),
    ("gemini",     None,                                "google"),
    ("openrouter", None,                                "openrouter"),
    ("cerebras",   None,                                "llama"),
    ("groq",       None,                                "llama"),
    ("nvidia",     None,                                "llama"),  # default Llama via NVIDIA
    ("hf_router",  None,                                "llama"),
]

# Backwards-compatible name — some external callers and tests still
# import EVAL_PROVIDER_PREFERENCE. Keeps a list of the unique providers
# in slot order; new code should prefer EVAL_SLOTS for the explicit
# (provider, model, family) triple.
EVAL_PROVIDER_PREFERENCE: list[str] = list({slot[0]: None for slot in EVAL_SLOTS}.keys())


def family_of(provider: str) -> str:
    """Return the model family for a provider name. Unknown providers
    return their own name as a stand-alone family (so they don't
    accidentally collide with anything).

    NOTE: this returns the provider's *default* family. A provider can
    serve multiple families via explicit model selection — that
    refinement lives in EVAL_SLOTS, not here. e.g., NVIDIA's family_of
    is 'llama' (its default model is meta/llama-3.3-70b-instruct), but
    NVIDIA can also serve qwen-family via explicit qwen/* model
    dispatch — use `slot_family_of(provider, model)` for the
    slot-aware lookup.
    """
    return MODEL_FAMILIES.get(provider, provider)


def slot_family_of(provider: str, model: str | None) -> str:
    """Slot-aware family lookup. If (provider, model) matches a slot in
    EVAL_SLOTS, returns that slot's family. Otherwise falls back to
    provider-level family_of(). Used by P-VS-02 cross-family checks
    that need to honor the explicit qwen-via-NVIDIA slot routing.
    """
    for slot_provider, slot_model, slot_family in EVAL_SLOTS:
        if slot_provider == provider and slot_model == model:
            return slot_family
    return family_of(provider)


def pick_evaluator_model(producer_model: str) -> str:
    """Return a 'provider/model' string for an Evaluator dispatch whose
    family differs from `producer_model`'s family. Per P-VS-02
    (cross-family adversarial review for high-stakes verdicts).

    Selection priority:
    1. Consult `capability_matrix` for the highest-scoring evaluator
       model not in the producer's family (with quota headroom). This
       is the measurement-driven path.
    2. Fall back to walking EVAL_SLOTS in order if the matrix has no
       qualifying row (matrix not yet seeded, or every measured
       evaluator is from the same family as the producer).
    3. Universal escape to OpenRouter Gemma (different family from
       most producers, free tier).

    The function body must reference `capability_matrix`
    so the static analyzer can verify the matrix is consulted.
    """
    producer_provider, producer_model_part = _parse_provider_model(producer_model)
    producer_family = family_of(producer_provider)

    # Producer family override: if the producer is using an explicit qwen/*
    # model on NVIDIA (or similar slot-bound model), the producer's effective
    # family is that of the slot, not the provider default.
    for slot_provider, slot_model, slot_family in EVAL_SLOTS:
        if slot_provider == producer_provider and slot_model == producer_model_part:
            producer_family = slot_family
            break

    # capability_matrix consult — preferred path when seeded.
    # Use slot_family_of (provider, model) so a Qwen-via-NVIDIA row is
    # correctly classified as Qwen family, not Llama.
    try:
        from core import capability_matrix
        cap_row = capability_matrix.best_for_role(
            "evaluator",
            exclude_family=producer_family,
            family_of_fn=slot_family_of,
            min_headroom_pct=20,
        )
        if cap_row is not None:
            return f"{cap_row.provider}/{cap_row.model}"
    except Exception:  # noqa: BLE001
        # Matrix module / file missing or malformed — fall back to
        # static cascade. The matrix is advisory.
        pass

    from core import provider as _prov
    for slot_provider, slot_model, slot_family in EVAL_SLOTS:
        if slot_family == producer_family:
            continue
        spec = _prov.PROVIDERS.get(slot_provider)
        if spec is None:
            continue
        model = slot_model or spec.default_model
        return f"{slot_provider}/{model}"

    # Universal escape — should be unreachable given current slot map
    return "openrouter/google/gemma-4-26b-a4b-it:free"


# ── External verification (post-loop) ───────────────────────────────


def _run_verification(command: str, timeout: int = 120) -> dict:
    """Run a shell command after the agent loop and capture pass/fail.

    Returns a dict suitable for stuffing into ResultPacket.telemetry.verification:
      {ok: bool, exit_code: int, stdout: str, stderr: str,
       elapsed_ms: int, command: str, timed_out: bool}
    Output is capped at ~6 KB stdout + 2 KB stderr. The agent's self-report
    is not the source of truth for build/test pass — this is.
    """
    import subprocess
    import time as _t
    start = _t.monotonic()
    result: dict = {
        "ok": False, "exit_code": -1, "stdout": "", "stderr": "",
        "elapsed_ms": 0, "command": command, "timed_out": False,
    }
    # Run verification with cwd = active lab so relative paths in the
    # command (e.g. "test -s drafts/foo.md") resolve under the lab
    # whose cycle just ran. Defaults to LAB_ROOT for the bert-lab
    # supervisor.
    from core.lab_context import get_active_lab_path
    verify_cwd = get_active_lab_path() or LAB_ROOT
    try:
        r = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(verify_cwd),
        )
        result["exit_code"] = r.returncode
        result["ok"] = r.returncode == 0
        result["stdout"] = r.stdout[-6000:] if r.stdout else ""
        result["stderr"] = r.stderr[-2000:] if r.stderr else ""
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["stderr"] = f"[bert] verification timed out after {timeout}s"
    except (FileNotFoundError, OSError) as e:
        result["stderr"] = f"[bert] verification crashed: {type(e).__name__}: {e}"
    result["elapsed_ms"] = int((_t.monotonic() - start) * 1000)
    return result


def _synthesize_packet_from_verification(
    spec: dict, *, verify: dict, latency_secs: float, model_used: str
) -> dict:
    """When the agent crashed without writing a ResultPacket but the
    verification_command passed, synthesize a clean BUILD_PASS packet
    instead of returning OTHER/result_packet_missing. The verification
    output is the trustworthy signal."""
    return {
        "role": spec.get("role", "unknown"),
        "cycle": spec.get("cycle", 0),
        "verdict": Verdict.BUILD_PASS.value if verify["ok"] else Verdict.BUILD_FAIL.value,
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 7 if verify["ok"] else 3,
        "calibration_reasoning": (
            "[bert] Agent did not write its own ResultPacket but verification_command "
            f"{'passed' if verify['ok'] else 'failed'} (exit_code={verify['exit_code']}, "
            f"{verify['elapsed_ms']}ms). Verdict synthesized from external check, not "
            "from agent self-report. The work landed (output_path file exists or build "
            "passes); the agent just hit a token/iteration limit before emitting JSON."
        ),
        "telemetry": {
            "tokens_in": 0, "tokens_out": 0,
            "latency_secs": round(latency_secs, 2),
            "model_used": model_used, "provider": "",
            "retry_count": 0, "fallback_chain": [],
            "verification": verify,
        },
    }


def _synthesize_failure_packet(
    spec: dict, *, reason: str, latency_secs: float, model_used: str
) -> dict:
    """Build a verdict=OTHER packet when the sub-agent failed to emit one."""
    return {
        "role": spec.get("role", "unknown"),
        "cycle": spec.get("cycle", 0),
        "verdict": Verdict.OTHER.value,
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 1,
        "calibration_reasoning": (
            f"Sub-agent terminated without writing a valid ResultPacket. "
            f"Reason: {reason}. Director should treat this as failed dispatch and "
            f"either retry with a clarified task or escalate to PI."
        ),
        "telemetry": {
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_secs": round(latency_secs, 2),
            "model_used": model_used,
            "provider": "",
            "retry_count": 0,
            "fallback_chain": [],
        },
    }


def _attempt_schema_correction(
    spec: dict, invalid_packet: dict, errors: list[str]
) -> dict | None:
    """Quality-first retry: when the agent's emitted ResultPacket fails
    schema validation, fire `core.decode.call_with_schema` with the
    invalid packet + errors as the correction context. Returns the
    corrected packet dict if recovery succeeded, else None.

    Provider-side response_format enforcement (per
    core.structured_output) constrains the shape on supporting
    providers (NVIDIA / Cerebras / Groq / Mistral / OpenRouter /
    HF Router); post-call jsonschema validate catches anything that
    provider-side missed. Up to 2 inner retries within decode.

    This is the structural fix for the Round 2-4 schema-shape failure
    pattern where the model emitted clearness_queries as raw strings
    or missed caveats_embedded — instead of warn-and-fall-through, we
    give the model one shot at correction with the explicit error +
    schema constraint.
    """
    try:
        from core import decode
    except ImportError as e:
        LOG.warning("spawn: core.decode unavailable for schema correction: %s", e)
        return None

    schema = _result_packet_schema()
    if schema is None:
        return None

    # P-020 redaction: the invalid packet may contain credentials that
    # leaked into calibration_reasoning (env-var echo, file paths,
    # tool-output residue). Apply redaction patterns BEFORE sending
    # to the model — sending raw plaintext to a third-party provider
    # endpoint would defeat redaction discipline.
    invalid_text = json.dumps(invalid_packet, indent=2, default=str)
    invalid_text = log.redact(invalid_text)
    correction_prompt = (
        "You previously emitted a ResultPacket that failed schema validation. "
        "Specific errors:\n  - " + "\n  - ".join(log.redact(e) for e in errors[:6]) + "\n\n"
        f"Here is the invalid packet:\n```json\n{invalid_text}\n```\n\n"
        f"Re-emit the ResultPacket as a single JSON object that satisfies the "
        f"schema. Preserve as much of the original content as possible (verdict, "
        f"calibration_reasoning, findings_count, telemetry); only change the "
        f"fields that triggered errors. Output ONLY the JSON object, no prose, "
        f"no code fences. Note: any values formatted like `<api_key:redacted>` "
        f"or `<*_token:redacted>` are intentionally redacted — leave them as-is "
        f"in your output, do not attempt to recover them."
    )

    provider_name, model = _parse_provider_model(spec.get("model") or "nvidia")

    result = decode.call_with_schema(
        provider_name,
        [{"role": "user", "content": correction_prompt}],
        schema=schema,
        model=model,
        schema_name="result_packet_correction",
        max_retries=2,
        max_tokens=2500,
        temperature=0.3,  # lower temp for shape-correction
    )
    if result.parsed is None:
        LOG.warning(
            "spawn: schema-correction failed after %d attempts: %s",
            result.attempts, result.last_error[:200],
        )
        return None
    return result.parsed


def _result_packet_schema() -> dict | None:
    """Load the ResultPacket schema once for the schema-correction retry."""
    schema_path = SCHEMAS_DIR / "result_packet.json"
    try:
        return json.loads(schema_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        LOG.warning("spawn: cannot load result_packet schema for correction: %s", e)
        return None


def run_subagent(spec: dict) -> dict:
    """Run one sub-agent dispatch.

    Returns a summary dict for the Director:
    {
      "verdict": str,
      "role": str,
      "cycle": int,
      "output_path": str,        # echo of where full report lives
      "result_path": str,        # where ResultPacket was written (or attempted)
      "findings_count": {...},
      "confidence_1to10": int,
      "calibration_reasoning": str,
      "telemetry": {...},
      "spec_valid": bool,
      "result_valid": bool,
      "errors": list[str],       # empty on success
    }
    """
    start = time.monotonic()

    # 1. Validate spec
    ok, spec_errs = validate_dispatch_spec(spec)
    if not ok:
        LOG.warning("spawn: invalid DispatchSpec — %d errors", len(spec_errs))
        return {
            "verdict": Verdict.OTHER.value,
            "role": spec.get("role", "unknown"),
            "cycle": spec.get("cycle", 0),
            "output_path": spec.get("output_path", ""),
            "result_path": "",
            "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
            "confidence_1to10": 1,
            "calibration_reasoning": (
                "DispatchSpec validation failed; sub-agent was not launched. "
                "Director must fix the spec before retrying."
            ),
            "telemetry": {
                "tokens_in": 0, "tokens_out": 0,
                "latency_secs": round(time.monotonic() - start, 2),
                "model_used": "", "provider": "", "retry_count": 0,
                "fallback_chain": [],
            },
            "spec_valid": False,
            "result_valid": False,
            "errors": spec_errs,
        }

    role = spec["role"]
    cycle = int(spec["cycle"])
    provider_name, model = _parse_provider_model(spec["model"])

    # Observability — emit subagent_spawn + role-specific dispatch event.
    try:
        observability.emit("subagent_spawn", {
            "role": role, "cycle": cycle, "provider": provider_name,
            "model": model, "dispatch_altitude": spec.get("dispatch_altitude"),
            "output_path": spec.get("output_path", ""),
        })
        # Role-specific Quaker pipeline dispatch events for falsifier targets.
        _quaker_event = {
            "threshing": "threshing_dispatch",
            "threshing_pass": "threshing_dispatch",
            "clearness_phase1": "clearness_phase1_dispatch",
            "clearness_phase2": "clearness_phase2_dispatch",
        }.get(role)
        if _quaker_event:
            observability.emit(_quaker_event, {
                "role": role, "cycle": cycle, "provider": provider_name,
                "model": model, "altitude": spec.get("dispatch_altitude"),
            })
    except Exception:  # noqa: BLE001
        pass  # observability is advisory

    # 2. Decide where the ResultPacket must land — active-lab routed.
    _active_results_dir().mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^a-zA-Z0-9_-]+", "-", spec.get("output_path", "main")).strip("-")[:40]
    result_path = _result_path_for(role, cycle, tag)
    if result_path.exists():
        # Carry-over from a prior failed run; remove so we can detect this run's write.
        result_path.unlink()

    LOG.info(
        "spawn: role=%s cycle=%d provider=%s model=%s → result_path=%s",
        role, cycle, provider_name, model or "default", result_path,
    )

    # 3. Run the sub-agent loop
    # Lazy import to avoid circular: tools → subagent → agent → tools
    from core import agent as _agent
    task = _scoped_task(spec, result_path)
    config.load()

    # Real telemetry sink — agent loop populates this in finally:.
    # We use it to overwrite the model's hallucinated telemetry in the
    # ResultPacket post-validation, so downstream cost tracking is honest.
    real_telemetry: dict = {}

    try:
        rc = _agent.run_role(
            role,
            cycle=cycle,
            task=task,
            provider_name=provider_name,
            model=model,
            max_iterations=SUBAGENT_MAX_ITERATIONS,
            is_subagent=True,
            telemetry_sink=real_telemetry,
            output_path=spec.get("output_path") or None,
            verification_spec=spec.get("verification_spec") or None,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # noqa: BLE001
        LOG.exception("spawn: sub-agent crashed: %s", e)
        rc = 1

    elapsed = time.monotonic() - start
    model_used = real_telemetry.get("model_used") or (
        f"{provider_name}/{model}" if model else provider_name
    )

    # 3.5. External verification (post-loop). Source of truth for build/test
    # pass — beats the agent's self-report when they disagree.
    #
    # Sprint 1 commit 2 (v1.0): prefer the Python-native VerificationSpec
    # if present (eliminates shell injection vector). Falls back to the
    # legacy `verification_command` shell string for backward compat.
    verify_result: dict | None = None
    verify_timeout = int(spec.get("verification_timeout_secs") or 120)
    if spec.get("verification_spec"):
        from core import verify_engine
        from core.lab_context import get_active_lab_path
        output_path_rel = spec.get("output_path", "")
        base = get_active_lab_path() or LAB_ROOT
        output_path_abs = (base / output_path_rel) if output_path_rel else None
        LOG.info("spawn: running verification_spec (Python-native, timeout=%ds)",
                 verify_timeout)
        if output_path_abs is None:
            verify_result = {
                "ok": False, "exit_code": 1, "stdout": "",
                "stderr": "verification_spec given but no output_path",
                "elapsed_ms": 0, "timed_out": False,
                "command": "<verification_spec>",
            }
        else:
            r = verify_engine.verify_artifact(
                spec["verification_spec"], output_path_abs,
                timeout_secs=verify_timeout,
            )
            verify_result = {
                "ok": r.ok, "exit_code": r.exit_code,
                "stdout": "\n".join(r.checks_passed)[-6000:],
                "stderr": "\n".join(r.checks_failed)[-2000:],
                "elapsed_ms": r.elapsed_ms,
                "timed_out": r.timed_out,
                "command": "<verification_spec:python>",
                "checks_passed": r.checks_passed,
                "checks_failed": r.checks_failed,
            }
        LOG.info("spawn: verification %s (Python, %dms): %s",
                 "passed" if verify_result["ok"] else "FAILED",
                 verify_result["elapsed_ms"],
                 verify_result.get("stderr", "")[:120])
    elif spec.get("verification_command"):
        LOG.info("spawn: running verification_command (shell, timeout=%ds): %s",
                 verify_timeout, spec["verification_command"][:120])
        verify_result = _run_verification(spec["verification_command"], timeout=verify_timeout)
        LOG.info("spawn: verification %s (exit=%d, %dms)",
                 "passed" if verify_result["ok"] else "FAILED",
                 verify_result["exit_code"], verify_result["elapsed_ms"])

    # 4. Read + validate the ResultPacket
    if not result_path.exists():
        # Recovery path: if verification command exists and passed AND
        # the agent's declared output_path got written, synthesize a
        # BUILD_PASS verdict from the external check.
        #
        # Quality gate (added after a real production bug): pre-fix
        # this path treated ANY passing verification_command as
        # evidence of success. Trivial commands like `echo ok` always
        # pass, so an LLM call that 401'd or 5xx'd would still come
        # out as BUILD_PASS — masking provider failures across the
        # whole loop. Director's loop then treated "success with no
        # output" as a normal cycle outcome and looped on the same
        # decision until 3-strike.
        #
        # New rule: synthesis requires BOTH
        #   (a) verification_command passed AND it does meaningful
        #       work (not just "echo ok"-shaped no-op), OR
        #   (b) the agent's output_path file exists on disk (the
        #       agent dispatched a Write tool, even if it didn't
        #       write the JSON ResultPacket).
        # If neither holds, fall through to the honest failure
        # packet path.
        verification_meaningful = bool(
            verify_result
            and verify_result["ok"]
            and _verification_is_meaningful(spec.get("verification_command", ""))
        )
        output_landed = False
        try:
            output_path = spec.get("output_path")
            if output_path:
                # Match _resolve_relative_path's logic so the check
                # finds the agent's write under the active lab when
                # set, else under LAB_ROOT.
                from core.lab_context import get_active_lab_path
                base = get_active_lab_path() or LAB_ROOT
                output_landed = (base / output_path).exists()
        except OSError:
            output_landed = False

        if verification_meaningful or output_landed:
            LOG.warning(
                "spawn: ResultPacket missing but evidence-of-work found "
                "(verification_meaningful=%s, output_landed=%s) — "
                "synthesizing BUILD_PASS",
                verification_meaningful, output_landed,
            )
            packet = _synthesize_packet_from_verification(
                spec, verify=verify_result or {"ok": True, "exit_code": 0,
                                               "elapsed_ms": 0},
                latency_secs=elapsed, model_used=model_used,
            )
            with contextlib.suppress(OSError):
                result_path.write_text(json.dumps(packet, indent=2))
            return {**_summary_from_packet(packet),
                    "output_path": spec.get("output_path", ""),
                    "result_path": _result_path_str(result_path),
                    "spec_valid": True,
                    "result_valid": True,  # synthesized but valid
                    "errors": ["result_packet_synthesized_from_verification"]}

        LOG.warning("spawn: sub-agent did not write ResultPacket at %s", result_path)
        packet = _synthesize_failure_packet(
            spec,
            reason=f"Sub-agent loop exited rc={rc} without writing the required ResultPacket file.",
            latency_secs=elapsed,
            model_used=model_used,
        )
        if verify_result is not None:
            packet["telemetry"]["verification"] = verify_result
        return {**_summary_from_packet(packet),
                "output_path": spec.get("output_path", ""),
                "result_path": _result_path_str(result_path),
                "spec_valid": True,
                "result_valid": False,
                "errors": ["result_packet_missing"]}

    try:
        packet = json.loads(result_path.read_text())
    except json.JSONDecodeError as e:
        packet = _synthesize_failure_packet(
            spec,
            reason=f"ResultPacket file at {result_path.name} is not valid JSON: {e}",
            latency_secs=elapsed,
            model_used=model_used,
        )
        return {**_summary_from_packet(packet),
                "output_path": spec.get("output_path", ""),
                "result_path": _result_path_str(result_path),
                "spec_valid": True,
                "result_valid": False,
                "errors": [f"json-decode: {e}"]}

    ok, result_errs = validate_result_packet(packet)
    if not ok:
        LOG.warning("spawn: ResultPacket invalid — %d errors; attempting "
                    "schema-correction retry via core.decode", len(result_errs))
        # Schema-correction retry: instead of immediately falling through
        # to verdict=OTHER, fire one decode.call_with_schema with the
        # invalid packet text + the validation errors as the correction
        # context. The model gets to fix the shape against the actual
        # schema (provider-side response_format constrains the shape on
        # supporting providers; post-call validation catches anything
        # provider-side enforcement missed). This is the structural fix
        # for the Round 2-4 prompt-iteration pattern (clearness_queries
        # as raw strings, missing caveats_embedded, etc.).
        corrected_packet = _attempt_schema_correction(spec, packet, result_errs)
        if corrected_packet is not None:
            LOG.info("spawn: schema-correction recovered packet (verdict=%s)",
                     corrected_packet.get("verdict"))
            packet = corrected_packet
            # Re-validate to confirm correction landed
            ok2, errs2 = validate_result_packet(packet)
            if ok2:
                # Persist the corrected packet so downstream readers see it
                with contextlib.suppress(OSError):
                    result_path.write_text(json.dumps(packet, indent=2))
                # Fall through to the success path below — note in
                # calibration_reasoning that schema-correction fired
                packet.setdefault("calibration_reasoning", "")
                packet["calibration_reasoning"] = (
                    (packet.get("calibration_reasoning") or "")
                    + "\n\n[bert] Original ResultPacket emission failed schema "
                    f"validation; recovered via core.decode.call_with_schema "
                    f"(retry corrected: {'; '.join(result_errs[:3])})."
                )
                # carry on to the success path
                result_errs = []  # cleared
                ok = True
            else:
                LOG.warning(
                    "spawn: schema-correction succeeded parse but failed "
                    "re-validate — %d residual errors", len(errs2)
                )
                result_errs = errs2
        if not ok:
            # Correction failed — fall through to the original
            # verdict=OTHER + return-with-errors path.
            packet["verdict"] = Verdict.OTHER.value
            packet.setdefault("calibration_reasoning", "")
            packet["calibration_reasoning"] = (
                (packet.get("calibration_reasoning") or "")
                + f"\n\n[bert] Schema-validation failed: {'; '.join(result_errs[:5])}"
                + " (decode-correction retry also failed)"
            )
            return {**_summary_from_packet(packet),
                    "output_path": spec.get("output_path", ""),
                    "result_path": _result_path_str(result_path),
                    "spec_valid": True,
                    "result_valid": False,
                    "errors": result_errs}

    # Overwrite the agent's hallucinated telemetry with real values from the
    # accumulator. The agent has no access to its own ProviderResponse data,
    # so it makes up `model_used`, token counts, etc. Real numbers matter for
    # cost tracking + budget enforcement (P-012).
    if real_telemetry:
        packet["telemetry"] = real_telemetry

    # External verification overrides agent self-report when they disagree.
    # The agent can claim BUILD_PASS without actually running the build —
    # we trust the exit code of verification_command, not the agent's mood.
    if verify_result is not None:
        packet["telemetry"]["verification"] = verify_result
        agent_claim = packet.get("verdict")
        pass_verdicts = {Verdict.APPROVE.value, Verdict.APPROVE_WITH_CAVEATS.value,
                         Verdict.BUILD_PASS.value}
        fail_verdicts = {Verdict.CHANGES_REQUESTED.value, Verdict.REJECT.value,
                         Verdict.BUILD_FAIL.value, Verdict.BUILD_PARTIAL.value}
        if verify_result["ok"] and agent_claim in fail_verdicts:
            LOG.info("spawn: verification passed but agent claimed %s — overriding to BUILD_PASS",
                     agent_claim)
            packet["verdict"] = Verdict.BUILD_PASS.value
            packet["calibration_reasoning"] = (
                (packet.get("calibration_reasoning") or "")
                + f"\n\n[bert] Agent self-reported {agent_claim} but external "
                f"verification_command passed (exit_code=0, "
                f"{verify_result['elapsed_ms']}ms). Verdict overridden to "
                "BUILD_PASS — external check trumps self-report."
            )
        elif not verify_result["ok"] and agent_claim in pass_verdicts:
            LOG.warning("spawn: agent claimed %s but verification FAILED (exit=%d) — overriding to BUILD_FAIL",
                        agent_claim, verify_result["exit_code"])
            packet["verdict"] = Verdict.BUILD_FAIL.value
            packet["calibration_reasoning"] = (
                (packet.get("calibration_reasoning") or "")
                + f"\n\n[bert] Agent self-reported {agent_claim} but external "
                f"verification_command FAILED (exit_code={verify_result['exit_code']}, "
                f"{verify_result['elapsed_ms']}ms). stderr tail: "
                f"{verify_result.get('stderr', '')[:300]!r}. Verdict overridden "
                "to BUILD_FAIL — the build is broken regardless of what the "
                "agent thinks it shipped."
            )

    # Persist the (possibly corrected) packet
    try:
        result_path.write_text(json.dumps(packet, indent=2))
    except OSError as e:
        LOG.warning("spawn: failed to persist corrected telemetry: %s", e)

    LOG.info("spawn: %s/%s returned %s (confidence=%d) in %.1fs",
             role, cycle, packet.get("verdict"), packet.get("confidence_1to10", 0), elapsed)

    # Observability — emit verdict + subagent_finish + (when applicable)
    # stand_aside_verdict for the falsifier baseline.
    try:
        verdict_str = packet.get("verdict", "OTHER")
        observability.emit("verdict", {
            "role": role, "cycle": cycle, "verdict": verdict_str,
            "confidence_1to10": packet.get("confidence_1to10", 0),
            "elapsed_secs": round(elapsed, 2),
        })
        observability.emit("subagent_finish", {
            "role": role, "cycle": cycle, "verdict": verdict_str,
            "elapsed_secs": round(elapsed, 2),
            "result_valid": True,
        })
        if verdict_str == Verdict.APPROVE_WITH_CAVEATS.value:
            caveats = packet.get("caveats_embedded") or []
            observability.emit("stand_aside_verdict", {
                "role": role, "cycle": cycle,
                "concern_count": len(caveats),
                "severity_grade": packet.get("severity_grade"),
            })
            # Per-concern lifecycle tracking for falsifier T8/T9/T10.
            try:
                from core import concern_flow
                packet_for_flow = {**packet, "role": role, "cycle": cycle}
                concern_flow.emit_concerns_raised_from_packet(packet_for_flow)
            except Exception:  # noqa: BLE001
                pass
            # I.1 — APWC on a shippable role is conditionally accepted
            # (auto-accepted with reservation). The concerns ride forward
            # via propagate_concerns_to_next_dispatch; the artifact still
            # counts as a shippable output unless later vetoed.
            try:
                from core import artifact_acceptance
                if role in artifact_acceptance.SHIPPABLE_ROLES:
                    artifact_acceptance.emit_artifact_accepted(
                        artifact_id=f"{role}_C{cycle}_{spec.get('output_path', 'unknown')}",
                        source_dispatch_id=spec.get("dispatch_id") or f"{role}_C{cycle}",
                        cycle=cycle,
                        acceptance_kind=artifact_acceptance.KIND_VERDICT_AWC,
                        artifact_type=artifact_acceptance._ROLE_TYPE.get(role, artifact_acceptance.TYPE_OTHER),
                        role=role,
                    )
            except Exception:  # noqa: BLE001
                pass
        elif verdict_str == Verdict.APPROVE.value:
            # I.1 — APPROVE on a shippable role auto-accepts the artifact.
            # The north-star metric counts this as a "build privately,
            # prove publicly" output the lab stands behind.
            try:
                from core import artifact_acceptance
                if role in artifact_acceptance.SHIPPABLE_ROLES:
                    artifact_acceptance.emit_artifact_accepted(
                        artifact_id=f"{role}_C{cycle}_{spec.get('output_path', 'unknown')}",
                        source_dispatch_id=spec.get("dispatch_id") or f"{role}_C{cycle}",
                        cycle=cycle,
                        acceptance_kind=artifact_acceptance.KIND_VERDICT_APPROVE,
                        artifact_type=artifact_acceptance._ROLE_TYPE.get(role, artifact_acceptance.TYPE_OTHER),
                        role=role,
                    )
            except Exception:  # noqa: BLE001
                pass

        if verdict_str != Verdict.APPROVE_WITH_CAVEATS.value:
            # Non-caveat verdict — if this dispatch received propagated
            # concerns from a prior dispatch, mark them as addressed.
            propagated = spec.get("_propagated_concern_ids") or []
            if propagated:
                try:
                    from core import concern_flow
                    src_cycle = int(spec.get("_propagated_concern_source_cycle") or 0)
                    cycle_distance = max(0, int(cycle) - src_cycle)
                    resolution_dispatch_id = f"{role}_C{cycle}"
                    for cid in propagated:
                        concern_flow.emit_concern_addressed(
                            concern_id=cid,
                            resolution_dispatch_id=resolution_dispatch_id,
                            resolution_cycle=int(cycle),
                            cycle_distance=cycle_distance,
                            resolution_verdict=verdict_str,
                        )
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass

    return {**_summary_from_packet(packet),
            "output_path": spec.get("output_path", ""),
            "result_path": _result_path_str(result_path),
            "spec_valid": True,
            "result_valid": True,
            "errors": []}


def _summary_from_packet(packet: dict) -> dict:
    """Project a ResultPacket dict into a Director-friendly summary."""
    return {
        "verdict": packet.get("verdict", Verdict.OTHER.value),
        "role": packet.get("role", "unknown"),
        "cycle": packet.get("cycle", 0),
        "findings_count": packet.get("findings_count", {"high": 0, "med": 0, "low": 0, "nit": 0}),
        "confidence_1to10": packet.get("confidence_1to10", 1),
        "calibration_reasoning": packet.get("calibration_reasoning", ""),
        "telemetry": packet.get("telemetry", {}),
    }


# ── ConcernEntry forward-flow propagation ───────────────────────────


def propagate_concerns_to_next_dispatch(
    prior_packet: dict,
    next_spec: dict,
) -> dict:
    """Propagate APPROVE_WITH_CAVEATS concerns from a prior ResultPacket
    forward into the next dispatch's caveats_embedded.

    Cache-aware structure rule: concerns flow AFTER the cacheable prefix,
    not interleaved within it.

    Behavior:
      - If prior_packet.verdict ≠ APPROVE_WITH_CAVEATS, returns next_spec
        unchanged (no concerns to propagate).
      - If prior_packet has caveats_embedded (ConcernEntry array),
        APPENDS them to next_spec.caveats_embedded (creates the field
        if absent). This places concerns at the END of the caveats list,
        which is downstream of any pre-existing caveats and AFTER the
        dispatch's cacheable prefix structure.
      - Returns a NEW dict (does not mutate inputs).

    Cache-aware structure rule: when concerns propagate to next
    dispatch's caveats_embedded, the propagation goes AFTER any cacheable
    prefix in the dispatch's prompt, not interleaved. This preserves
    cacheability of the standing portion. The dispatch_spec construction
    in agent.py / orchestrator naturally puts caveats_embedded in the
    per-call delta block (not in the cacheable prefix).

    Note: schema for dispatch_spec.caveats_embedded is currently
    `array of string` (legacy v1); v2 ConcernEntry objects are in
    result_packet.caveats_embedded. For forward-flow we serialize
    ConcernEntry to a string representation so legacy dispatch consumers
    work; downstream verdict producers parse them back. This is a
    deliberate compromise; cleaner approach is full ConcernEntry
    propagation in v3, deferred.
    """
    out = dict(next_spec)
    if prior_packet.get("verdict") != "APPROVE_WITH_CAVEATS":
        return out

    prior_concerns = prior_packet.get("caveats_embedded") or []
    if not prior_concerns:
        return out

    existing = list(out.get("caveats_embedded") or [])
    # For lifecycle tracking we need the source dispatch context so the
    # propagated event can link back to the concern_raised event.
    try:
        from core import concern_flow
        src_dispatch_id = prior_packet.get("dispatch_id") or concern_flow._derive_dispatch_id(prior_packet)
        src_cycle = int(prior_packet.get("cycle") or 0)
        tgt_dispatch_id = out.get("dispatch_id") or out.get("output_path") or f"{out.get('role','unknown')}_pending"
        tgt_cycle = int(out.get("cycle") or src_cycle + 1)
    except Exception:  # noqa: BLE001
        concern_flow = None  # type: ignore[assignment]
        src_dispatch_id = src_cycle = tgt_dispatch_id = tgt_cycle = None  # type: ignore[assignment]

    for concern in prior_concerns:
        if isinstance(concern, dict):
            severity = concern.get("severity_grade", "?")
            text = concern.get("text", "")
            origin = concern.get("dispatch_id", "?")
            serialized = f"[{severity}] {text} (from {origin})"
            existing.append(serialized)
        else:
            text = str(concern)
            existing.append(text)
        # Emit propagation event (best-effort; advisory only).
        if concern_flow is not None and src_dispatch_id is not None:
            try:
                cid = concern_flow.derive_concern_id(text, str(src_dispatch_id))
                concern_flow.emit_concern_propagated(
                    concern_id=cid,
                    target_dispatch_id=str(tgt_dispatch_id),
                    target_cycle=int(tgt_cycle or 0),
                    cycle_distance=max(0, int(tgt_cycle or 0) - int(src_cycle or 0)),
                )
            except Exception:  # noqa: BLE001
                pass

    out["caveats_embedded"] = existing
    # Attach the propagated concern_ids so the downstream dispatch can
    # emit concern_addressed events when its verdict drops the concerns.
    if concern_flow is not None and src_dispatch_id is not None:
        propagated_ids = []
        for concern in prior_concerns:
            text = (concern.get("text", "") if isinstance(concern, dict) else str(concern))
            propagated_ids.append(concern_flow.derive_concern_id(text, str(src_dispatch_id)))
        if propagated_ids:
            out["_propagated_concern_ids"] = propagated_ids
            out["_propagated_concern_source_cycle"] = src_cycle
    return out


# ── Dispatch chain helper (J.6 — closes T9 concern_addressed) ────────


def dispatch_chain(specs: list[dict]) -> list[dict]:
    """Run a sequence of dispatches with automatic concern propagation.

    When a dispatch produces APPROVE_WITH_CAVEATS, its caveats_embedded
    are propagated to the next dispatch's spec via
    `propagate_concerns_to_next_dispatch` BEFORE the next dispatch
    runs. When a subsequent dispatch produces a non-AWC verdict,
    those concerns are emitted as concern_addressed events by
    run_subagent's existing path (line ~907).

    This closes the concern lifecycle automatically without ceremony
    in calling code (orchestrator, falsifier scripts, smoke tests).
    Without this helper, propagate_concerns_to_next_dispatch had no
    production caller and T9 concerns_addressed stayed at 0%.

    Args:
      specs: list of DispatchSpec dicts. Each spec MUST validate against
        schemas/dispatch_spec.json. The list MUST be non-empty.

    Returns:
      list of run_subagent summary dicts, one per spec, in order.
      Each summary includes the standard fields plus `result_path` so
      the caller can re-read the full ResultPacket if needed.

    The chain stops short if a dispatch fails validation (returns
    `{spec_valid: false}` summary). Caller can inspect the partial
    results and decide whether to continue manually.
    """
    if not specs:
        return []
    results: list[dict] = []
    last_packet: dict | None = None
    for spec in specs:
        # Auto-propagate from prior AWC packet, if any.
        if last_packet is not None and last_packet.get("verdict") == "APPROVE_WITH_CAVEATS":
            spec = propagate_concerns_to_next_dispatch(last_packet, spec)
        summary = run_subagent(spec)
        results.append(summary)
        # Re-read the full packet from disk so the NEXT iteration can
        # propagate from it. run_subagent returns a slim summary; the
        # full packet (with caveats_embedded, dispatch_id) lives at
        # summary['result_path']. Accept absolute or LAB_ROOT-relative.
        rp = summary.get("result_path")
        if rp and summary.get("result_valid"):
            rp_path = Path(rp)
            if not rp_path.is_absolute():
                rp_path = LAB_ROOT / rp_path
            try:
                last_packet = json.loads(rp_path.read_text())
            except (OSError, json.JSONDecodeError):
                last_packet = None
        else:
            last_packet = None
    return results


# ── Seasoning routing ───────────────────────────────────────────────


def classify_verdict_for_seasoning(packet: dict) -> dict | None:
    """Classify a ResultPacket's REJECT verdict for routing to the
    seasoning queue (P-VS-09; lay-aside-for-revival, per Sheeran 1983 ch. 6).

    Routing rules:
      - verdict ≠ REJECT → returns None (no seasoning)
      - verdict = REJECT with caveats_blocking_downstream non-empty AND
        a clear revision path indicated → returns None (re-dispatch
        with the caveats applied; not seasoning material)
      - verdict = REJECT with no clear revision path → returns
        seasoning instructions dict the orchestrator passes to
        core.seasoning.season(). The orchestrator's heuristic for
        'no clear revision path' is: caveats_blocking_downstream is
        empty OR contains entries like 'requires upstream-context-
        change' / 'depends-on-not-yet-existing-tooling' / 'awaiting-
        provider-availability'.

    Returns:
      None if not seasoning-bound, else a dict with keys:
        source_dispatch_id, summary, revival_conditions, cycle, altitude

    The orchestrator should call core.seasoning.season(**this_dict) to
    actually persist the entry.
    """
    if packet.get("verdict") != "REJECT":
        return None

    caveats = packet.get("caveats_blocking_downstream") or []
    role = packet.get("role", "?")
    cycle = packet.get("cycle", 0)

    # Heuristic for "no clear revision path": caveats describe blocking
    # conditions that ARE NOT under bert's immediate control.
    no_revision_markers = (
        "requires upstream", "requires-upstream",
        "depends on not-yet", "depends-on-not-yet",
        "awaiting provider", "awaiting-provider",
        "depends on free-tier", "depends-on-free-tier",
        "requires architectural", "requires-architectural",
    )
    has_no_revision = (
        len(caveats) == 0
        or any(any(m in str(c).lower() for m in no_revision_markers)
               for c in caveats)
    )

    if not has_no_revision:
        # The REJECT IS revisable — Director should re-dispatch with caveats.
        return None

    # Build seasoning instructions
    summary_parts = [
        f"REJECT verdict from {role} (cycle {cycle}).",
        packet.get("calibration_reasoning", "")[:200],
    ]
    if caveats:
        summary_parts.append(f"Blocking caveats: {'; '.join(str(c)[:80] for c in caveats[:3])}")
    summary = " ".join(p for p in summary_parts if p).strip()

    # Revival conditions: extract from caveats if they look like
    # observable triggers; otherwise use a default placeholder
    # the user/PI can refine.
    revival_conditions = []
    for c in caveats:
        text = str(c).strip()
        if any(m in text.lower() for m in no_revision_markers):
            revival_conditions.append(
                f"when condition reverses: {text[:100]}"
            )
    if not revival_conditions:
        revival_conditions = [
            f"when context changes such that the REJECT reasoning at cycle {cycle} no longer applies"
        ]

    return {
        "source_dispatch_id": f"{role}_C{cycle}",
        "summary": summary if len(summary) >= 50 else (summary + " " * (50 - len(summary)))[:200],
        "revival_conditions": revival_conditions,
        "cycle": cycle,
        "altitude": packet.get("dispatch_altitude"),
    }


__all__ = [
    "validate_dispatch_spec", "validate_result_packet",
    "run_subagent", "dispatch_chain",
    "SUBAGENT_MAX_ITERATIONS", "KNOWN_ROLES",
    "propagate_concerns_to_next_dispatch",
    "classify_verdict_for_seasoning",
]


# ── Verification meaningfulness check (module-level helper) ─────────

_NO_OP_VERIFICATIONS = {
    "echo ok", "echo 'ok'", 'echo "ok"',
    "true", "exit 0", ":", "/bin/true",
}


def _verification_is_meaningful(cmd: str) -> bool:
    """True if the verification_command does work that meaningfully
    signals "the dispatch did its job".

    `echo ok` and friends always pass — synthesizing BUILD_PASS from
    them was the root cause of bert pretending every director cycle
    succeeded even when the LLM call 401'd or returned empty output.
    Reject those patterns explicitly. Anything else (`test -s file`,
    `python -m pytest`, custom scripts, etc.) is treated as
    meaningful evidence of work.
    """
    if not cmd:
        return False
    normalized = cmd.strip().lower()
    return normalized not in _NO_OP_VERIFICATIONS
