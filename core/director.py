"""core/director.py — autonomous-loop director.

Per-iteration of bert_run.py --autonomous, the director reads recent
lab state and picks the NEXT cycle's shape + focus_area. The decision
shapes how the cycle's researcher dispatch is prompted.

This is the MVP director — bounded decision space, single-pass
dispatch via core.subagent. The long-term design in prompts/director.md
(7-step orchestrator) is more ambitious; we ship the MVP first and
keep the long-term prompt untouched as the destination.

References:
- prompts/director_decision.md   — the role prompt
- prompts/director.md             — long-term full-persona design (not used here)
- findings/architecture/09_autonomy.md  — honest scope of "autonomous"

CoALA framing: this module is the **decision-making** process. The
substrate (memory) and reasoning (per-dispatch chain-of-thought) are
already in place; bert was missing only the decision loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("bert.director")

# Director model — nvidia/llama-3.3-70b is the default because it has
# 128K context (cerebras llama3.1-8b's 8192-token limit blows the
# observation JSON budget after ~5 events). Override via
# BERT_DIRECTOR_MODEL or by passing director_model= explicitly.
DEFAULT_DIRECTOR_MODEL = os.environ.get(
    "BERT_DIRECTOR_MODEL", "nvidia/meta/llama-3.3-70b-instruct",
)
PROMPT_PATH = LAB_ROOT / "prompts" / "director_decision.md"


# ── Decision taxonomy ───────────────────────────────────────────────


class CycleShape(StrEnum):
    RESEARCH_DEEPER = "research-deeper"
    STRATEGY_REFINE = "strategy-refine"
    VERIFICATION_TIGHTEN = "verification-tighten"
    SYNTHESIS = "synthesis"
    IDLE = "idle"
    # Director's "we have an answer" signal. Picked when the seed
    # brief's question has been answered with at least one synthesis
    # cycle's defensible report standing behind it. Distinct from
    # IDLE (which is "nothing to do right now, come back later") —
    # MISSION_COMPLETE means "the lab's purpose is fulfilled; stop
    # spending cycles on this seed." The runner emits a top-level
    # mission_complete event so the UI can show a receipt.
    MISSION_COMPLETE = "mission-complete"


class FocusArea(StrEnum):
    ROUTING = "routing"
    MEMORY = "memory"
    DISCIPLINE = "discipline"
    UX = "ux"
    UNSPECIFIED = "unspecified"


VALID_SHAPES = {s.value for s in CycleShape}
VALID_AREAS = {a.value for a in FocusArea}


@dataclass
class Decision:
    """The director's per-iteration output."""
    cycle_shape: str
    focus_area: str
    rationale: str
    researcher_prompt_focus: str
    expected_runtime_secs: int
    termination_condition: str
    confidence_1to10: int
    # Provenance fields (set by decide_next_cycle, not by the model):
    director_model: str = ""
    iteration: int = 0
    ts: str = ""
    errors: list[str] = field(default_factory=list)

    def is_terminal(self) -> bool:
        """The director signaled stop — either IDLE (come back later)
        or MISSION_COMPLETE (we have an answer; stop entirely)."""
        return self.cycle_shape in (
            CycleShape.IDLE.value, CycleShape.MISSION_COMPLETE.value,
        )

    def is_complete(self) -> bool:
        """The director declared the mission answered."""
        return self.cycle_shape == CycleShape.MISSION_COMPLETE.value

    def to_event(self) -> dict[str, Any]:
        """Render as a director_decision event_class entry for events.jsonl."""
        return {
            "ts": self.ts or datetime.now(UTC).isoformat(),
            "event_class": "director_decision",
            "iteration": self.iteration,
            "cycle_shape": self.cycle_shape,
            "focus_area": self.focus_area,
            "rationale": self.rationale,
            "researcher_prompt_focus": self.researcher_prompt_focus,
            "expected_runtime_secs": self.expected_runtime_secs,
            "termination_condition": self.termination_condition,
            "confidence_1to10": self.confidence_1to10,
            "director_model": self.director_model,
            "errors": self.errors,
        }


@dataclass
class Observation:
    """What the director sees before deciding."""
    seed_brief: str
    recent_events: list[dict]   # last 30 verdict/grade events
    falsifier_baseline: dict    # {pass, fail, insufficient, total}
    pending_count: int          # /api/pending equivalent
    last_weekly_grade: dict | None
    last_decisions: list[dict]  # last 3 director_decision events
    iteration: int
    # EE.3 — CoALA episodic feedback. Recent (decision, outcome)
    # tuples + aggregate calibration stats. Empty on first iteration
    # of a fresh lab; populated as outcomes accumulate.
    recent_outcomes: list[dict] = field(default_factory=list)
    calibration_stats: dict = field(default_factory=dict)
    # FF-A.2 — per-lab configuration. `lab_config` is the full
    # LabConfig.to_dict() snapshot; `focus_areas` is the bounded set
    # the director must pick from for THIS lab.
    lab_config: dict = field(default_factory=dict)
    focus_areas: list[str] = field(default_factory=list)
    # FF-B.2 — cross-lab signal (populated ONLY for role:supervisor
    # labs). Empty dict for standard labs — they don't see other labs.
    cross_lab_signal: dict = field(default_factory=dict)
    # GG-B — PI talk-to-lab messages since the last director cycle.
    # The director MUST address each one in its rationale (locked
    # by the new "PI messages this iteration" prompt section).
    pi_messages: list[dict] = field(default_factory=list)
    # A1 — Saturation signal from core/cycle_budget. When recent
    # cycles have produced no new findings/memory, the director should
    # consider emitting cycle_shape=mission-complete. The signal is
    # ADVISORY; per P-8 quality-first the director makes the final
    # call (a known-incomplete mission may still saturate temporarily).
    saturation_hint: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "seed_brief_preview": self.seed_brief[:1200],
            "iteration": self.iteration,
            "recent_events_count": len(self.recent_events),
            "recent_events_sample": self.recent_events[:8],
            "falsifier_baseline": self.falsifier_baseline,
            "pending_count": self.pending_count,
            "last_weekly_grade": self.last_weekly_grade,
            "last_decisions": self.last_decisions,
            "recent_outcomes_count": len(self.recent_outcomes),
            "recent_outcomes_sample": self.recent_outcomes[-8:],
            "calibration_stats": self.calibration_stats,
            "lab_config": self.lab_config,
            "focus_areas": self.focus_areas,
            "cross_lab_signal": self.cross_lab_signal,
            "pi_messages_count": len(self.pi_messages),
            "pi_messages": self.pi_messages,
            "saturation_hint": self.saturation_hint,
        }, indent=2)


# ── Observation gathering ───────────────────────────────────────────


def _read_seed_brief(lab_path: Path) -> str:
    f = lab_path / "seed_brief.md"
    return f.read_text() if f.exists() else ""


def _read_recent_events(lab_path: Path, *, n: int = 30,
                        keep_classes: tuple[str, ...] = (
                            "dispatch_result", "verdict",
                            "artifact_accepted", "falsifier_fire",
                            "director_decision", "director_terminated",
                            "pi_message",  # GG-B — talk-to-lab channel
                        )) -> list[dict]:
    f = lab_path / "sor" / "events.jsonl"
    if not f.exists():
        return []
    out: list[dict] = []
    # Read tail efficiently
    with f.open("rb") as fh:
        try:
            size = f.stat().st_size
            fh.seek(max(0, size - 512 * 1024))
            tail = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return []
    for line in tail.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event_class") in keep_classes:
            out.append(ev)
    # Keep only the last n
    return out[-n:]


def _read_falsifier_baseline() -> dict:
    """Best-effort read of the most-recent falsifier baseline."""
    findings_dir = LAB_ROOT / "findings"
    if not findings_dir.exists():
        return {"pass": None, "fail": None, "insufficient": None, "total": 14,
                "note": "findings/ missing"}
    candidates = sorted(findings_dir.glob("falsifier_baseline_*.json"))
    if not candidates:
        return {"pass": None, "fail": None, "insufficient": None, "total": 14,
                "note": "no baseline files found"}
    latest = candidates[-1]
    try:
        d = json.loads(latest.read_text())
        # Some baselines report at top-level, others nest. Handle both.
        if "pass" in d:
            return {"pass": d.get("pass"), "fail": d.get("fail"),
                    "insufficient": d.get("insufficient"),
                    "total": d.get("total", 14),
                    "source": str(latest.relative_to(LAB_ROOT))}
        return {"pass": None, "note": "unknown schema",
                "source": str(latest.relative_to(LAB_ROOT))}
    except (OSError, json.JSONDecodeError) as exc:
        return {"pass": None, "note": f"unreadable: {exc}"}


def _read_pending_count(lab_path: Path) -> int:
    """Count of pending blessings in the lab's state dir."""
    state = lab_path / "state"
    if not state.exists():
        return 0
    pending = state / "dev_pending.jsonl"
    if not pending.exists():
        return 0
    try:
        return sum(1 for line in pending.read_text().splitlines() if line.strip())
    except OSError:
        return 0


def _read_last_weekly_grade() -> dict | None:
    findings = LAB_ROOT / "findings"
    if not findings.exists():
        return None
    candidates = sorted(findings.glob("weekly_quality_report_*.json"))
    if not candidates:
        return None
    try:
        d = json.loads(candidates[-1].read_text())
        return {
            "date": candidates[-1].stem.replace("weekly_quality_report_", ""),
            "grades": d.get("grades", {}),
            "source": str(candidates[-1].relative_to(LAB_ROOT)),
        }
    except (OSError, json.JSONDecodeError):
        return None


def _last_director_decisions(events: list[dict], *, n: int = 3) -> list[dict]:
    """Pull the last N director_decision events for the 3-strike check."""
    decisions = [e for e in events if e.get("event_class") == "director_decision"]
    return decisions[-n:]


def gather_observation(lab_path: Path, *, iteration: int) -> Observation:
    """Read the lab's state into a structured observation for the director.

    FF-A.2 — reads `lab.yaml` via `core.lab_config.load()` and surfaces
    per-lab `focus_areas` + `role` + `mission` to the Observation so
    the director's decision is bounded to the lab's declared areas
    (not bert-internal routing/memory/discipline/ux verbatim).

    FF-B.2 — when `role: supervisor`, also reads cross-lab telemetry
    via `core.lab_aggregator.gather_cross_lab_signal()`. Standard labs
    skip the aggregator entirely (zero cost, perfect isolation).
    """
    events = _read_recent_events(lab_path)
    # EE.3 — episodic feedback: previous (decision, outcome) tuples
    # + computed calibration stats. Lazy import to avoid a circular
    # dep at module load time (outcome → director would loop if both
    # top-imported each other).
    from core import lab_config as lc_mod
    from core import outcome as out_mod
    recent_outcomes = out_mod.read_recent_outcomes(lab_path, n=30)
    stats = out_mod.compute_calibration_stats(recent_outcomes)
    cfg = lc_mod.load(lab_path)

    # FF-B.2 — supervisor-only cross-lab read
    cross_lab_signal: dict = {}
    if cfg.is_supervisor:
        try:
            from core import lab_aggregator as agg_mod
            cross_lab_signal = agg_mod.gather_cross_lab_signal().to_obs_dict()
        except Exception as exc:  # noqa: BLE001
            # Aggregator must NEVER crash gather_observation. Log + carry on
            # with an empty signal so the supervisor still runs.
            LOG.warning("supervisor aggregator failed: %s", exc)
            cross_lab_signal = {"lab_count": 0, "labs": [], "rollups": {},
                                 "excluded_labs": [], "exclusion_reasons": {},
                                 "note": f"aggregator failed: {exc.__class__.__name__}"}

    # GG-B — extract the pi_message events from the recent stream so
    # the director's prompt can address them in a dedicated section.
    # We still keep them in recent_events for full context.
    pi_messages = [e for e in events if e.get("event_class") == "pi_message"]

    # A1 — Compute saturation hint from recent cycles' novelty scores.
    # Lazy import to avoid coupling director module load to cycle_budget.
    saturation_hint: dict = {}
    try:
        from core import cycle_budget
        sat, scores = cycle_budget.is_saturated(
            lab_path, current_cycle=iteration, window=3, threshold=0.3
        )
        if scores:  # only emit hint when we have enough history
            saturation_hint = {
                "saturated": sat,
                "recent_novelty_scores": scores,   # most-recent-first
                "window": 3,
                "threshold": 0.3,
                "advisory": (
                    "If saturated AND the seed mission is answered, emit "
                    "cycle_shape=mission-complete. Per P-8 quality-first, "
                    "saturation is ADVISORY — if the mission is genuinely "
                    "unfinished, continue and explain why in rationale."
                ),
            }
    except Exception as exc:  # noqa: BLE001 — never break observation
        LOG.warning("saturation_hint compute failed: %s", exc)
        saturation_hint = {}

    return Observation(
        seed_brief=_read_seed_brief(lab_path),
        recent_events=events,
        falsifier_baseline=_read_falsifier_baseline(),
        pending_count=_read_pending_count(lab_path),
        last_weekly_grade=_read_last_weekly_grade(),
        last_decisions=_last_director_decisions(events),
        iteration=iteration,
        recent_outcomes=recent_outcomes,
        calibration_stats=stats.to_obs_dict(),
        lab_config=cfg.to_dict(),
        focus_areas=list(cfg.focus_areas),
        cross_lab_signal=cross_lab_signal,
        pi_messages=pi_messages,
        saturation_hint=saturation_hint,
    )


# ── Decision parsing ───────────────────────────────────────────────


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_decision_text(raw: str, *,
                          valid_focus_areas: set[str] | None = None
                          ) -> tuple[Decision | None, list[str]]:
    """Parse the director's free-form output into a Decision dataclass.

    The prompt asks for a strict JSON block; this parser is forgiving —
    it strips code fences, finds the largest JSON block in the response,
    validates the shape, and returns (Decision, errors). On any failure
    returns (None, errors).

    FF-A.2 — `valid_focus_areas` is now per-lab. When None (legacy
    callers, smoke tests pre-FF), falls back to the global VALID_AREAS
    so backwards-compat smoke tests still pass.
    """
    errors: list[str] = []
    if not raw or not raw.strip():
        return None, ["empty model output"]

    # Strip code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Find the JSON object
    m = _JSON_BLOCK.search(cleaned)
    if not m:
        return None, [f"no JSON object found in: {raw[:200]!r}"]

    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return None, [f"JSON parse failed: {exc}"]

    # Validate required fields
    required = ("cycle_shape", "focus_area", "rationale",
                "researcher_prompt_focus", "expected_runtime_secs",
                "termination_condition", "confidence_1to10")
    for field_name in required:
        if field_name not in d:
            errors.append(f"missing required field {field_name!r}")

    # Validate enum values
    if d.get("cycle_shape") not in VALID_SHAPES:
        errors.append(
            f"invalid cycle_shape {d.get('cycle_shape')!r}; "
            f"must be one of {sorted(VALID_SHAPES)}"
        )
    areas_for_check = valid_focus_areas if valid_focus_areas else VALID_AREAS
    if d.get("focus_area") not in areas_for_check:
        errors.append(
            f"invalid focus_area {d.get('focus_area')!r}; "
            f"must be one of {sorted(areas_for_check)}"
        )

    # Validate numeric ranges
    runtime = d.get("expected_runtime_secs")
    if not isinstance(runtime, int) or runtime < 0 or runtime > 3600:
        errors.append(f"expected_runtime_secs must be 0..3600; got {runtime!r}")

    conf = d.get("confidence_1to10")
    if not isinstance(conf, int) or conf < 1 or conf > 10:
        errors.append(f"confidence_1to10 must be 1..10; got {conf!r}")

    # Validate rationale length
    rationale = d.get("rationale", "")
    if isinstance(rationale, str) and len(rationale) < 60:
        errors.append(
            f"rationale too short ({len(rationale)} chars; require ≥60); "
            f"the director must explain its reasoning concretely"
        )

    if errors:
        return None, errors

    return (
        Decision(
            cycle_shape=d["cycle_shape"],
            focus_area=d["focus_area"],
            rationale=d["rationale"],
            researcher_prompt_focus=d["researcher_prompt_focus"],
            expected_runtime_secs=int(d["expected_runtime_secs"]),
            termination_condition=d["termination_condition"],
            confidence_1to10=int(d["confidence_1to10"]),
            errors=[],
        ),
        [],
    )


# ── Termination guardrails ─────────────────────────────────────────


class TerminationReason(StrEnum):
    DIRECTOR_IDLE = "director_idle"               # director chose IDLE
    MISSION_COMPLETE = "mission_complete"         # director declared answered
    THREE_STRIKE = "three_strike_identical_decisions"
    FAILURE_CASCADE = "two_consecutive_invalid_dispatches"
    PENDING_BACKLOG = "pending_approvals_exceeded_threshold"
    MAX_ITERATIONS = "max_iterations_reached"
    SIGNAL_INTERRUPT = "signal_interrupt"          # Ctrl-C


def check_three_strike(decisions: list[Decision | dict]) -> bool:
    """True if the last 3 decisions are identical (cycle_shape + focus_area)."""
    if len(decisions) < 3:
        return False
    last_three = decisions[-3:]

    def key(d):
        if isinstance(d, Decision):
            return (d.cycle_shape, d.focus_area)
        return (d.get("cycle_shape"), d.get("focus_area"))

    keys = [key(d) for d in last_three]
    return all(k == keys[0] for k in keys)


def check_failure_cascade(recent_dispatch_events: list[dict], *,
                          window: int = 2) -> bool:
    """True if the last N dispatch_result events all had result_valid=False."""
    recent = [e for e in recent_dispatch_events
              if e.get("event_class") == "dispatch_result"]
    if len(recent) < window:
        return False
    return all(not e.get("result_valid", True) for e in recent[-window:])


def check_pending_threshold(pending_count: int, *, threshold: int = 3) -> bool:
    """True if pending approvals exceeds the threshold."""
    return pending_count >= threshold


# ── The actual decision call ───────────────────────────────────────


def _load_director_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"director prompt missing at {PROMPT_PATH}. "
            f"Reinstall or git-restore prompts/director_decision.md"
        )
    return PROMPT_PATH.read_text()


def decide_next_cycle(
    lab_path: Path,
    iteration: int,
    *,
    director_model: str = DEFAULT_DIRECTOR_MODEL,
    observation: Observation | None = None,
    dispatch_fn: Any = None,  # injectable for testing; default = run_subagent
) -> Decision:
    """Fire the director dispatch and parse the result.

    Returns a Decision. On any failure (parse error, model 5xx, etc.) returns
    a Decision with `cycle_shape=IDLE` and the error captured — the runner
    then safely terminates rather than running a malformed cycle.

    Tests inject `dispatch_fn` to mock the model call.
    """
    obs = observation or gather_observation(lab_path, iteration=iteration)
    prompt = _load_director_prompt()

    task = (
        prompt
        + "\n\n## Observation (read carefully, then emit the decision JSON):\n\n"
        + "```json\n" + obs.to_json() + "\n```\n"
    )

    if dispatch_fn is None:
        # Lazy import so tests can mock without importing the whole subagent stack
        from core import subagent
        dispatch_fn = subagent.run_subagent

    # DispatchSpec schema (schemas/dispatch_spec.json) constraints:
    #   - dispatch_altitude must be one of META/SPEC/IMPL/INFRA/NIT-cleanup
    #     → director-level reasoning is META (deciding what to do next is
    #       above any single feature spec/impl)
    #   - role must be in KNOWN_ROLES or prefixed with "custom-"
    #     → director isn't in the canonical role set so we use "custom-director"
    #   - process_hygiene + falsifier_text + success_criterion have
    #     minLength constraints; the previous spec passed empty strings
    #     and short fragments, causing the validator to reject the call
    #     before the model ever ran. The dispatch then fell through to
    #     IDLE every time, silently killing the autonomous loop.
    #   - confidence_required is a required boolean
    spec = {
        "dispatch_altitude": "META",
        "role": "custom-director",
        "cycle": iteration,
        "model": director_model,
        "task": task,
        "success_criterion": (
            "Sub-agent writes a strict-JSON Decision block to "
            f"drafts/director_decision_C{iteration}.md containing all 7 required "
            "fields (cycle_shape, focus_area, rationale, "
            "researcher_prompt_focus, expected_runtime_secs, "
            "termination_condition, confidence_1to10) with values inside "
            "the locked enums and rationale ≥80 chars."
        ),
        "output_path": f"drafts/director_decision_C{iteration}.md",
        "falsifier_text": (
            "Decision JSON parses cleanly via core.director.parse_decision_text "
            "AND cycle_shape ∈ {research-deeper, strategy-refine, "
            "verification-tighten, synthesis, idle, mission-complete} "
            "AND focus_area is in the observation's focus_areas list."
        ),
        # Verification_command must do meaningful work (not no-op like
        # echo ok) — see core.subagent._verification_is_meaningful.
        # Otherwise an LLM 401 / 5xx would still synthesize BUILD_PASS
        # from the always-passing echo, masking real failures.
        "verification_command": (
            f"test -s drafts/director_decision_C{iteration}.md"
        ),
        "process_hygiene": (
            "Director writes JSON only — no commentary outside the block. "
            "If signals are thin, lower confidence_1to10 rather than "
            "fabricating evidence in the rationale."
        ),
        "confidence_required": True,
        "forbidden_actions": [],
        "caveats_embedded": [],
    }

    t0 = time.monotonic()
    try:
        summary = dispatch_fn(spec)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("director dispatch raised: %s", exc)
        return _idle_decision(
            iteration=iteration,
            director_model=director_model,
            errors=[f"dispatch_exception: {type(exc).__name__}: {exc}"],
            rationale=("director dispatch raised an exception; defaulting "
                       "to IDLE to terminate the autonomous loop safely. "
                       "Operator review recommended."),
        )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    LOG.info("director dispatch elapsed=%dms model=%s",
             elapsed_ms, director_model)

    # The model's response lives in the subagent's output file. Read
    # it from the active lab's directory (with LAB_ROOT fallback for
    # the bert-lab supervisor default).
    from core.lab_context import get_active_lab_path
    _base = get_active_lab_path() or LAB_ROOT
    raw_path = _base / "drafts" / f"director_decision_C{iteration}.md"
    if raw_path.exists():
        raw = raw_path.read_text()
    else:
        # Synthesize from summary if the file wasn't written
        raw = (summary or {}).get("calibration_reasoning", "")

    # FF-A.2 — validate focus_area against THIS lab's declared set
    lab_focus_areas = set(obs.focus_areas) if obs.focus_areas else None
    decision, errors = parse_decision_text(raw, valid_focus_areas=lab_focus_areas)
    if decision is None:
        LOG.warning("director output unparseable: %s; raw=%r",
                    errors, raw[:200] if raw else "(empty)")
        return _idle_decision(
            iteration=iteration,
            director_model=director_model,
            errors=errors,
            rationale=(
                "director output failed to parse as the locked Decision "
                "schema; defaulting to IDLE so the operator can review "
                f"the raw output at drafts/director_decision_C{iteration}.md."

            ),
        )

    decision.director_model = director_model
    decision.iteration = iteration
    decision.ts = datetime.now(UTC).isoformat()
    return decision


def _idle_decision(*, iteration: int, director_model: str,
                   errors: list[str], rationale: str) -> Decision:
    """Construct a safe-terminate Decision when the director can't reason."""
    return Decision(
        cycle_shape=CycleShape.IDLE.value,
        focus_area=FocusArea.UNSPECIFIED.value,
        rationale=rationale,
        researcher_prompt_focus=(
            "(no cycle scheduled — director defaulted to IDLE due to "
            "parse failure or dispatch error)"
        ),
        expected_runtime_secs=0,
        termination_condition="(safe-terminate via director_idle)",
        confidence_1to10=1,
        director_model=director_model,
        iteration=iteration,
        ts=datetime.now(UTC).isoformat(),
        errors=errors,
    )


# ── Helpers used by bert_run.py ─────────────────────────────────────


def emit_pattern_observed_event(
    lab_path: Path, *,
    pattern_summary: str,
    evidence_labs: list[str],
    iteration: int,
    related_event_classes: list[str] | None = None,
) -> dict:
    """FF-B.2 — emit a `pattern_observed` event when the supervisor's
    researcher/strategist surfaces a cross-lab pattern. The
    `supervisor_pattern_evidence` falsifier (FF-B.3) asserts
    len(evidence_labs) >= 2.

    Returns the emitted event dict so the caller can verify or log it.
    """
    ev = {
        "event_class": "pattern_observed",
        "ts": datetime.now(UTC).isoformat(),
        "iteration": iteration,
        "pattern_summary": pattern_summary[:600],
        "evidence_labs": list(evidence_labs),
        "evidence_lab_count": len(set(evidence_labs)),
        "related_event_classes": list(related_event_classes or []),
    }
    events_path = lab_path / "sor" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        f.write(json.dumps(ev, separators=(",", ":")) + "\n")
    return ev


def emit_decision_event(lab_path: Path, decision: Decision) -> None:
    """Append the decision as a director_decision event to events.jsonl."""
    events_path = lab_path / "sor" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        f.write(json.dumps(decision.to_event()) + "\n")


def emit_termination_event(lab_path: Path, *, iteration: int,
                           reason: TerminationReason, detail: str = "") -> None:
    """Append a director_terminated event to events.jsonl."""
    events_path = lab_path / "sor" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(UTC).isoformat(),
            "event_class": "director_terminated",
            "iteration": iteration,
            "reason": reason.value,
            "detail": detail,
        }) + "\n")


def emit_mission_complete_event(lab_path: Path, *, iteration: int,
                                decision: Decision) -> None:
    """Append a top-level mission_complete event.

    Distinct from director_terminated so the UI can recognise this
    case specifically and show a receipt (rather than treating it
    like any other termination). The runner emits this BEFORE the
    matching director_terminated event so the receipt lands first
    in the stream."""
    events_path = lab_path / "sor" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(UTC).isoformat(),
            "event_class": "mission_complete",
            "iteration": iteration,
            "rationale": decision.rationale,
            "confidence_1to10": decision.confidence_1to10,
            "focus_area": decision.focus_area,
        }) + "\n")


def compose_researcher_prompt_from_decision(decision: Decision,
                                            seed_brief: str) -> str:
    """Build the cycle's researcher dispatch prompt from the decision +
    the lab's seed brief. This is what makes the autonomous loop
    actually direct each cycle: the researcher gets a focused prompt
    instead of always reading the full seed."""
    return (
        "AUTONOMOUS CYCLE — director-composed researcher task.\n\n"
        f"Cycle shape: {decision.cycle_shape}\n"
        f"Focus area: {decision.focus_area}\n"
        f"Director rationale: {decision.rationale}\n\n"
        "Your specific research focus this cycle:\n"
        f"  {decision.researcher_prompt_focus}\n\n"
        "Termination condition (for THIS cycle):\n"
        f"  {decision.termination_condition}\n\n"
        "--- Lab's locked seed brief (context, not the immediate task) ---\n"
        f"{seed_brief[:800]}\n"
    )
