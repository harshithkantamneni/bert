"""Feature auto-promotion from mission patterns (Sprint 6 #29 — organic growth).

AC: repeated mission patterns surface as feature suggestions. Missions are
classified once per lab by mission_profile; the classification site emits a
`mission_classified` observability event (via record_mission_classified). This
module mines that stream: missions sharing a profile SIGNATURE
(domain, primary_work, data_shape, output_kind) at least `min_frequency` times
surface as a feature SUGGESTION written to state/feature_promotion_candidates.md
for PI review.

SUGGEST only — never auto-activates a feature. Activation stays PI-gated, the
same contract as creator.propose_promotion for skills: the lab proposes, the
human decides. Already-suggested signatures are not re-proposed (dedupe against
the candidates file), so re-running is idempotent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core import log

LOG = log.get_logger("bert.feature_promoter")

LAB_ROOT = Path(__file__).resolve().parent.parent
MISSION_EVENTS = LAB_ROOT / "state" / "observability" / "mission_classified.jsonl"
CANDIDATES_PATH = LAB_ROOT / "state" / "feature_promotion_candidates.md"

# The dims that define a "mission pattern". Two missions with the same signature
# would be served by the same feature.
SIG_DIMS = ("domain", "primary_work", "data_shape", "output_kind")
DEFAULT_MIN_FREQUENCY = 3


@dataclass
class FeatureSuggestion:
    signature: str
    count: int
    dims: dict
    example_missions: list[str] = field(default_factory=list)


# ── signature (pure) ─────────────────────────────────────────────────


def mission_signature(profile: dict) -> str:
    """Stable signature from the pattern-defining dims. Missing dims -> empty."""
    return "|".join(str(profile.get(d, "")) for d in SIG_DIMS)


# ── mining (pure) ────────────────────────────────────────────────────


def mine_mission_patterns(events: list[dict], *,
                          min_frequency: int = DEFAULT_MIN_FREQUENCY,
                          already_suggested: set[str] | None = None,
                          ) -> list[FeatureSuggestion]:
    """Bucket events by signature; return suggestions for buckets with count >=
    min_frequency that are not already suggested. Sorted by count desc."""
    already = already_suggested or set()
    buckets: dict[str, list[dict]] = {}
    for ev in events:
        buckets.setdefault(mission_signature(ev), []).append(ev)
    out: list[FeatureSuggestion] = []
    for sig, evs in buckets.items():
        if len(evs) < min_frequency or sig in already:
            continue
        dims = {d: evs[0].get(d, "") for d in SIG_DIMS}
        examples = [str(e.get("mission") or e.get("seed_excerpt") or "")
                    for e in evs if (e.get("mission") or e.get("seed_excerpt"))][:3]
        out.append(FeatureSuggestion(signature=sig, count=len(evs),
                                     dims=dims, example_missions=examples))
    out.sort(key=lambda s: s.count, reverse=True)
    return out


# ── candidates-file dedupe ───────────────────────────────────────────


_SIG_LINE = re.compile(r"^\s*-\s*\*\*signature:\*\*\s*(.+?)\s*$", re.MULTILINE)


def already_suggested_signatures(candidates_path: Path = CANDIDATES_PATH) -> set[str]:
    """Read the signatures already written to the candidates file, so we never
    re-propose the same mission pattern."""
    if not candidates_path.exists():
        return set()
    return {m.group(1) for m in _SIG_LINE.finditer(candidates_path.read_text())}


# ── event reading ────────────────────────────────────────────────────


def read_mission_events(events_path: Path = MISSION_EVENTS) -> list[dict]:
    if not events_path.exists():
        return []
    rows: list[dict] = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ── propose (file write, PI-gated review) ────────────────────────────


def propose_feature(suggestion: FeatureSuggestion, *,
                    candidates_path: Path | None = None) -> str:
    """Append a PI-review entry for a suggested feature. Returns the candidate
    id. Does NOT create or activate a feature — a human reviews + builds it.

    `candidates_path` resolves at call time so the module default stays patchable."""
    if candidates_path is None:
        candidates_path = CANDIDATES_PATH
    digest = suggestion.signature.replace("|", "-").strip("-") or "unknown"
    candidate_id = f"feat-{digest}"[:64]
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    dims = suggestion.dims
    examples = "\n".join(f"  - {m}" for m in suggestion.example_missions) or "  - (none recorded)"
    entry = (
        f"\n## {candidate_id}\n"
        f"\n- **signature:** {suggestion.signature}\n"
        f"- **observed_count:** {suggestion.count}\n"
        f"- **domain:** {dims.get('domain')}\n"
        f"- **primary_work:** {dims.get('primary_work')}\n"
        f"- **data_shape:** {dims.get('data_shape')}\n"
        f"- **output_kind:** {dims.get('output_kind')}\n"
        f"- **proposed_at:** {datetime.now(UTC).isoformat()}\n"
        f"- **status:** pending\n"
        f"- **rationale:** {suggestion.count} missions shared this pattern; a "
        f"feature would template the roster + skill plan for it.\n"
        f"- **example_missions:**\n{examples}\n"
    )
    with candidates_path.open("a") as f:
        f.write(entry)
    LOG.info("feature_promoter: suggested %s (count=%d)", candidate_id, suggestion.count)
    return candidate_id


def run(*, events_path: Path | None = None,
        candidates_path: Path | None = None,
        min_frequency: int = DEFAULT_MIN_FREQUENCY) -> list[str]:
    """Mine the mission stream and propose any new repeated pattern. Idempotent:
    a pattern already in the candidates file is skipped. Returns new candidate ids."""
    if events_path is None:
        events_path = MISSION_EVENTS
    if candidates_path is None:
        candidates_path = CANDIDATES_PATH
    events = read_mission_events(events_path)
    already = already_suggested_signatures(candidates_path)
    suggestions = mine_mission_patterns(events, min_frequency=min_frequency,
                                        already_suggested=already)
    return [propose_feature(s, candidates_path=candidates_path) for s in suggestions]


# ── emit hook (called at the classification site) ────────────────────


def record_mission_classified(profile: dict, *, seed_excerpt: str = "") -> None:
    """Emit a mission_classified observability event capturing the profile
    signature dims. Best-effort: a failure here must NOT break classification."""
    from core import observability as obs
    payload = {d: profile.get(d, "") for d in SIG_DIMS}
    payload["rigor"] = profile.get("rigor", "")
    payload["seed_excerpt"] = (seed_excerpt or "")[:240]
    payload["mission"] = (seed_excerpt or "")[:240]
    try:
        obs.emit("mission_classified", payload)
    except Exception as e:  # noqa: BLE001
        LOG.debug("feature_promoter: mission_classified emit skipped (advisory): %s", e)
