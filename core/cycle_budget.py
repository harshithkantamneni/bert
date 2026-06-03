"""Auto-derive cycle budgets from mission profile + detect saturation.

Phase A1 of the v3 plan. Provides:

  • estimate_budget(profile, archetype, mission_text) → CycleBudget
        — derives target cycle count from mission features
  • resolve_budget(arg, ...) → CycleBudget
        — parses MCP `lab_cycle(budget=...)` arg into CycleBudget
  • novelty_score(lab_path, cycle_id) → float ∈ [0, 1]
        — proxy for "did this cycle produce new knowledge"
  • is_saturated(lab_path, window=3, threshold=0.3) → (bool, scores)
        — last `window` cycles all below `threshold` → director should
          consider mission-complete

Budget enum values (used by `lab_cycle` MCP tool):

  "quick"           — 1-3 cycles                  (one-shot exploration)
  "standard"        — 5-10 cycles                 (typical research/analysis)
  "deep"            — 15-30 cycles                (multi-iteration build/research)
  "until_complete"  — run until director signals mission-complete (cap 50)
  "auto"            — pick preset from profile

Saturation rule:
  novelty per cycle is a proxy combining finding count + memory-write count
  weighted; normalized into [0, 1]. If `window` consecutive cycles fall
  below `threshold` (default 0.3), director emits `cycle_shape:
  mission-complete` automatically (subject to P-8 quality-first — if user
  set budget=until_complete, ask user before terminating instead).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# ── Budget presets ────────────────────────────────────────────────────


@dataclass(frozen=True)
class CycleBudget:
    """Cycle budget for one `lab_cycle` invocation.

    target              — soft target; cycles continue until reached OR
                          saturation OR safety_cap, whichever first
    safety_cap          — hard maximum; runner never exceeds
    can_terminate_early — whether saturation can stop the run before target
                          (False for "until_complete" mode — user-only stop)
    preset_name         — provenance for logs / lab_cost reporting
    """
    target: int
    safety_cap: int
    can_terminate_early: bool
    preset_name: str


PRESETS: dict[str, CycleBudget] = {
    "quick": CycleBudget(
        target=3, safety_cap=5, can_terminate_early=True, preset_name="quick"
    ),
    "standard": CycleBudget(
        target=8, safety_cap=15, can_terminate_early=True, preset_name="standard"
    ),
    "deep": CycleBudget(
        target=20, safety_cap=30, can_terminate_early=True, preset_name="deep"
    ),
    "until_complete": CycleBudget(
        target=50, safety_cap=50, can_terminate_early=True,
        preset_name="until_complete",
    ),
}


# ── Heuristic budget estimator ─────────────────────────────────────────


# Keywords that signal long-running / ongoing missions
_ONGOING_KEYWORDS = (
    "monitor", "watch", "track", "weekly", "daily", "ongoing",
    "continuously", "surveillance",
)
_QUICK_KEYWORDS = (
    "quick question", "one-shot", "single", "just check", "tldr",
)


def estimate_budget(
    profile: dict | None = None,
    *,
    archetype: str = "research",
    mission_text: str = "",
) -> CycleBudget:
    """Derive a CycleBudget from mission profile (preferred) or legacy
    archetype + mission text (fallback).

    Profile-aware rules (when profile is present):
      • horizon=ongoing                → "until_complete"
      • horizon=one_shot               → "quick"
      • primary_work=decide            → "standard" (decisions saturate)
      • primary_work=build             → "deep" (multi-iteration)
      • primary_work=monitor + horizon ≠ ongoing → "standard"
      • else (research/synthesize)     → "standard"

    Archetype-fallback rules:
      research → standard, product → deep, strategy → standard,
      demo_note_cli → quick. Mission text keyword overrides apply.
    """
    if profile:
        horizon = profile.get("horizon", "short")
        primary_work = profile.get("primary_work", "discover")
        if horizon == "ongoing":
            return PRESETS["until_complete"]
        if horizon == "one_shot":
            return PRESETS["quick"]
        if primary_work == "build":
            return PRESETS["deep"]
        if primary_work in ("decide", "monitor", "synthesize", "audit"):
            return PRESETS["standard"]
        return PRESETS["standard"]

    # Legacy archetype-based fallback
    text = (mission_text or "").lower()
    if any(k in text for k in _ONGOING_KEYWORDS):
        return PRESETS["until_complete"]
    if any(k in text for k in _QUICK_KEYWORDS):
        return PRESETS["quick"]
    # Short mission → quick
    if mission_text and len(mission_text.split()) <= 30:
        return PRESETS["quick"]

    archetype_defaults = {
        "research": "standard",
        "product": "deep",
        "strategy": "standard",
        "demo_note_cli": "quick",
    }
    preset_name = archetype_defaults.get(archetype, "standard")
    return PRESETS[preset_name]


def resolve_budget(
    arg: str | int | None,
    *,
    profile: dict | None = None,
    archetype: str = "research",
    mission_text: str = "",
) -> CycleBudget:
    """Resolve the MCP `lab_cycle(budget=...)` argument into a CycleBudget.

    Accepts:
      None or "auto"                     → estimate_budget(...)
      "quick" | "standard" | "deep" |
      "until_complete"                   → PRESETS[arg]
      int 1..50                          → CycleBudget(target=N, safety_cap=2N)
      str numeric like "5"               → parsed as int

    Raises ValueError for unknown presets or out-of-range integers.
    """
    if arg is None or arg == "auto":
        return estimate_budget(
            profile, archetype=archetype, mission_text=mission_text
        )
    if isinstance(arg, str):
        if arg in PRESETS:
            return PRESETS[arg]
        # Try parsing as int from string
        try:
            arg = int(arg)
        except ValueError as e:
            raise ValueError(
                f"unknown budget: {arg!r}. Use 'auto' | 'quick' | "
                f"'standard' | 'deep' | 'until_complete' | int(1..50)"
            ) from e
    if isinstance(arg, int):
        if arg < 1 or arg > 50:
            raise ValueError(f"cycle budget must be 1..50 (got {arg})")
        return CycleBudget(
            target=arg,
            safety_cap=min(50, max(arg * 2, 5)),
            can_terminate_early=True,
            preset_name=f"explicit_{arg}",
        )
    raise ValueError(f"unsupported budget type: {type(arg).__name__}")


# ── Saturation detection ──────────────────────────────────────────────


# Event class weights for novelty scoring.
# Higher weight = more "novel knowledge produced this cycle"
_NOVELTY_WEIGHTS = {
    "finding": 1.0,         # new research artifact
    "memory_write": 0.5,    # new semantic/episodic memory entry
    "artifact_accepted": 0.8,  # PI-signed artifact
}


def _scan_cycle_events(lab_path: Path, cycle_id: int) -> dict[str, int]:
    """Count events of each class for a given cycle from sor/events.jsonl.

    Returns a dict {event_class: count}. Empty dict if no events found
    or if events.jsonl is missing.
    """
    events_path = lab_path / "sor" / "events.jsonl"
    if not events_path.exists():
        return {}
    counts: dict[str, int] = {}
    try:
        with events_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("cycle") != cycle_id:
                    continue
                ec = ev.get("event_class") or ev.get("class") or "unknown"
                counts[ec] = counts.get(ec, 0) + 1
    except OSError:
        return {}
    return counts


def novelty_score(lab_path: Path, cycle_id: int) -> float:
    """Compute a novelty score for `cycle_id` ∈ [0, 1].

    Proxy combining weighted counts of finding + memory_write +
    artifact_accepted events. Normalized so a "typical" cycle that
    produces a researcher finding + strategist finding + memory writes
    lands around 0.7-0.9. A cycle with no findings and no memory writes
    scores near 0.

    Returns 0.0 if no events found for the cycle (treat as no novelty).
    """
    counts = _scan_cycle_events(lab_path, cycle_id)
    if not counts:
        return 0.0
    raw = sum(
        counts.get(ec, 0) * weight
        for ec, weight in _NOVELTY_WEIGHTS.items()
    )
    # Normalize: a baseline "good" cycle has ~3.0 weighted units
    # (2 findings + 2 memory_writes = 2*1.0 + 2*0.5 = 3.0)
    normalized = min(1.0, raw / 3.0)
    return round(normalized, 3)


def is_saturated(
    lab_path: Path,
    *,
    current_cycle: int,
    window: int = 3,
    threshold: float = 0.3,
) -> tuple[bool, list[float]]:
    """Check if the last `window` COMPLETED cycles all have novelty
    < `threshold`.

    `current_cycle` is the cycle ABOUT to run (or just ran). We inspect
    cycles [current_cycle - 1, current_cycle - 2, ..., current_cycle - window].
    The in-flight cycle is excluded because its events haven't fully
    landed yet, which would skew the score toward 0.

    Returns (saturated, scores) where scores is the list of novelty
    scores for the inspected window, most-recent-first.

    Saturation is the director's hint to consider mission-complete.
    Per P-8 quality-first, this is ADVISORY — the director makes the
    final call. If budget="until_complete", saturation surfaces via
    needs_user_input (Phase C) rather than auto-terminating.
    """
    if window < 1:
        raise ValueError(f"window must be ≥1, got {window}")
    if current_cycle <= window:
        # Not enough completed history yet (need current_cycle - 1 ≥ window)
        return (False, [])
    scores: list[float] = []
    # Look at COMPLETED cycles only: [current-1, current-2, ..., current-window]
    for c in range(current_cycle - 1, current_cycle - 1 - window, -1):
        scores.append(novelty_score(lab_path, c))
    saturated = all(s < threshold for s in scores)
    return (saturated, scores)


# ── CLI for smoke testing ─────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """Quick smoke-test from the CLI:

      python -m core.cycle_budget novelty <lab_path> <cycle_id>
      python -m core.cycle_budget saturated <lab_path> <current_cycle> [window=3]
      python -m core.cycle_budget estimate <archetype> "<mission_text>"
    """
    import sys
    if len(argv) < 2:
        print("usage: cycle_budget novelty|saturated|estimate ...", file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "novelty":
        if len(argv) < 4:
            print("usage: cycle_budget novelty <lab_path> <cycle_id>",
                  file=sys.stderr)
            return 2
        score = novelty_score(Path(argv[2]).expanduser(), int(argv[3]))
        print(f"cycle {argv[3]}: novelty={score}")
        return 0
    if cmd == "saturated":
        if len(argv) < 4:
            print("usage: cycle_budget saturated <lab_path> <current_cycle> "
                  "[window=3]", file=sys.stderr)
            return 2
        win = int(argv[4]) if len(argv) >= 5 else 3
        sat, scores = is_saturated(
            Path(argv[2]).expanduser(),
            current_cycle=int(argv[3]),
            window=win,
        )
        print(f"saturated={sat} window={win} scores={scores}")
        return 0
    if cmd == "estimate":
        archetype = argv[2] if len(argv) >= 3 else "research"
        mission_text = argv[3] if len(argv) >= 4 else ""
        budget = estimate_budget(None, archetype=archetype,
                                  mission_text=mission_text)
        print(f"preset={budget.preset_name} target={budget.target} "
              f"safety_cap={budget.safety_cap}")
        return 0
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv))
