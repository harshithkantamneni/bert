"""Agent upskilling: AutoSkill-style trace mining + skill drafting.

Replaces the 10-LoC "Implementation pending" stub.

Algorithm (training-free, AutoSkill-shaped):

  1. Trace mining: scan the previous N cycles of `lab/sor/events.jsonl`
     for recurring sub-agent scaffolding sequences — same tool calls
     in similar order, same prompt skeleton, similar result-packet
     shape.
  2. Skill drafting: reduce a high-frequency pattern to a callable
     skill — a markdown file in `skills/draft/<skill_id>/SKILL.md`
     with metadata, parameters, body, falsifier.
  3. Sandbox validation: candidate runs in `core/sandbox.py` against a
     held-out trace pair; falsifier must pass.
  4. Permission gate (P-005 / P-011): DEFAULT mode logs the proposal
     to `state/proposals_pending_pi.md`; PI ratifies via bert
     `/api/approve/{id}` or Telegram `/approve <id>`.
  5. Registration: ratified skills move to
     `skills/active/<skill_id>/SKILL.md` and register in
     `core/tool_registry.py` (or are loaded lazily by `core/skills.py`).
  6. Usage telemetry: skill calls log via `core/stream.emit('tool_call',
     ...)`. SkillOS-style curation prunes skills with zero invocations
     after M cycles.

This module is the *drafting + permission-gate + registration* slice.
The trace-mining heuristic is intentionally simple in this commit
(frequency-based clustering on tool-call sequences); SkillFoundry-style
heterogeneous mining is a Phase 2 operational upgrade.

CLI:

  python -m core.creator mine                 # mine recent events
  python -m core.creator draft <pattern_id>   # write skills/draft/...
  python -m core.creator promote <skill_id>   # move draft → active
                                                # (requires PI bless via
                                                # the bert canvas first)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

LOG = logging.getLogger("bert.creator")

LAB_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
SKILLS_DRAFT = LAB_ROOT / "skills" / "draft"
SKILLS_ACTIVE = LAB_ROOT / "skills" / "active"
PROPOSALS_PATH = LAB_ROOT / "state" / "proposals_pending_pi.md"

# Pattern must repeat ≥this many times before drafting (anti-noise floor).
MIN_FREQUENCY = 3
# Minimum tool-call sequence length to consider (singletons are
# almost always already abstracted).
MIN_SEQUENCE_LENGTH = 2


@dataclass
class SkillDraft:
    """One drafted skill — a clustered tool-call pattern."""
    skill_id: str
    name: str
    description: str
    tool_sequence: list[str]
    frequency: int
    example_dispatches: list[str] = field(default_factory=list)


# ── Trace mining ─────────────────────────────────────────────────────


def _read_events(path: Path = EVENTS_PATH, limit: int = 10_000) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _tool_sequences_by_dispatch(events: list[dict]) -> dict[str, list[str]]:
    """Group tool_call events by dispatch_id (or agent_cycle) and return
    the ordered tool-name sequence per dispatch."""
    by_dispatch: dict[str, list[str]] = {}
    for ev in events:
        if ev.get("event_class") != "tool_call":
            continue
        agent = ev.get("agent") or "unknown"
        cycle = ev.get("cycle")
        if cycle is None:
            continue
        key = f"{agent}_C{cycle}"
        tool_name = _extract_tool_name(ev)
        if tool_name:
            by_dispatch.setdefault(key, []).append(tool_name)
    return by_dispatch


def _extract_tool_name(ev: dict) -> str | None:
    """Pull the tool name out of a tool_call event."""
    # Try common fields; observability events vary slightly in shape.
    for key in ("tool_name", "tool", "name"):
        v = ev.get(key)
        if isinstance(v, str) and v:
            return v
    # Fall back to parsing the content blob.
    content = ev.get("content", "")
    m = re.match(r"^(\w+)\(", content) if isinstance(content, str) else None
    return m.group(1) if m else None


def mine_patterns(*, events_path: Path = EVENTS_PATH,
                  min_frequency: int = MIN_FREQUENCY,
                  min_length: int = MIN_SEQUENCE_LENGTH) -> list[SkillDraft]:
    """Find recurring tool-call subsequences across recent dispatches.

    Returns SkillDraft candidates sorted by frequency desc. The drafts
    are NOT written to disk yet — call draft_skill() to materialize.

    Algorithm: for each dispatch's tool sequence, enumerate all
    contiguous subsequences of length ≥ min_length; bucket by tuple
    key; report buckets with count ≥ min_frequency. This is the
    simplest AutoSkill-shaped mining; a Phase 2 upgrade swaps in
    SkillOS curation heuristics.
    """
    events = _read_events(events_path)
    sequences = _tool_sequences_by_dispatch(events)

    counter: Counter = Counter()
    examples: dict[tuple, list[str]] = {}
    for dispatch_id, seq in sequences.items():
        if len(seq) < min_length:
            continue
        # Enumerate contiguous subsequences of length 2..len(seq)
        for length in range(min_length, min(len(seq), 6) + 1):
            for start in range(len(seq) - length + 1):
                sub = tuple(seq[start:start + length])
                counter[sub] += 1
                examples.setdefault(sub, []).append(dispatch_id)

    drafts: list[SkillDraft] = []
    for sub, count in counter.most_common():
        if count < min_frequency:
            break
        skill_id = "skill-" + hashlib.sha256("|".join(sub).encode()).hexdigest()[:10]
        drafts.append(SkillDraft(
            skill_id=skill_id,
            name=f"{'-'.join(sub).lower()}-chain",
            description=f"Recurring {len(sub)}-step pattern: {' → '.join(sub)}",
            tool_sequence=list(sub),
            frequency=count,
            example_dispatches=examples[sub][:5],
        ))
    return drafts


# ── Skill drafting ───────────────────────────────────────────────────


def draft_skill(skill: SkillDraft, *, drafts_dir: Path = SKILLS_DRAFT) -> Path:
    """Materialize a SkillDraft into `skills/draft/<skill_id>/SKILL.md`.

    The drafted file follows the SKILL.md format (frontmatter + body)
    used by Anthropic Agent Skills + bert-lab.
    """
    drafts_dir.mkdir(parents=True, exist_ok=True)
    skill_dir = drafts_dir / skill.skill_id
    skill_dir.mkdir(exist_ok=True)
    skill_file = skill_dir / "SKILL.md"

    body = f"""---
name: {skill.name}
description: {skill.description}
status: draft
frequency_observed: {skill.frequency}
example_dispatches: {skill.example_dispatches[:3]}
ts: {_now_iso()}
---

# {skill.name}

This skill was *drafted by bert* (AutoSkill) from {skill.frequency}
observations of the following tool sequence in `lab/sor/events.jsonl`:

  {' → '.join(skill.tool_sequence)}

## When to use

When the dispatch goal matches the shape of the sequence above. The
mining heuristic is contiguous-subsequence frequency, so this
*tends* to capture short reusable scaffolding chains (e.g. "Read X,
Read Y, Write Z" or "list_tools, call_tool"), not deep reasoning.

## Falsifier (pre-registered, P-VS-03)

Before promotion to `skills/active/`, this skill must:
- Run inside `core/sandbox.py` against a held-out trace from one of
  `example_dispatches` and reproduce the original output shape.
- Receive an APPROVE verdict from the cross-family evaluator
  (`pick_evaluator_model` consults `lab/state/capability_matrix.jsonl`).

## Permission gate

This lives under P-005 (skill creation) and P-011 (destructive-op
hard-gate). Promotion is PI-blessed via the bert canvas:
`POST /api/approve/{skill.skill_id}` or Telegram `/approve {skill.skill_id}`.

## Usage telemetry

Once promoted, every invocation emits a tool_call event to
`lab/sor/events.jsonl` with `tool_name = '{skill.name}'`. SkillOS-
style curation prunes this skill if invocation count stays at zero for
M cycles (configurable; default M=30).
"""
    skill_file.write_text(body)
    LOG.info("creator: drafted %s at %s", skill.skill_id, skill_file)
    return skill_file


def list_drafts(drafts_dir: Path = SKILLS_DRAFT) -> list[str]:
    if not drafts_dir.exists():
        return []
    return sorted(p.name for p in drafts_dir.iterdir() if p.is_dir())


def list_active(active_dir: Path = SKILLS_ACTIVE) -> list[str]:
    if not active_dir.exists():
        return []
    return sorted(p.name for p in active_dir.iterdir() if p.is_dir())


# ── Organic-growth trigger (Sprint 6 #28) ────────────────────────────


def mine_and_propose(*, top_n: int = 3, events_path: Path = EVENTS_PATH,
                     drafts_dir: Path = SKILLS_DRAFT,
                     proposals_path: Path = PROPOSALS_PATH,
                     min_frequency: int = MIN_FREQUENCY,
                     min_length: int = MIN_SEQUENCE_LENGTH) -> list[dict]:
    """Mine recurring tool-call patterns, draft the top N, and queue each for PI
    approval. Returns one dict per queued skill: {skill_id, proposal_id,
    frequency, tool_sequence}. This is the organic-growth TRIGGER over the
    existing mine -> draft -> propose pipeline — no skill is activated here
    (promote() remains the PI-blessed step)."""
    drafts = mine_patterns(events_path=events_path, min_frequency=min_frequency,
                           min_length=min_length)
    results: list[dict] = []
    for d in drafts[:top_n]:
        draft_skill(d, drafts_dir=drafts_dir)
        pid = propose_promotion(d.skill_id, drafts_dir=drafts_dir,
                                proposals_path=proposals_path)
        results.append({"skill_id": d.skill_id, "proposal_id": pid,
                        "frequency": d.frequency, "tool_sequence": d.tool_sequence})
    LOG.info("creator: mine_and_propose queued %d skill(s) for PI review", len(results))
    return results


# ── Promotion (PI-blessed) ───────────────────────────────────────────


def propose_promotion(skill_id: str, *, drafts_dir: Path = SKILLS_DRAFT,
                       proposals_path: Path = PROPOSALS_PATH) -> str:
    """Append a PI-approval-pending entry. Returns the proposal id.

    The actual move from drafts → active happens only after a blessing
    arrives via the bert canvas (api/main.py /api/approve/{id})
    or Telegram bot.
    """
    skill_dir = drafts_dir / skill_id
    if not skill_dir.exists():
        raise FileNotFoundError(f"draft {skill_id!r} not in {drafts_dir}")
    proposal_id = f"prop-{skill_id}-{int(datetime.now(UTC).timestamp())}"
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        draft_path_display = str(skill_dir.relative_to(LAB_ROOT))
    except ValueError:
        # Tests run with tempdirs outside the lab root; fall back to absolute.
        draft_path_display = str(skill_dir)
    line = (
        f"\n## {proposal_id}\n"
        f"\n- **skill_id:** {skill_id}\n"
        f"- **draft_path:** {draft_path_display}\n"
        f"- **proposed_at:** {_now_iso()}\n"
        f"- **status:** pending\n"
        f"- **approve_command:** `/approve {proposal_id}` or "
        f"POST /api/approve/{proposal_id}\n"
    )
    with proposals_path.open("a") as f:
        f.write(line)
    return proposal_id


def promote(skill_id: str, *, drafts_dir: Path = SKILLS_DRAFT,
            active_dir: Path = SKILLS_ACTIVE,
            validate_in_sandbox: bool = True) -> Path:
    """Move a drafted skill into the active set. Caller is responsible
    for verifying PI blessing first (e.g. checking lab/state/blessings.jsonl).

    When validate_in_sandbox=True (default), runs the AutoSkill step 3
    sandbox validation via core.sandbox.validate_skill before moving
    the draft. Set False for tests that don't have a sandbox-compatible
    environment.
    """
    src = drafts_dir / skill_id
    if not src.exists():
        raise FileNotFoundError(f"draft {skill_id!r} not in {drafts_dir}")
    if validate_in_sandbox:
        try:
            from core import sandbox
            manifest = src / "SKILL.md"
            res = sandbox.validate_skill(manifest)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"sandbox validation failed (tier={res.tier_used.value}, "
                    f"exit={res.exit_code}): {res.stderr[:240]}"
                )
            LOG.info("creator: %s sandbox-validated (tier=%s, %dms)",
                     skill_id, res.tier_used.value, res.elapsed_ms)
        except ImportError:
            LOG.warning("creator: core.sandbox unavailable; promoting without validation")
    active_dir.mkdir(parents=True, exist_ok=True)
    dst = active_dir / skill_id
    if dst.exists():
        LOG.warning("creator: %s already in active; overwriting", skill_id)
        import shutil
        shutil.rmtree(dst)
    src.rename(dst)
    LOG.info("creator: promoted %s → %s", skill_id, dst)
    # G.4 — sign the promoted manifest + append to local Rekor.
    # Signing failures are advisory; the skill is still promoted.
    try:
        from core import signing
        manifest_now = dst / "SKILL.md"
        if manifest_now.exists():
            sig = signing.sign_skill_manifest(manifest_now)
            log_id = signing.append_to_local_rekor(sig)
            LOG.info("creator: %s signed (log_id=%d, hash=%s)",
                     skill_id, log_id, sig.artifact_hash[:12])
    except Exception as e:  # noqa: BLE001
        LOG.warning("creator: signing failed (advisory): %s", e)
    return dst


# ── activation (PI approved a skill proposal) ────────────────────────


def _skill_id_from_proposal(proposal_id: str) -> str | None:
    """proposal id format is `prop-<skill_id>-<timestamp>`; recover skill_id.
    Returns None for non-skill (e.g. tool-*) proposal ids."""
    if not proposal_id.startswith("prop-"):
        return None
    rest = proposal_id[len("prop-"):]
    skill_id = rest.rsplit("-", 1)[0]
    return skill_id or None


def activate(proposal_id: str, *, drafts_dir: Path = SKILLS_DRAFT,
             active_dir: Path = SKILLS_ACTIVE,
             validate_in_sandbox: bool = True) -> dict:
    """Promote an approved drafted skill (prop-* proposal) to active. Idempotent:
    if already active, returns {ok, already}. Caller confirms the PI blessing."""
    skill_id = _skill_id_from_proposal(proposal_id)
    if skill_id is None:
        return {"ok": False, "error": f"cannot parse skill_id from {proposal_id!r}"}
    # Containment: skill_id becomes a directory name — reject traversal so a
    # crafted proposal id can't promote/move files outside the skills dirs.
    if ".." in skill_id or "/" in skill_id or "\\" in skill_id or skill_id.startswith("."):
        return {"ok": False, "error": f"unsafe skill_id {skill_id!r}"}
    if (active_dir / skill_id).exists() and not (drafts_dir / skill_id).exists():
        return {"ok": True, "skill_id": skill_id, "already": True}
    if not (drafts_dir / skill_id).exists():
        return {"ok": False, "error": f"draft {skill_id!r} not found"}
    dst = promote(skill_id, drafts_dir=drafts_dir, active_dir=active_dir,
                  validate_in_sandbox=validate_in_sandbox)
    return {"ok": True, "skill_id": skill_id, "path": str(dst)}


# ── helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="agent upskilling")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mine", help="scan events.jsonl for recurring patterns")
    p_draft = sub.add_parser("draft", help="write a draft for the top N patterns")
    p_draft.add_argument("--top", type=int, default=3)
    p_propose = sub.add_parser("propose", help="propose a draft for PI approval")
    p_propose.add_argument("skill_id")
    p_auto = sub.add_parser("auto", help="mine + draft + propose the top N in one step")
    p_auto.add_argument("--top", type=int, default=3)
    p_promote = sub.add_parser("promote", help="move a draft → active (requires blessing)")
    p_promote.add_argument("skill_id")
    sub.add_parser("ls", help="list drafts and active skills")
    args = parser.parse_args(argv)

    if args.cmd == "mine":
        drafts = mine_patterns()
        if not drafts:
            print("no recurring patterns at frequency threshold")
            return 0
        for d in drafts[:10]:
            print(f"  {d.frequency:>4d}  {d.skill_id}  {' → '.join(d.tool_sequence)}")
        return 0
    if args.cmd == "draft":
        drafts = mine_patterns()
        for d in drafts[:args.top]:
            p = draft_skill(d)
            print(f"  wrote {p.relative_to(LAB_ROOT)}")
        return 0
    if args.cmd == "propose":
        pid = propose_promotion(args.skill_id)
        print(f"proposal {pid} written to {PROPOSALS_PATH.relative_to(LAB_ROOT)}")
        return 0
    if args.cmd == "auto":
        queued = mine_and_propose(top_n=args.top)
        if not queued:
            print("no recurring patterns at frequency threshold")
            return 0
        for q in queued:
            print(f"  queued {q['skill_id']}  ({q['frequency']}×)  -> {q['proposal_id']}")
        return 0
    if args.cmd == "promote":
        dst = promote(args.skill_id)
        print(f"promoted to {dst.relative_to(LAB_ROOT)}")
        return 0
    if args.cmd == "ls":
        print("drafts:", list_drafts())
        print("active:", list_active())
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
