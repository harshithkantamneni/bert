"""Smoke test for core/creator.py — agent upskilling.

Tests the AutoSkill-shaped mining → drafting → propose-promotion →
promote pipeline. The mining heuristic is contiguous-subsequence
frequency on tool-call events.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import creator  # noqa: E402


def _seed_events(path: Path, dispatches: list[tuple[str, int, list[str]]]) -> None:
    """Write synthetic tool_call events. Each tuple is (agent, cycle, tools)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for agent, cycle, tools in dispatches:
        for tool in tools:
            lines.append(json.dumps({
                "event_class": "tool_call",
                "agent": agent,
                "cycle": cycle,
                "tool_name": tool,
                "content": f"{tool}(...)",
            }))
    path.write_text("\n".join(lines) + "\n")


def test_mine_returns_empty_when_no_events() -> None:
    tmp = Path(tempfile.mkdtemp()) / "events.jsonl"
    # Don't create the file
    drafts = creator.mine_patterns(events_path=tmp)
    assert drafts == []


def test_mine_finds_repeated_pattern() -> None:
    tmp = Path(tempfile.mkdtemp()) / "events.jsonl"
    # Pattern Read → Write repeats 4 times across 4 dispatches
    _seed_events(tmp, [
        ("r1", 1, ["Read", "Write"]),
        ("r1", 2, ["Read", "Write"]),
        ("r2", 3, ["Read", "Write", "Bash"]),
        ("r3", 4, ["Read", "Write"]),
    ])
    drafts = creator.mine_patterns(events_path=tmp, min_frequency=3)
    assert drafts
    # Top draft should be the Read→Write pair (count = 4)
    top = drafts[0]
    assert top.tool_sequence == ["Read", "Write"]
    assert top.frequency == 4


def test_mine_respects_min_frequency() -> None:
    tmp = Path(tempfile.mkdtemp()) / "events.jsonl"
    _seed_events(tmp, [
        ("r1", 1, ["Read", "Write"]),
        ("r1", 2, ["Read", "Write"]),
    ])
    # Pattern repeats only 2× — below default MIN_FREQUENCY=3.
    drafts = creator.mine_patterns(events_path=tmp)
    assert drafts == []


def test_mine_respects_min_length() -> None:
    tmp = Path(tempfile.mkdtemp()) / "events.jsonl"
    _seed_events(tmp, [
        ("r1", 1, ["Read"]),
        ("r1", 2, ["Read"]),
        ("r1", 3, ["Read"]),
        ("r1", 4, ["Read"]),
    ])
    # Singletons don't get drafted (min_length=2).
    drafts = creator.mine_patterns(events_path=tmp)
    assert drafts == []


def test_draft_skill_writes_skill_md() -> None:
    tmp = Path(tempfile.mkdtemp()) / "drafts"
    skill = creator.SkillDraft(
        skill_id="skill-test123",
        name="test-chain",
        description="test pattern",
        tool_sequence=["Read", "Write"],
        frequency=5,
        example_dispatches=["r1_C1", "r1_C2"],
    )
    path = creator.draft_skill(skill, drafts_dir=tmp)
    assert path.exists()
    text = path.read_text()
    assert "name: test-chain" in text
    assert "frequency_observed: 5" in text
    assert "Read → Write" in text
    assert "P-005" in text  # permission gate reference


def test_propose_promotion_appends_to_file() -> None:
    drafts_dir = Path(tempfile.mkdtemp()) / "drafts"
    proposals = Path(tempfile.mkdtemp()) / "proposals.md"
    skill = creator.SkillDraft(
        skill_id="skill-prop", name="prop", description="", tool_sequence=["A"],
        frequency=3,
    )
    creator.draft_skill(skill, drafts_dir=drafts_dir)
    pid = creator.propose_promotion(
        "skill-prop", drafts_dir=drafts_dir, proposals_path=proposals,
    )
    assert pid.startswith("prop-skill-prop-")
    assert proposals.exists()
    text = proposals.read_text()
    assert pid in text
    assert "skill_id:** skill-prop" in text
    assert "/approve" in text


def test_propose_promotion_raises_on_missing_draft() -> None:
    drafts_dir = Path(tempfile.mkdtemp()) / "nope"
    proposals = Path(tempfile.mkdtemp()) / "p.md"
    try:
        creator.propose_promotion("missing", drafts_dir=drafts_dir, proposals_path=proposals)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_promote_moves_draft_to_active() -> None:
    drafts_dir = Path(tempfile.mkdtemp()) / "drafts"
    active_dir = Path(tempfile.mkdtemp()) / "active"
    skill = creator.SkillDraft(
        skill_id="skill-prom", name="prom", description="", tool_sequence=["X"],
        frequency=3,
    )
    creator.draft_skill(skill, drafts_dir=drafts_dir)
    # validate_in_sandbox=False to skip the sandbox roundtrip in tests
    # (sandbox.run takes ~50ms; we cover the sandbox path separately).
    dst = creator.promote("skill-prom", drafts_dir=drafts_dir,
                           active_dir=active_dir,
                           validate_in_sandbox=False)
    assert dst.exists()
    assert (dst / "SKILL.md").exists()
    # Draft directory should be gone
    assert not (drafts_dir / "skill-prom").exists()


def test_promote_with_sandbox_validation() -> None:
    """E.3 hook: promote runs sandbox.validate_skill on the draft."""
    drafts_dir = Path(tempfile.mkdtemp()) / "drafts"
    active_dir = Path(tempfile.mkdtemp()) / "active"
    skill = creator.SkillDraft(
        skill_id="skill-sb", name="sb", description="",
        tool_sequence=["Read"], frequency=3,
    )
    creator.draft_skill(skill, drafts_dir=drafts_dir)
    dst = creator.promote("skill-sb", drafts_dir=drafts_dir,
                           active_dir=active_dir,
                           validate_in_sandbox=True)
    assert dst.exists()


def test_list_drafts_and_active() -> None:
    drafts_dir = Path(tempfile.mkdtemp()) / "drafts"
    active_dir = Path(tempfile.mkdtemp()) / "active"
    for sid in ["skill-a", "skill-b"]:
        s = creator.SkillDraft(skill_id=sid, name=sid, description="", tool_sequence=["X"], frequency=3)
        creator.draft_skill(s, drafts_dir=drafts_dir)
    assert creator.list_drafts(drafts_dir) == ["skill-a", "skill-b"]
    assert creator.list_active(active_dir) == []


def main() -> int:
    tests = [
        test_mine_returns_empty_when_no_events,
        test_mine_finds_repeated_pattern,
        test_mine_respects_min_frequency,
        test_mine_respects_min_length,
        test_draft_skill_writes_skill_md,
        test_propose_promotion_appends_to_file,
        test_propose_promotion_raises_on_missing_draft,
        test_promote_moves_draft_to_active,
        test_promote_with_sandbox_validation,
        test_list_drafts_and_active,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
