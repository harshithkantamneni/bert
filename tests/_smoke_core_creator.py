"""Smoke: core/creator.py — AutoSkill trace mining + promotion (was 69%).

File/event-based. Covers _read_events (+ bad-json), _extract_tool_name
(all field variants + content-regex + none), _tool_sequences_by_dispatch,
mine_patterns (recurring subsequence → SkillDraft; empty), draft_skill +
list_drafts/list_active, propose_promotion (ok + missing), promote
(validate_in_sandbox=False: move / missing / active-exists overwrite),
and the mine/ls CLI (real-state, read-only).
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import creator  # noqa: E402


def _mining_events():
    out = []
    for c in (1, 2, 3):
        out.append({"event_class": "tool_call", "agent": "researcher", "cycle": c, "tool_name": "Read"})
        out.append({"event_class": "tool_call", "agent": "researcher", "cycle": c, "tool_name": "Bash"})
    return out


def test_event_helpers(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text(json.dumps(_mining_events()[0]) + "\nbad json\n")
    assert len(creator._read_events(ev)) == 1            # bad line skipped
    assert creator._read_events(tmp_path / "nope.jsonl") == []
    assert creator._extract_tool_name({"tool_name": "Read"}) == "Read"
    assert creator._extract_tool_name({"tool": "Bash"}) == "Bash"
    assert creator._extract_tool_name({"name": "Edit"}) == "Edit"
    assert creator._extract_tool_name({"content": "Grep(pattern=x)"}) == "Grep"
    assert creator._extract_tool_name({"content": "no call"}) is None
    seqs = creator._tool_sequences_by_dispatch(_mining_events())
    assert seqs["researcher_C1"] == ["Read", "Bash"]


def test_mine_patterns(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in _mining_events()) + "\n")
    drafts = creator.mine_patterns(events_path=ev, min_frequency=2, min_length=2)
    assert drafts and drafts[0].tool_sequence == ["Read", "Bash"]
    assert drafts[0].frequency >= 2
    # empty corpus → no drafts
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert creator.mine_patterns(events_path=empty, min_frequency=2, min_length=2) == []


def test_draft_and_list(tmp_path):
    draft = creator.SkillDraft(
        skill_id="skill-test01", name="read-bash-chain",
        description="Recurring 2-step pattern: Read → Bash",
        tool_sequence=["Read", "Bash"], frequency=3,
        example_dispatches=["researcher_C1"])
    drafts_dir = tmp_path / "drafts"
    p = creator.draft_skill(draft, drafts_dir=drafts_dir)
    assert p.exists() and p.name == "SKILL.md"
    assert "skill-test01" in creator.list_drafts(drafts_dir=drafts_dir)
    assert creator.list_drafts(drafts_dir=tmp_path / "nope") == []
    assert creator.list_active(active_dir=tmp_path / "nope") == []


def test_propose_and_promote(tmp_path):
    drafts_dir = tmp_path / "drafts"
    active_dir = tmp_path / "active"
    draft = creator.SkillDraft(
        skill_id="skill-prom01", name="x", description="d",
        tool_sequence=["Read"], frequency=2)
    creator.draft_skill(draft, drafts_dir=drafts_dir)
    # propose
    pid = creator.propose_promotion("skill-prom01", drafts_dir=drafts_dir,
                                    proposals_path=tmp_path / "proposals.md")
    assert pid.startswith("prop-")
    try:
        creator.propose_promotion("ghost", drafts_dir=drafts_dir,
                                  proposals_path=tmp_path / "proposals.md")
        raise SystemExit("no raise")
    except FileNotFoundError:
        pass
    # promote (no sandbox)
    dst = creator.promote("skill-prom01", drafts_dir=drafts_dir,
                          active_dir=active_dir, validate_in_sandbox=False)
    assert dst.exists() and "skill-prom01" in creator.list_active(active_dir=active_dir)
    # missing draft → FileNotFoundError
    try:
        creator.promote("ghost", drafts_dir=drafts_dir, active_dir=active_dir,
                        validate_in_sandbox=False)
        raise SystemExit("no raise")
    except FileNotFoundError:
        pass
    # active-exists overwrite branch
    creator.draft_skill(draft, drafts_dir=drafts_dir)
    dst2 = creator.promote("skill-prom01", drafts_dir=drafts_dir,
                           active_dir=active_dir, validate_in_sandbox=False)
    assert dst2.exists()


def test_cli_readonly():
    with contextlib.redirect_stdout(io.StringIO()):
        assert creator.main(["mine"]) == 0     # real events → patterns or none
        assert creator.main(["ls"]) == 0


def main() -> int:
    tests = [
        test_event_helpers,
        test_mine_patterns,
        test_draft_and_list,
        test_propose_and_promote,
        test_cli_readonly,
    ]
    for t in tests:
        td = Path(tempfile.mkdtemp())
        try:
            kwargs = {"tmp_path": td} if "tmp_path" in inspect.signature(t).parameters else {}
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        finally:
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
