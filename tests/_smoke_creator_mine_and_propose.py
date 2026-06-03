"""Smoke + TDD: core.creator.mine_and_propose — skill-mining trigger (Sprint 6 #28).

The mine->draft->propose->sandbox-validate->promote pipeline already existed in
creator.py; Sprint 6's organic-growth slice is the TRIGGER that runs it end to
end. mine_and_propose() mines recurring tool-call patterns, drafts the top N, and
queues each for PI approval — returning the proposal ids. No skill is activated
(promote() stays the PI-blessed step). All paths are tmp-isolated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import creator  # noqa: E402


def _events():
    # researcher does Read->Bash->Write three times -> a recurring 3-chain.
    out = []
    for c in (1, 2, 3):
        for tool in ("Read", "Bash", "Write"):
            out.append({"event_class": "tool_call", "agent": "researcher",
                        "cycle": c, "tool_name": tool})
    return out


def _write_events(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in _events()) + "\n")
    return ev


def test_mine_and_propose_drafts_and_queues(tmp_path):
    ev = _write_events(tmp_path)
    drafts_dir = tmp_path / "drafts"
    active_dir = tmp_path / "active"
    proposals = tmp_path / "proposals_pending_pi.md"
    out = creator.mine_and_propose(
        top_n=2, events_path=ev, drafts_dir=drafts_dir,
        proposals_path=proposals, min_frequency=2, min_length=2)
    assert 1 <= len(out) <= 2
    for entry in out:
        assert entry["proposal_id"].startswith("prop-")
        assert entry["skill_id"].startswith("skill-")
        assert entry["frequency"] >= 2
        assert (drafts_dir / entry["skill_id"] / "SKILL.md").exists()
    # queued for review, NOT activated
    body = proposals.read_text()
    assert "pending" in body
    assert not active_dir.exists() or list(active_dir.iterdir()) == []


def test_mine_and_propose_caps_top_n(tmp_path):
    ev = _write_events(tmp_path)
    out = creator.mine_and_propose(
        top_n=1, events_path=ev, drafts_dir=tmp_path / "d",
        proposals_path=tmp_path / "p.md", min_frequency=2, min_length=2)
    assert len(out) == 1


def test_mine_and_propose_empty_corpus(tmp_path):
    ev = tmp_path / "empty.jsonl"
    ev.write_text("")
    out = creator.mine_and_propose(
        top_n=3, events_path=ev, drafts_dir=tmp_path / "d",
        proposals_path=tmp_path / "p.md", min_frequency=2, min_length=2)
    assert out == []


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_mine_and_propose_drafts_and_queues,
        test_mine_and_propose_caps_top_n,
        test_mine_and_propose_empty_corpus,
    ]
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
