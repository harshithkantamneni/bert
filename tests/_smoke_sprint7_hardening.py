"""Smoke + TDD: Sprint 7 hardening — fixes from the adversarial bug-hunt.

Confirmed findings addressed here:
- #1 project_snapshot.delta first-snapshot new_events counted raw lines, not
  parsed events (mismatch with the rest of delta()).
- #2 tool_synthesizer.activate did rec["source"] -> KeyError on a corrupt sidecar.
- #3 project_snapshot.delta int() on prior values could crash on a corrupt snapshot.
- #6 creator.activate / _skill_id_from_proposal allowed path traversal via the
  proposal id (prop-skill-../../etc-123 -> writes outside the skills dir).
- #8/#11 proposal_activate + tool_synthesizer.read_pending didn't reject a
  traversing proposal id.
- #7 finalize output_path was not validated -> ../../etc/passwd escapes the lab.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from core import (  # noqa: E402
    creator,
    proposal_activate,
)
from core import (
    project_snapshot as ps,
)
from core import (
    tool_synthesizer as ts,
)

# ── project_snapshot bug fixes ───────────────────────────────────────


def _lab(tmp_path, events_text):
    (tmp_path / "lab" / "sor").mkdir(parents=True)
    (tmp_path / "lab" / "sor" / "events.jsonl").write_text(events_text)
    (tmp_path / "findings").mkdir()
    return tmp_path


def test_first_snapshot_new_events_counts_parsed_only(tmp_path):
    # 2 valid JSON events + 1 garbage line -> new_events must be 2, not 3.
    lab = _lab(tmp_path, json.dumps({"cycle": 1}) + "\nGARBAGE\n" + json.dumps({"cycle": 2}) + "\n")
    d = ps.delta(lab, snapshots_dir=tmp_path / "snaps")
    assert d["new_events"] == 2


def test_delta_survives_corrupt_prior_snapshot(tmp_path):
    lab = _lab(tmp_path, json.dumps({"cycle": 1}) + "\n")
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    # corrupt snapshot: non-int fields
    (snaps / f"{lab.name}.json").write_text(json.dumps(
        {"lab": lab.name, "cycle": "??", "artifacts_accepted": None,
         "events_offset": "bad"}))
    # append an event so delta takes the incremental path
    with (lab / "lab" / "sor" / "events.jsonl").open("a") as f:
        f.write(json.dumps({"cycle": 2}) + "\n")
    d = ps.delta(lab, snapshots_dir=snaps)  # must not raise
    assert isinstance(d, dict) and "cycle" in d


# ── tool_synthesizer.activate hardening ──────────────────────────────


def test_activate_missing_source_returns_error(tmp_path):
    pend = tmp_path / "pending"
    pend.mkdir()
    (pend / "tool-x-abc.json").write_text(json.dumps({"name": "x"}))  # no source
    out = ts.activate("tool-x-abc", pending_dir=pend, lib_dir=tmp_path / "lib")
    assert out["ok"] is False and "source" in out["error"].lower()


def test_read_pending_rejects_traversal(tmp_path):
    pend = tmp_path / "pending"
    pend.mkdir()
    (tmp_path / "secret.json").write_text(json.dumps({"name": "pwned", "source": "x"}))
    # proposal id that would resolve to ../secret.json must NOT be read
    rec = ts.read_pending("../secret", pending_dir=pend)
    assert rec is None


# ── creator.activate traversal ───────────────────────────────────────


def test_creator_activate_rejects_traversal(tmp_path):
    out = creator.activate("prop-skill-../../../etc-1700000000",
                           drafts_dir=tmp_path / "d", active_dir=tmp_path / "a",
                           validate_in_sandbox=False)
    assert out["ok"] is False


# ── proposal_activate traversal ──────────────────────────────────────


def test_proposal_activate_rejects_traversal(tmp_path):
    out = proposal_activate.activate("tool-../../etc/x",
                                     log_path=tmp_path / "a.jsonl")
    assert out["ok"] is False


# ── finalize output_path traversal ───────────────────────────────────


def test_finalize_cli_rejects_traversal_output(monkeypatch):
    from tools import project_cli
    from tools.mcp import bert_lab
    monkeypatch.setattr(bert_lab, "_resolve_lab", lambda n: Path("/tmp/lab"))
    rc = project_cli.cmd_finalize(SimpleNamespace(
        lab="l", objective="Q", output="../../etc/passwd", json=False))
    assert rc == 1


def test_finalize_mcp_rejects_traversal_output(monkeypatch):
    from tools.mcp import bert_lab
    monkeypatch.setattr(bert_lab, "_resolve_lab", lambda n: Path("/tmp/lab"))
    out = bert_lab._t_lab_finalize({"objective": "Q", "output_path": "../../etc/x", "lab": "l"})
    assert out["ok"] is False


class _MP:
    def __init__(self):
        self._u = []

    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        self._u.clear()


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_first_snapshot_new_events_counts_parsed_only,
        test_delta_survives_corrupt_prior_snapshot,
        test_activate_missing_source_returns_error,
        test_read_pending_rejects_traversal,
        test_creator_activate_rejects_traversal,
        test_proposal_activate_rejects_traversal,
        test_finalize_cli_rejects_traversal_output,
        test_finalize_mcp_rejects_traversal_output,
    ]
    mp = _MP()
    for t in tests:
        params = inspect.signature(t).parameters
        try:
            if "monkeypatch" in params:
                t(mp)
            elif "tmp_path" in params:
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
        finally:
            mp.undo()
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
