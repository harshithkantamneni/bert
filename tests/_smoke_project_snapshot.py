"""Smoke + TDD: core/project_snapshot.py — project-state snapshot + delta (Q-6).

A cheap rolled-up view of a project's state (cycle, findings, artifacts_accepted)
so `bert project status` doesn't re-read the full events.jsonl every call. The
snapshot stores the byte offset it processed; delta() folds only the events
appended past that offset (append-only log), re-counts findings from the
filesystem (cheap), and rewrites the snapshot. No new events -> cached return.

All tmp-isolated; pure file I/O, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import project_snapshot as ps  # noqa: E402


def _lab(tmp_path, *, events, findings=0):
    (tmp_path / "lab" / "sor").mkdir(parents=True)
    ev = tmp_path / "lab" / "sor" / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in events) + ("\n" if events else ""))
    fdir = tmp_path / "findings"
    fdir.mkdir()
    for i in range(findings):
        (fdir / f"f{i}.md").write_text(f"# finding {i}")
    return tmp_path


def _ev(cycle, cls="tool_call"):
    return {"event_class": cls, "cycle": cycle}


def test_compute_state_rolls_up(tmp_path):
    lab = _lab(tmp_path, events=[
        _ev(1), _ev(2), _ev(3, "artifact_accepted"), _ev(3, "artifact_accepted")],
        findings=4)
    st = ps.compute_state(lab)
    assert st["cycle"] == 3
    assert st["findings"] == 4
    assert st["artifacts_accepted"] == 2
    assert st["events_offset"] > 0


def test_snapshot_writes_file(tmp_path):
    lab = _lab(tmp_path, events=[_ev(1)], findings=1)
    snaps = tmp_path / "snaps"
    st = ps.snapshot(lab, snapshots_dir=snaps)
    files = list(snaps.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["cycle"] == st["cycle"] == 1


def test_delta_no_change_returns_cached(tmp_path):
    lab = _lab(tmp_path, events=[_ev(1), _ev(2)], findings=1)
    snaps = tmp_path / "snaps"
    ps.snapshot(lab, snapshots_dir=snaps)
    d = ps.delta(lab, snapshots_dir=snaps)
    assert d["new_events"] == 0
    assert d["cycle"] == 2


def test_delta_folds_appended_events(tmp_path):
    lab = _lab(tmp_path, events=[_ev(1)], findings=1)
    snaps = tmp_path / "snaps"
    ps.snapshot(lab, snapshots_dir=snaps)
    # append more events + a finding
    ev = lab / "lab" / "sor" / "events.jsonl"
    with ev.open("a") as f:
        f.write(json.dumps(_ev(2)) + "\n")
        f.write(json.dumps(_ev(2, "artifact_accepted")) + "\n")
    (lab / "findings" / "new.md").write_text("# new")
    d = ps.delta(lab, snapshots_dir=snaps)
    assert d["new_events"] == 2
    assert d["cycle"] == 2
    assert d["artifacts_accepted"] == 1
    assert d["findings"] == 2  # re-counted from fs
    # offset advanced + persisted
    again = ps.delta(lab, snapshots_dir=snaps)
    assert again["new_events"] == 0


def test_delta_without_prior_snapshot_computes_full(tmp_path):
    lab = _lab(tmp_path, events=[_ev(1), _ev(2)], findings=1)
    snaps = tmp_path / "snaps"
    d = ps.delta(lab, snapshots_dir=snaps)  # no snapshot yet
    assert d["cycle"] == 2
    assert (snaps / f"{lab.name}.json").exists()  # snapshot created


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_compute_state_rolls_up,
        test_snapshot_writes_file,
        test_delta_no_change_returns_cached,
        test_delta_folds_appended_events,
        test_delta_without_prior_snapshot_computes_full,
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
