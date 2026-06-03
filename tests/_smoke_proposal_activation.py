"""Smoke + TDD: PI approve -> activate wiring (Sprint 7 caveat).

Sprint 6 left lab_synthesize_tool / skill mining able to PROPOSE but with no
path to ACTIVATE an approved proposal. Two gaps fixed here:

1. tool_synthesizer.propose discarded the generated SOURCE (the markdown said
   "source recorded" but the bytes were lost) — so install() had nothing to
   recover. propose now also writes a machine-readable sidecar
   state/tools_pending/<proposal_id>.json carrying the full candidate.
2. New activation dispatcher core/proposal_activate.activate(proposal_id) routes
   tool-* -> tool_synthesizer.activate (install) and prop-* ->
   creator.activate (promote). Idempotent; unknown prefix -> ok=False.

All paths tmp-isolated; no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import creator, proposal_activate  # noqa: E402
from core import tool_synthesizer as ts

_SRC = "def activ8_tool(**kwargs):\n    return {'ok': True}\n"


def _candidate(name="activ8_tool"):
    src = _SRC.replace("activ8_tool", name)
    spec = ts.ToolSpec(name=name, description="returns ok",
                       params_schema={"type": "object", "properties": {}})
    return ts.SynthesisCandidate(spec=spec, source=src, smoke_test="assert True\n",
                                 scan=ts.static_safety_scan(src), sandbox=None,
                                 method="llm-v1")


# ── propose now persists the source (the bug fix) ────────────────────


def test_propose_writes_sidecar_with_source(tmp_path):
    cand = _candidate()
    pid = ts.propose(cand, proposals_path=tmp_path / "tools_pending_pi.md",
                     pending_dir=tmp_path / "pending")
    sidecar = tmp_path / "pending" / f"{pid}.json"
    assert sidecar.exists()
    rec = json.loads(sidecar.read_text())
    assert rec["source"] == cand.source        # source is NOT lost
    assert rec["name"] == "activ8_tool"
    assert "params_schema" in rec               # needed by install()


# ── tool activation (install from sidecar) ───────────────────────────


def test_activate_tool_installs_from_sidecar(tmp_path):
    from core import tool_registry
    cand = _candidate(name="activ8_install_me")
    pid = ts.propose(cand, proposals_path=tmp_path / "p.md",
                     pending_dir=tmp_path / "pending")
    assert tool_registry.get("activ8_install_me") is None  # not active pre-approve
    out = ts.activate(pid, pending_dir=tmp_path / "pending", lib_dir=tmp_path / "lib")
    assert out["ok"] is True
    assert tool_registry.get("activ8_install_me") is not None
    assert (tmp_path / "lib" / "activ8_install_me.py").exists()


def test_activate_tool_idempotent(tmp_path):
    cand = _candidate(name="activ8_idem")
    pid = ts.propose(cand, proposals_path=tmp_path / "p.md",
                     pending_dir=tmp_path / "pending")
    ts.activate(pid, pending_dir=tmp_path / "pending", lib_dir=tmp_path / "lib")
    again = ts.activate(pid, pending_dir=tmp_path / "pending", lib_dir=tmp_path / "lib")
    assert again["ok"] is True and again.get("already") is True


def test_activate_unknown_tool_proposal(tmp_path):
    out = ts.activate("tool-nope-deadbeef00", pending_dir=tmp_path / "pending",
                      lib_dir=tmp_path / "lib")
    assert out["ok"] is False


# ── skill activation (promote from draft) ────────────────────────────


def test_skill_id_from_proposal_parse():
    assert creator._skill_id_from_proposal("prop-skill-abc123def0-1700000000") == "skill-abc123def0"
    assert creator._skill_id_from_proposal("tool-x-y") is None


def _events(tmp_path):
    ev = tmp_path / "events.jsonl"
    rows = []
    for c in (1, 2, 3):
        for tool in ("Read", "Bash"):
            rows.append({"event_class": "tool_call", "agent": "r", "cycle": c, "tool_name": tool})
    ev.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return ev


def test_creator_activate_promotes_draft(tmp_path):
    ev = _events(tmp_path)
    drafts, active = tmp_path / "drafts", tmp_path / "active"
    queued = creator.mine_and_propose(top_n=1, events_path=ev, drafts_dir=drafts,
                                      proposals_path=tmp_path / "pp.md",
                                      min_frequency=2, min_length=2)
    pid = queued[0]["proposal_id"]
    out = creator.activate(pid, drafts_dir=drafts, active_dir=active,
                           validate_in_sandbox=False)
    assert out["ok"] is True
    assert (active / queued[0]["skill_id"]).exists()


# ── unified dispatcher ───────────────────────────────────────────────


def test_dispatch_routes_by_prefix(tmp_path):
    # tool-* path
    cand = _candidate(name="activ8_dispatch")
    pid = ts.propose(cand, proposals_path=tmp_path / "p.md",
                     pending_dir=tmp_path / "pending")
    out = proposal_activate.activate(
        pid, pending_dir=tmp_path / "pending", lib_dir=tmp_path / "lib",
        log_path=tmp_path / "activations.jsonl")
    assert out["ok"] is True and out["kind"] == "tool"
    # an activation audit row is recorded
    assert (tmp_path / "activations.jsonl").exists()


def test_dispatch_unknown_prefix(tmp_path):
    out = proposal_activate.activate("weird-123", log_path=tmp_path / "a.jsonl")
    assert out["ok"] is False


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_propose_writes_sidecar_with_source,
        test_activate_tool_installs_from_sidecar,
        test_activate_tool_idempotent,
        test_activate_unknown_tool_proposal,
        test_skill_id_from_proposal_parse,
        test_creator_activate_promotes_draft,
        test_dispatch_routes_by_prefix,
        test_dispatch_unknown_prefix,
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
