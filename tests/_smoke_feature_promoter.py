"""Smoke + TDD: core/feature_promoter.py — feature auto-promotion (Sprint 6 #29).

AC: repeated mission patterns surface as feature suggestions. Missions are
classified once per lab (mission_profile); we emit a `mission_classified`
observability event capturing the profile signature, then this module mines that
stream: missions sharing a (domain, primary_work, data_shape, output_kind)
signature ≥ threshold times surface as a feature SUGGESTION written to
state/feature_promotion_candidates.md for PI review.

SUGGEST only — never auto-activates a feature (activation stays PI-gated, mirrors
creator.propose_promotion for skills). Already-suggested signatures are not
re-proposed (dedupe against the candidates file).

Pure-testable layers: mission_signature, mine_mission_patterns,
already_suggested_signatures. File layer: propose_feature / run. Emit layer:
record_mission_classified (observability.emit stubbed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import feature_promoter as fp  # noqa: E402


def _ev(domain="ml_research", primary_work="discover", data_shape="document_corpus",
        output_kind="report", mission="survey vector DBs"):
    return {"domain": domain, "primary_work": primary_work,
            "data_shape": data_shape, "output_kind": output_kind, "mission": mission}


# ── signature (pure) ─────────────────────────────────────────────────


def test_mission_signature_stable_and_discriminating():
    a = fp.mission_signature(_ev())
    b = fp.mission_signature(_ev())
    c = fp.mission_signature(_ev(primary_work="build"))
    assert a == b
    assert a != c


# ── mining (pure) ────────────────────────────────────────────────────


def test_mine_below_threshold_no_suggestion():
    events = [_ev(), _ev()]  # 2 < 3
    assert fp.mine_mission_patterns(events, min_frequency=3) == []


def test_mine_at_threshold_suggests_with_dims():
    events = [_ev(mission=f"m{i}") for i in range(3)]
    out = fp.mine_mission_patterns(events, min_frequency=3)
    assert len(out) == 1
    s = out[0]
    assert s.count == 3
    assert s.dims["domain"] == "ml_research" and s.dims["primary_work"] == "discover"
    assert len(s.example_missions) >= 1


def test_mine_skips_already_suggested():
    events = [_ev() for _ in range(4)]
    sig = fp.mission_signature(_ev())
    out = fp.mine_mission_patterns(events, min_frequency=3, already_suggested={sig})
    assert out == []


def test_mine_two_distinct_patterns():
    events = [_ev() for _ in range(3)] + [_ev(primary_work="build", data_shape="code_repo") for _ in range(3)]
    out = fp.mine_mission_patterns(events, min_frequency=3)
    assert len(out) == 2


# ── candidates-file dedupe (pure-ish, reads a file) ──────────────────


def test_already_suggested_signatures_parses_file(tmp_path):
    cand = tmp_path / "feature_promotion_candidates.md"
    sig = fp.mission_signature(_ev())
    cand.write_text(f"## feat-abc\n- **signature:** {sig}\n- **status:** pending\n")
    sigs = fp.already_suggested_signatures(cand)
    assert sig in sigs


def test_already_suggested_signatures_missing_file_is_empty(tmp_path):
    assert fp.already_suggested_signatures(tmp_path / "nope.md") == set()


# ── propose (file write) ─────────────────────────────────────────────


def test_propose_feature_writes_candidate(tmp_path):
    cand = tmp_path / "feature_promotion_candidates.md"
    events = [_ev(mission=f"m{i}") for i in range(3)]
    s = fp.mine_mission_patterns(events, min_frequency=3)[0]
    cid = fp.propose_feature(s, candidates_path=cand)
    assert cid.startswith("feat-")
    body = cand.read_text()
    assert "ml_research" in body and "discover" in body
    assert "pending" in body
    assert s.signature in body  # so the next run dedupes it


# ── record_mission_classified (emit) ─────────────────────────────────


def test_record_mission_classified_emits(monkeypatch):
    from core import observability as obs
    captured = {}

    def fake_emit(event_class, payload):
        captured["class"] = event_class
        captured["payload"] = payload

    monkeypatch.setattr(obs, "emit", fake_emit)
    profile = {"domain": "ml_research", "primary_work": "discover",
               "data_shape": "document_corpus", "output_kind": "report",
               "rigor": "cited"}
    fp.record_mission_classified(profile, seed_excerpt="survey vector DBs and rerankers")
    assert captured["class"] == "mission_classified"
    assert captured["payload"]["domain"] == "ml_research"
    assert captured["payload"]["primary_work"] == "discover"
    assert "seed_excerpt" in captured["payload"]


# ── run() end-to-end with dedupe ─────────────────────────────────────


def test_run_end_to_end_then_dedupes(tmp_path):
    events_path = tmp_path / "mission_classified.jsonl"
    cand = tmp_path / "feature_promotion_candidates.md"
    with events_path.open("w") as f:
        for i in range(3):
            f.write(json.dumps(_ev(mission=f"m{i}")) + "\n")
    first = fp.run(events_path=events_path, candidates_path=cand, min_frequency=3)
    assert len(first) == 1
    # second run sees the same events but the signature is already suggested
    second = fp.run(events_path=events_path, candidates_path=cand, min_frequency=3)
    assert second == []


# ── standalone runner ────────────────────────────────────────────────


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
        test_mission_signature_stable_and_discriminating,
        test_mine_below_threshold_no_suggestion,
        test_mine_at_threshold_suggests_with_dims,
        test_mine_skips_already_suggested,
        test_mine_two_distinct_patterns,
        test_already_suggested_signatures_parses_file,
        test_already_suggested_signatures_missing_file_is_empty,
        test_propose_feature_writes_candidate,
        test_record_mission_classified_emits,
        test_run_end_to_end_then_dedupes,
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
