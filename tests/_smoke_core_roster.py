"""Smoke: core/roster.py — template spawn + promotion tracking (was 62%).

File-based against a temp lab + the real template library. Covers tracker
load/save, _find_template_file (hit + miss), list_permanent_roster,
spawn_inline (base / already-permanent / inline-spec / bad-template /
empty), list_specializations, candidates_for_promotion (after ≥3 reuses),
mark_promoted (hit + miss), and the spawn/list/candidates CLI.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import roster  # noqa: E402

# discover a real base template name from the library
_TEMPLATE = next(
    (p.stem for p in (roster.LIBRARY_DIR / "agents" / "_base").glob("*.md")),
    None) or next((p.stem for p in (roster.LIBRARY_DIR / "agents").glob("**/*.md")), "researcher")


def test_tracker_roundtrip(tmp_path):
    t = roster.load_tracker(tmp_path)            # missing → fresh
    assert isinstance(t, roster.SpawnTracker)
    t.record_use(_TEMPLATE, "deep_diver", cycle=1)
    roster.save_tracker(tmp_path, t)
    reloaded = roster.load_tracker(tmp_path)
    assert reloaded.specializations
    # corrupt tracker → fresh (no crash)
    roster._tracker_path(tmp_path).write_text("{not json")
    assert isinstance(roster.load_tracker(tmp_path), roster.SpawnTracker)


def test_find_template_file():
    assert roster._find_template_file(_TEMPLATE) is not None
    assert roster._find_template_file("definitely_not_a_template_xyz") is None


def test_list_permanent_roster(tmp_path):
    assert roster.list_permanent_roster(tmp_path) == []     # no agents dir
    role_dir = tmp_path / "agents" / "custom_role"
    role_dir.mkdir(parents=True)
    (role_dir / "procedural.md").write_text("# role")
    (tmp_path / "agents" / "_hidden").mkdir()               # underscore → skipped
    assert roster.list_permanent_roster(tmp_path) == ["custom_role"]


def test_spawn_inline(tmp_path):
    # empty + bad template
    assert roster.spawn_inline(lab_path=tmp_path, template="")["ok"] is False
    assert roster.spawn_inline(lab_path=tmp_path, template="no_such_tpl_xyz")["ok"] is False
    # base template (no inline)
    base = roster.spawn_inline(lab_path=tmp_path, template=_TEMPLATE)
    assert base["ok"] and base["already_permanent"] is False and base["procedural"]
    # already-permanent role
    perm = tmp_path / "agents" / _TEMPLATE
    perm.mkdir(parents=True)
    (perm / "procedural.md").write_text("# permanent")
    base2 = roster.spawn_inline(lab_path=tmp_path, template=_TEMPLATE)
    assert base2["already_permanent"] is True
    # inline specialization records the tracker
    inl = roster.spawn_inline(lab_path=tmp_path, template=_TEMPLATE,
                              inline_name="deep_diver", cycle=2)
    assert inl["ok"] and inl["role"] == f"{_TEMPLATE}__deep_diver"
    assert inl["use_count"] >= 1


def test_promotion_tracking(tmp_path):
    for c in range(3):                          # 3 reuses → promotion candidate
        roster.spawn_inline(lab_path=tmp_path, template=_TEMPLATE,
                            inline_name="auditor", cycle=c)
    specs = roster.list_specializations(tmp_path)
    assert any(s["key"].endswith("auditor") for s in specs)
    cands = roster.candidates_for_promotion(tmp_path, threshold=3)
    assert cands and any(c.inline_name == "auditor" for c in cands)
    assert roster.mark_promoted(tmp_path, _TEMPLATE, "auditor") is True
    # after promotion → no longer a candidate
    assert roster.candidates_for_promotion(tmp_path, threshold=3) == [] or \
        all(c.inline_name != "auditor" for c in roster.candidates_for_promotion(tmp_path, threshold=3))
    assert roster.mark_promoted(tmp_path, _TEMPLATE, "never_spawned") is False


def test_cli(tmp_path):
    assert roster._cli(["x"]) == 2
    assert roster._cli(["x", "spawn"]) == 2
    with contextlib.redirect_stdout(io.StringIO()):
        assert roster._cli(["x", "spawn", str(tmp_path), _TEMPLATE]) == 0
        assert roster._cli(["x", "spawn", str(tmp_path), _TEMPLATE, "spec1", "3"]) == 0
        assert roster._cli(["x", "list", str(tmp_path)]) == 0
        assert roster._cli(["x", "candidates", str(tmp_path)]) == 0


def main() -> int:
    tests = [
        test_tracker_roundtrip,
        test_find_template_file,
        test_list_permanent_roster,
        test_spawn_inline,
        test_promotion_tracking,
        test_cli,
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
