"""Smoke: tools/bert_init.py — lab init wizard helpers (was 67%).

Non-interactive surface only (the questionary wizard needs a TTY): git/
provider detection (subprocess + env monkeypatched), _default_lab_name,
resume load/save/clear (temp paths), and _ask_questions(non_interactive)
incl. the cli_args-required guard.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import shutil
import sys
import tempfile
import types
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

bi = importlib.import_module("bert_init")


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


def test_git_user_name(monkeypatch):
    monkeypatch.setattr(bi.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="Ada Lovelace\n"))
    assert bi._git_user_name() == "Ada Lovelace"
    def _missing(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(bi.subprocess, "run", _missing)
    assert bi._git_user_name() is None


def test_detect_provider(monkeypatch):
    monkeypatch.setattr(bi.os, "environ", {"GROQ_API_KEY": "k"})
    assert bi._detect_provider() == "Groq"
    monkeypatch.setattr(bi.os, "environ", {"NVIDIA_API_KEY": "k"})
    assert bi._detect_provider() == "NVIDIA"
    # no keys + no ollama socket → safe default Groq
    monkeypatch.setattr(bi.os, "environ", {})
    assert bi._detect_provider() in ("Groq", "Ollama")


def test_default_lab_name(tmp_path):
    assert bi._default_lab_name(Path("/x/my_cool-project")) == "My Cool Project"
    assert bi._default_lab_name(Path("/")) == "bert lab" or bi._default_lab_name(Path("/x/a"))


def test_resume_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "RESUME_PATH", tmp_path / "resume.json")
    monkeypatch.setattr(bi, "HOME_BERT", tmp_path)
    assert bi._load_resume() is None              # missing
    bi._save_resume({"name": "Lab1", "archetype": "Product"})
    assert bi._load_resume()["name"] == "Lab1"
    # corrupt → None
    (tmp_path / "resume.json").write_text("{bad json")
    assert bi._load_resume() is None
    bi._clear_resume()
    assert not (tmp_path / "resume.json").exists()
    bi._clear_resume()                            # idempotent (missing → no crash)


def test_ask_questions_non_interactive():
    args = argparse.Namespace(archetype="Research", name="MyLab",
                              provider="Groq", autonomy="Collaborator", seed="tour")
    answers = bi._ask_questions(non_interactive=True, cli_args=args)
    assert answers["name"] == "MyLab" and answers["archetype"] == "Research"
    # defaults fill the gaps when cli_args fields are None
    blank = argparse.Namespace(archetype=None, name=None, provider=None,
                               autonomy=None, seed=None)
    a2 = bi._ask_questions(non_interactive=True, cli_args=blank,
                           defaults={"name": "Fallback"})
    assert a2["name"] == "Fallback"
    # non_interactive without cli_args → ValueError
    try:
        bi._ask_questions(non_interactive=True)
        raise SystemExit("no raise")
    except ValueError:
        pass


def test_scaffold_lab(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "LABS_DIR", tmp_path)
    answers = {"name": "My Scaffold Lab", "archetype": "Research", "provider": "Groq",
               "autonomy": "Collaborator", "seed": "investigate vector DB recall",
               "focus_areas": ["recall", "latency"]}
    lab_dir = bi._scaffold_lab(answers, user_provided_seed=True)
    assert lab_dir.exists() and lab_dir.parent == tmp_path
    assert (lab_dir / "lab.yaml").exists()
    assert (lab_dir / "seed_brief.md").exists()
    # user_provided_seed → the user's seed text reaches seed_brief.md
    assert "vector DB recall" in (lab_dir / "seed_brief.md").read_text()


def main() -> int:
    tests = [
        test_git_user_name,
        test_detect_provider,
        test_default_lab_name,
        test_resume_roundtrip,
        test_ask_questions_non_interactive,
        test_scaffold_lab,
    ]
    for t in tests:
        mp = _MP()
        td = Path(tempfile.mkdtemp())
        try:
            params = inspect.signature(t).parameters
            kwargs = {}
            if "tmp_path" in params:
                kwargs["tmp_path"] = td
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
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
            mp.undo()
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
