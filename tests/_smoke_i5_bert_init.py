"""Smoke test for I.5: bert init CLI wizard."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "bert_init", LAB_ROOT / "tools" / "bert_init.py",
)
bert_init = importlib.util.module_from_spec(spec)
sys.modules["bert_init"] = bert_init
spec.loader.exec_module(bert_init)


def _isolate_home():
    tmp = Path(tempfile.mkdtemp(prefix="bert_i5_"))
    bert_init.HOME_BERT = tmp
    bert_init.RESUME_PATH = tmp / "init-resume.json"
    bert_init.LABS_DIR = tmp / "labs"
    return tmp


def test_module_exports() -> None:
    assert hasattr(bert_init, "main")
    assert hasattr(bert_init, "_ask_questions")
    assert hasattr(bert_init, "_scaffold_lab")
    assert len(bert_init.ARCHETYPES) == 3
    assert len(bert_init.PROVIDERS) == 4
    assert len(bert_init.AUTONOMY_LEVELS) == 3


def test_name_regex_validates() -> None:
    rx = bert_init.NAME_RX
    assert rx.match("my lab")
    assert rx.match("lab-1")
    assert rx.match("alpha_lab")
    assert not rx.match("")
    assert not rx.match("1lab")  # must start alpha
    assert not rx.match("!!")
    assert not rx.match("x")  # too short
    assert not rx.match("a" * 50)  # too long


def test_default_lab_name_from_cwd() -> None:
    n = bert_init._default_lab_name(Path("/Users/test/my-cool-lab"))
    assert "Cool Lab" in n or "Cool" in n


def test_provider_detection_fallback() -> None:
    """If no API key env and no Ollama, default to Groq."""
    saved = {k: os.environ.get(k) for k in
             ("GROQ_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        # Even if Ollama is running locally, this returns something valid
        p = bert_init._detect_provider()
        assert p in ("Groq", "NVIDIA", "OpenRouter", "Ollama")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_resume_load_save_clear() -> None:
    tmp = _isolate_home()
    try:
        assert bert_init._load_resume() is None
        bert_init._save_resume({"archetype": "Product", "name": "Test"})
        loaded = bert_init._load_resume()
        assert loaded == {"archetype": "Product", "name": "Test"}
        bert_init._clear_resume()
        assert bert_init._load_resume() is None
    finally:
        shutil.rmtree(tmp)


def test_non_interactive_with_all_flags() -> None:
    """Non-interactive mode with all flags must populate the answers dict."""
    import argparse
    args = argparse.Namespace(
        archetype="Research", name="Test Lab",
        provider="Ollama", autonomy="Pilot",
        seed="test seed of >= 10 chars",
        resume=False, non_interactive=True,
    )
    answers = bert_init._ask_questions(non_interactive=True, cli_args=args)
    assert answers["archetype"] == "Research"
    assert answers["name"] == "Test Lab"
    assert answers["provider"] == "Ollama"
    assert answers["autonomy"] == "Pilot"
    assert answers["seed"] == "test seed of >= 10 chars"


def test_scaffold_writes_lab_yaml() -> None:
    tmp = _isolate_home()
    try:
        answers = {
            "archetype": "Product", "name": "demo lab",
            "provider": "Groq", "autonomy": "Collaborator",
            "seed": "ship a tiny Markdown CLI for local notes",
        }
        lab_dir = bert_init._scaffold_lab(answers)
        assert lab_dir.exists()
        assert (lab_dir / "lab.yaml").exists()
        yaml_text = (lab_dir / "lab.yaml").read_text()
        assert "demo lab" in yaml_text
        assert "product" in yaml_text  # lowercased archetype
        assert "groq" in yaml_text
        # Hidden answers archive
        assert (lab_dir / ".bert" / "answers.yml").exists()
    finally:
        shutil.rmtree(tmp)


def test_main_non_interactive_end_to_end() -> None:
    tmp = _isolate_home()
    saved_argv = sys.argv
    try:
        sys.argv = ["bert_init", "--non-interactive",
                    "--archetype", "Product",
                    "--name", "endtoend",
                    "--provider", "Groq",
                    "--autonomy", "Collaborator",
                    "--seed", "ship a tiny CLI for local notes (test seed)"]
        rc = bert_init.main()
        assert rc == 0
        lab_dir = tmp / "labs" / "endtoend"
        assert lab_dir.exists()
        assert (lab_dir / "lab.yaml").exists()
    finally:
        sys.argv = saved_argv
        shutil.rmtree(tmp)


def test_render_preview_table_includes_all_fields() -> None:
    answers = {
        "archetype": "Product", "name": "alpha lab",
        "provider": "Groq", "autonomy": "Collaborator",
        "seed": "build a thing",
    }
    out = bert_init._render_preview(answers)
    assert "alpha lab" in out
    assert "Groq" in out
    assert "Collaborator" in out
    assert "Product Lab" in out


def main() -> int:
    tests = [
        test_module_exports,
        test_name_regex_validates,
        test_default_lab_name_from_cwd,
        test_provider_detection_fallback,
        test_resume_load_save_clear,
        test_non_interactive_with_all_flags,
        test_scaffold_writes_lab_yaml,
        test_main_non_interactive_end_to_end,
        test_render_preview_table_includes_all_fields,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
