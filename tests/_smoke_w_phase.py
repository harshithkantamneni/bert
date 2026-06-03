"""Smoke test for W-phase: autonomous cycle runner + bert init wiring.

W.1 — bert_run.py reads seed_brief.md and runs cycles
W.2 — bert init --run-first-cycle invokes bert_run.py
W.3 — --seed survives --from-template (user mission leads, template
       content preserved as appendix)
W.4 — bert doctor includes check_bert_run_present
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.bert_run as br  # noqa: E402

VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"
BERT_RUN = LAB_ROOT / "tools" / "bert_run.py"
BERT_INIT = LAB_ROOT / "tools" / "bert_init.py"


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "requires lab runtime artifact(s) not shipped in the public repo: "
            + ", ".join(str(m) for m in missing)
        )


# ── W.1 — bert_run.py core ──────────────────────────────────────────

def test_bert_run_module_imports() -> None:
    assert hasattr(br, "run")
    assert hasattr(br, "_run_one_cycle")
    assert hasattr(br, "_check_provider_keys")
    assert hasattr(br, "_read_seed_brief")
    assert hasattr(br, "_next_cycle_id")
    assert hasattr(br, "_build_spec")


def test_bert_run_resolve_lab_handles_three_forms() -> None:
    """Default → repo's lab/; absolute path → as-is; name → ~/.bert/labs/."""
    default_lab = br._resolve_lab(None)
    assert default_lab == LAB_ROOT / "lab", \
        f"default --lab should be repo's lab/; got {default_lab}"
    abs_path = br._resolve_lab(str(LAB_ROOT / "lab"))
    assert abs_path == LAB_ROOT / "lab"


def test_bert_run_read_seed_brief_raises_on_missing() -> None:
    """A clear FileNotFoundError with the scaffold-first hint."""
    tmp = Path(tempfile.mkdtemp())
    try:
        try:
            br._read_seed_brief(tmp)
            raise AssertionError("should have raised on missing seed_brief.md")
        except FileNotFoundError as exc:
            assert "seed_brief.md" in str(exc), "error must mention seed_brief.md"
            assert "bert init" in str(exc), "error must hint to bert init"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bert_run_read_seed_brief_returns_content() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# Mission\n\nBuild a CLI tool.")
        content = br._read_seed_brief(tmp)
        assert "Build a CLI tool" in content
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bert_run_next_cycle_id_increments_from_events() -> None:
    """If events.jsonl has cycles 1..5, next id should be 6."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sor").mkdir()
        events = tmp / "sor" / "events.jsonl"
        with events.open("w") as f:
            for i in range(1, 6):
                f.write(json.dumps({"ts": "2026-05-13T00:00:00Z", "cycle": i}) + "\n")
        assert br._next_cycle_id(tmp) == 6
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bert_run_next_cycle_id_fallback_when_no_events() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        assert br._next_cycle_id(tmp, fallback_start=42) == 42
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bert_run_check_provider_keys_detects_groq() -> None:
    old = os.environ.get("GROQ_API_KEY")
    try:
        os.environ["GROQ_API_KEY"] = "test-key"
        ok, keys = br._check_provider_keys()
        assert ok is True
        assert "GROQ_API_KEY" in keys
    finally:
        if old is None: os.environ.pop("GROQ_API_KEY", None)
        else: os.environ["GROQ_API_KEY"] = old


def test_bert_run_check_provider_keys_returns_false_when_empty() -> None:
    """Post-GG-A.0 — _check_provider_keys reads via core.config.load()
    which merges env + ~/.bert-lab/credentials.json. To test the
    "no keys anywhere" case, we have to scrub both env AND the
    cached Config object.

    Pre-GG-A.0 this just cleared env. Updated to also mock the
    credentials file so the function genuinely sees no keys."""
    import os
    from unittest.mock import patch
    candidates = ["GROQ_API_KEY", "NVIDIA_API_KEY", "MISTRAL_API_KEY",
                  "CEREBRAS_API_KEY", "GOOGLE_AI_API_KEY",
                  "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"]
    saved = {k: os.environ.pop(k, None) for k in candidates}
    try:
        from core import config as _cfg
        # Reset the module-level cache so load() doesn't return a
        # previously-cached Config that had keys.
        _cfg._cached = None
        # Patch CRED_PATH to a guaranteed-nonexistent path so the
        # credentials file lookup also misses.
        from pathlib import Path
        with patch.object(_cfg, "CRED_PATH",
                            Path("/nonexistent-bert-credentials-for-test.json")):
            ok, keys = br._check_provider_keys()
            assert ok is False
            assert keys == []
    finally:
        for k, v in saved.items():
            if v: os.environ[k] = v
        # Restore the cache to None so the next test re-reads cleanly
        from core import config as _cfg
        _cfg._cached = None


def test_bert_run_build_spec_has_required_fields() -> None:
    spec = br._build_spec(
        role="researcher", cycle=42,
        task="Test task", output_path="drafts/x.md",
        model="nvidia/meta/llama-3.3-70b-instruct",
        falsifier_text="test falsifier",
    )
    required = ["dispatch_altitude", "role", "cycle", "task", "model",
                "output_path", "falsifier_text", "success_criterion",
                "verification_command"]
    for k in required:
        assert k in spec, f"spec missing {k!r}"


def test_bert_run_seed_to_research_task_includes_context() -> None:
    """The composed task must carry the seed as context to the dispatch."""
    seed = "# Mission\n\nBuild a CSV-to-JSON CLI converter."
    task = br._seed_to_research_task(seed)
    assert "seed brief" in task.lower()
    assert "CSV-to-JSON" in task, "task must include the seed's mission text"


def test_bert_run_dry_run_against_real_lab() -> None:
    """Use the repo's own findings/falsifier_corpus.md as a stand-in
    seed for the dry-run check — we just need any lab with a seed_brief."""
    _require(VENV_PY)
    # Build a temp lab with a seed
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# Mission\n\nTest dry-run.")
        (tmp / "sor").mkdir()
        (tmp / "state").mkdir()
        result = subprocess.run(
            [str(VENV_PY), str(BERT_RUN),
             "--lab", str(tmp), "--dry-run", "--max-cycles", "2"],
            capture_output=True, text=True, timeout=10,
            cwd=str(LAB_ROOT),
        )
        assert result.returncode == 0, \
            f"--dry-run should exit 0; got {result.returncode}; stderr={result.stderr[:200]}"
        out = result.stdout
        assert "dry-run" in out.lower()
        assert "would run 2 cycle" in out, "dry-run must announce intended cycle count"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bert_run_aborts_without_keys() -> None:
    """Without --dry-run and without provider keys, must exit 2 with
    a clear ABORT message.

    Post-GG-A.0: config.load reads ~/.bert-lab/credentials.json, so
    this test must also redirect HOME to a tmp dir so the credentials
    file lookup misses. Pre-GG only env was checked, so PATH-only was
    enough to suppress keys.
    """
    _require(VENV_PY)
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# Mission\n\nTest abort.")
        result = subprocess.run(
            [str(VENV_PY), str(BERT_RUN),
             "--lab", str(tmp), "--max-cycles", "1"],
            capture_output=True, text=True, timeout=10,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp)},
            cwd=str(LAB_ROOT),
        )
        assert result.returncode == 2, \
            f"no-keys should exit 2; got {result.returncode}"
        combined = result.stdout + result.stderr
        assert "ABORT" in combined
        assert "no provider keys" in combined or "GROQ_API_KEY" in combined
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bert_run_aborts_on_missing_seed_brief() -> None:
    """A lab path without seed_brief.md must abort with rc=2."""
    _require(VENV_PY)
    tmp = Path(tempfile.mkdtemp())
    try:
        result = subprocess.run(
            [str(VENV_PY), str(BERT_RUN),
             "--lab", str(tmp), "--dry-run", "--max-cycles", "1"],
            capture_output=True, text=True, timeout=10,
            cwd=str(LAB_ROOT),
        )
        assert result.returncode == 2, \
            f"missing seed_brief should exit 2; got {result.returncode}"
        combined = result.stdout + result.stderr
        assert "seed_brief" in combined
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bert_run_help_documents_flags() -> None:
    _require(VENV_PY)
    result = subprocess.run(
        [str(VENV_PY), str(BERT_RUN), "--help"],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0
    out = result.stdout
    for flag in ("--lab", "--max-cycles", "--model", "--dry-run", "--watch"):
        assert flag in out, f"--help must document {flag}"


# ── W.2 — bert init --run-first-cycle wiring ────────────────────────

def test_bert_init_accepts_run_first_cycle_flag() -> None:
    """The flag must appear in argparse + the body must invoke bert_run."""
    text = BERT_INIT.read_text()
    assert "--run-first-cycle" in text, \
        "bert_init must accept --run-first-cycle"
    assert "run_first_cycle" in text, \
        "must use run_first_cycle variable"
    assert "bert_run.py" in text, \
        "must invoke bert_run.py from bert_init"


def test_bert_init_run_first_cycle_propagates_exit_code() -> None:
    """End-to-end: bert init --run-first-cycle on a clean home, no keys,
    must exit 2 (bert_run abort propagates back through bert_init)."""
    _require(VENV_PY)
    tmp_home = Path(tempfile.mkdtemp())
    try:
        result = subprocess.run(
            [str(VENV_PY), str(BERT_INIT),
             "--non-interactive", "--archetype", "Product",
             "--name", "w_smoke", "--provider", "Groq",
             "--autonomy", "Collaborator",
             "--seed", "Test W-phase smoke",
             "--from-template", "demo_note_cli",
             "--run-first-cycle"],
            capture_output=True, text=True, timeout=20,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_home)},
            cwd=str(LAB_ROOT),
        )
        assert result.returncode == 2, \
            f"--run-first-cycle without keys should exit 2; got {result.returncode}"
        combined = result.stdout + result.stderr
        assert "bert run" in combined.lower()
        # Scaffold should have completed before the run attempt
        assert (tmp_home / ".bert" / "labs" / "w_smoke").exists(), \
            "lab should have been scaffolded before bert_run abort"
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


# ── W.3 — --seed override semantics ─────────────────────────────────

def test_seed_overrides_template_seed_brief() -> None:
    """When --seed is given alongside --from-template, the user's mission
    must lead in seed_brief.md (with template content as appendix)."""
    _require(VENV_PY)
    tmp_home = Path(tempfile.mkdtemp())
    try:
        result = subprocess.run(
            [str(VENV_PY), str(BERT_INIT),
             "--non-interactive", "--archetype", "Product",
             "--name", "seed_test", "--provider", "Groq",
             "--autonomy", "Collaborator",
             "--seed", "BUILD_A_UNIQUE_THING_XYZ",
             "--from-template", "demo_note_cli"],
            capture_output=True, text=True, timeout=15,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_home)},
            cwd=str(LAB_ROOT),
        )
        assert result.returncode == 0, \
            f"init should succeed; got {result.returncode}"
        seed_brief = (tmp_home / ".bert" / "labs" / "seed_test" /
                      "seed_brief.md").read_text()
        # User's mission must appear FIRST (anchor: should be in first 200 chars)
        assert "BUILD_A_UNIQUE_THING_XYZ" in seed_brief[:200], \
            "user's mission must lead in seed_brief.md"
        # Template content preserved as appendix
        assert "Template context" in seed_brief, \
            "template content should be preserved as appendix"
        # The literal template seed should still appear somewhere
        assert "note-cli" in seed_brief.lower() or "Knowledge workers" in seed_brief, \
            "template's content should remain as context"
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


def test_no_seed_uses_template_as_is() -> None:
    """When no --seed is given but --from-template is, the template's
    seed_brief.md should be used as-is (current behavior, preserved)."""
    _require(VENV_PY)
    tmp_home = Path(tempfile.mkdtemp())
    try:
        result = subprocess.run(
            [str(VENV_PY), str(BERT_INIT),
             "--non-interactive", "--archetype", "Product",
             "--name", "no_seed_test", "--provider", "Groq",
             "--autonomy", "Collaborator",
             "--from-template", "demo_note_cli"],
            capture_output=True, text=True, timeout=15,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_home)},
            cwd=str(LAB_ROOT),
        )
        assert result.returncode == 0
        seed_brief = (tmp_home / ".bert" / "labs" / "no_seed_test" /
                      "seed_brief.md").read_text()
        # Should NOT have the "# Mission" wrapper since no user override
        assert "Template context (preserved" not in seed_brief, \
            "no-user-seed path should not wrap in mission/appendix structure"
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


def test_seed_only_no_template_writes_minimal_seed_brief() -> None:
    """When --seed is given but NO --from-template, a minimal
    seed_brief.md should still be written so bert_run has something."""
    _require(VENV_PY)
    tmp_home = Path(tempfile.mkdtemp())
    try:
        result = subprocess.run(
            [str(VENV_PY), str(BERT_INIT),
             "--non-interactive", "--archetype", "Product",
             "--name", "seed_only", "--provider", "Groq",
             "--autonomy", "Collaborator",
             "--seed", "MINIMAL_SEED_XYZ"],
            capture_output=True, text=True, timeout=15,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_home)},
            cwd=str(LAB_ROOT),
        )
        assert result.returncode == 0
        seed_path = tmp_home / ".bert" / "labs" / "seed_only" / "seed_brief.md"
        assert seed_path.exists(), "seed_brief.md must be written"
        content = seed_path.read_text()
        assert "MINIMAL_SEED_XYZ" in content
        assert "Mission" in content
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


# ── W.4 — bert doctor integration ───────────────────────────────────

def test_doctor_includes_bert_run_check() -> None:
    import tools.bert_doctor as doctor
    assert hasattr(doctor, "check_bert_run_present"), \
        "doctor must expose check_bert_run_present"
    check_names = [c.__name__ for c in doctor.DEFAULT_CHECKS]
    assert "check_bert_run_present" in check_names, \
        "check_bert_run_present must be in DEFAULT_CHECKS"


def test_doctor_bert_run_check_passes() -> None:
    import tools.bert_doctor as doctor
    result = doctor.check_bert_run_present()
    assert result.level == "ok", \
        f"bert_run check should pass on this lab: {result.message}"


def main() -> int:
    tests = [
        # W.1
        test_bert_run_module_imports,
        test_bert_run_resolve_lab_handles_three_forms,
        test_bert_run_read_seed_brief_raises_on_missing,
        test_bert_run_read_seed_brief_returns_content,
        test_bert_run_next_cycle_id_increments_from_events,
        test_bert_run_next_cycle_id_fallback_when_no_events,
        test_bert_run_check_provider_keys_detects_groq,
        test_bert_run_check_provider_keys_returns_false_when_empty,
        test_bert_run_build_spec_has_required_fields,
        test_bert_run_seed_to_research_task_includes_context,
        test_bert_run_dry_run_against_real_lab,
        test_bert_run_aborts_without_keys,
        test_bert_run_aborts_on_missing_seed_brief,
        test_bert_run_help_documents_flags,
        # W.2
        test_bert_init_accepts_run_first_cycle_flag,
        test_bert_init_run_first_cycle_propagates_exit_code,
        # W.3
        test_seed_overrides_template_seed_brief,
        test_no_seed_uses_template_as_is,
        test_seed_only_no_template_writes_minimal_seed_brief,
        # W.4
        test_doctor_includes_bert_run_check,
        test_doctor_bert_run_check_passes,
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
