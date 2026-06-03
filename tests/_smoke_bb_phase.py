"""Smoke test for BB-phase: bug-fix cleanup.

  BB.1 — _index_corpus respects BERT_SKIP_INDEXER + graceful on
         sqlite corruption (was hanging robustness for >5 min)
  BB.2 — run_canonical.sh no longer silently exits on first failure
         (was `set -e` + substitution swallowing stdout)
  BB.3 — Mission UI disables RUN/dry-run buttons when content is empty
  BB.4 — Doctor: missing GROQ → WARN if other keys present, FAIL only
         when zero keys
  BB.5 — Default lab/seed_brief.md exists so bert_run.py works against
         the default lab
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"
MEMORY_PY = LAB_ROOT / "core" / "memory.py"
RUN_CANONICAL = LAB_ROOT / "tests" / "run_canonical.sh"
MISSION_TSX = LAB_ROOT / "bert" / "v4" / "src" / "components" / "RunCycleControls.tsx"
DOCTOR = LAB_ROOT / "tools" / "bert_doctor.py"
DEFAULT_SEED = LAB_ROOT / "lab" / "seed_brief.md"


# ── BB.1: indexer escape hatch + corruption tolerance ──────────────

def test_index_corpus_respects_skip_env() -> None:
    text = MEMORY_PY.read_text()
    assert "BERT_SKIP_INDEXER" in text, \
        "_index_corpus must honor BERT_SKIP_INDEXER"
    result = subprocess.run(
        [str(VENV_PY), "-c",
         "import sys; sys.path.insert(0, '.'); "
         "from core import memory; print('idx:', memory._index_corpus())"],
        capture_output=True, text=True, timeout=10,
        env={**os.environ, "BERT_SKIP_INDEXER": "1"},
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, \
        f"_index_corpus with skip flag should not raise; got {result.stderr[:200]}"
    assert "idx: 0" in result.stdout, \
        f"with skip=1, should return 0 immediately. Got: {result.stdout!r}"


def test_robustness_completes_fast_under_skip() -> None:
    # 120s budget accommodates sentence-transformers cold-cache load
    # on disk-pressured machines. "fast" relative to the alternative
    # (full indexer run takes minutes).
    result = subprocess.run(
        [str(VENV_PY), str(LAB_ROOT / "tests" / "_smoke_robustness.py")],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "BERT_SKIP_INDEXER": "1"},
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, \
        f"robustness should pass under skip; got rc={result.returncode}, " \
        f"stderr={result.stderr[:300]}"
    assert "All 9 robustness tests passed" in result.stdout


# ── BB.2: run_canonical.sh stdout reliability ──────────────────────

def test_run_canonical_no_set_e() -> None:
    text = RUN_CANONICAL.read_text()
    lines = [l.strip() for l in text.splitlines()]
    assert "set -e" not in lines, \
        "run_canonical.sh must NOT use `set -e` (substitution swallows stdout)"


def test_run_canonical_exports_skip_indexer() -> None:
    text = RUN_CANONICAL.read_text()
    assert "BERT_SKIP_INDEXER" in text, \
        "run_canonical.sh must export BERT_SKIP_INDEXER=1"


def test_run_canonical_produces_summary_line() -> None:
    """BB.2 structural fix: the script must produce a summary line on
    every invocation (was silently exiting on failure). We can't invoke
    `bash run_canonical.sh` end-to-end here — that runs ALL canonical
    smokes INCLUDING this one, causing infinite recursion. Instead we
    verify the script's *invariant*: the echo statement that prints
    the summary always runs (no early exit, no set -e clobbering it)."""
    text = RUN_CANONICAL.read_text()
    # The summary echo must be unconditional (no if-guard, no exit
    # before it in the happy path).
    assert 'echo "Canonical regression:' in text, \
        "summary echo line missing from run_canonical.sh"
    # The summary echo must come BEFORE any per-failure list +
    # AFTER the test loop. Look for the relative order.
    lines = text.splitlines()
    summary_idx = next((i for i, l in enumerate(lines)
                        if 'echo "Canonical regression:' in l), -1)
    loop_idx = next((i for i, l in enumerate(lines)
                     if l.strip().startswith("for f in tests/_smoke")), -1)
    assert loop_idx > 0 and summary_idx > loop_idx, \
        "summary echo must come after the test loop"
    # The script must NOT have an `exit 1` between the loop and the
    # summary (that would skip the summary on failure).
    between = lines[loop_idx:summary_idx]
    early_exit = [l for l in between if l.strip().startswith("exit ")]
    assert not early_exit, \
        f"early exit before summary would swallow stdout: {early_exit}"


# ── BB.3: Mission UI button enable logic ───────────────────────────

def test_mission_run_disabled_when_empty() -> None:
    text = MISSION_TSX.read_text()
    assert "hasContent" in text, "Mission.tsx must track hasContent state"
    assert "canRun" in text, "Mission.tsx must use a canRun computed flag"
    assert "write a mission first" in text, \
        "Mission.tsx button title must explain the empty-content disabled state"


# ── BB.4: doctor severity logic ────────────────────────────────────

def test_doctor_warn_when_other_keys_present() -> None:
    import tools.bert_doctor as doctor
    saved = {k: os.environ.pop(k, None) for k in (
        "GROQ_API_KEY", "NVIDIA_API_KEY", "MISTRAL_API_KEY",
        "OPENROUTER_API_KEY", "GOOGLE_API_KEY")}
    try:
        os.environ["NVIDIA_API_KEY"] = "nvapi-test"
        result = doctor.check_groq_key()
        assert result.level == "warn", \
            f"missing GROQ + NVIDIA present should be WARN; got {result.level}"
        assert "falling back" in result.message.lower(), \
            f"warn message should explain fallback: {result.message!r}"
    finally:
        for k, v in saved.items():
            if v: os.environ[k] = v
        os.environ.pop("NVIDIA_API_KEY", None)


def test_doctor_fail_when_no_keys() -> None:
    import tools.bert_doctor as doctor
    saved = {k: os.environ.pop(k, None) for k in (
        "GROQ_API_KEY", "NVIDIA_API_KEY", "MISTRAL_API_KEY",
        "OPENROUTER_API_KEY", "GOOGLE_API_KEY")}
    try:
        result = doctor.check_groq_key()
        assert result.level == "fail", \
            f"all keys missing should be FAIL; got {result.level}"
    finally:
        for k, v in saved.items():
            if v: os.environ[k] = v


# ── BB.5: default lab seed_brief ───────────────────────────────────

def test_default_seed_brief_exists() -> None:
    assert DEFAULT_SEED.exists(), \
        "lab/seed_brief.md must exist so bert_run.py works against the default lab"


def test_default_seed_brief_has_mission_structure() -> None:
    text = DEFAULT_SEED.read_text()
    assert "# Mission" in text
    assert len(text) >= 500
    for axis in ("Routing", "Memory", "Discipline", "UX"):
        assert axis in text, f"default seed should reference the '{axis}' axis"


def test_default_seed_references_honest_disclosure() -> None:
    text = DEFAULT_SEED.read_text()
    for disclosure in ("heuristic-v1", "placeholder", "local-dev",
                       "no acquired customers"):
        assert disclosure in text, \
            f"default seed must reference '{disclosure}' for honest scope"


def test_bert_run_dry_run_finds_default_seed() -> None:
    result = subprocess.run(
        [str(VENV_PY), str(LAB_ROOT / "tools" / "bert_run.py"),
         "--dry-run", "--max-cycles", "1"],
        capture_output=True, text=True, timeout=10,
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, \
        f"bert_run --dry-run should succeed against default lab; got {result.returncode}"
    combined = result.stdout + result.stderr
    assert "[seed]" in combined and "loaded" in combined


def main() -> int:
    tests = [
        test_index_corpus_respects_skip_env,
        test_robustness_completes_fast_under_skip,
        test_run_canonical_no_set_e,
        test_run_canonical_exports_skip_indexer,
        test_run_canonical_produces_summary_line,
        test_mission_run_disabled_when_empty,
        test_doctor_warn_when_other_keys_present,
        test_doctor_fail_when_no_keys,
        test_default_seed_brief_exists,
        test_default_seed_brief_has_mission_structure,
        test_default_seed_references_honest_disclosure,
        test_bert_run_dry_run_finds_default_seed,
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
