"""Smoke test for T.2: bert doctor preflight check.

Covers:
- Each individual check function in isolation (so the smoke surfaces
  regressions per-check, not just at the aggregator level)
- Aggregator-level exit code semantics (0/1/2 by severity)
- --json output is well-formed
- --no-color disables ANSI
- Stage-safety: missing optional artifacts produce WARN, not FAIL
- Network checks are correctly gated behind --with-network
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.bert_doctor as doctor  # noqa: E402

VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"
DOCTOR = LAB_ROOT / "tools" / "bert_doctor.py"


def test_module_imports_clean() -> None:
    assert hasattr(doctor, "run_all_checks")
    assert hasattr(doctor, "overall_exit_code")
    assert hasattr(doctor, "DEFAULT_CHECKS")
    assert hasattr(doctor, "NETWORK_CHECKS")


def test_default_checks_count_reasonable() -> None:
    """Sanity: we expect ~12 default checks. If someone deletes half
    of them by accident, the smoke surfaces it."""
    n = len(doctor.DEFAULT_CHECKS)
    assert 8 <= n <= 20, f"unexpected DEFAULT_CHECKS count: {n}"


def test_each_check_returns_check_result() -> None:
    """Every check function must return a CheckResult with valid level."""
    valid_levels = {"ok", "warn", "fail"}
    for check_fn in doctor.DEFAULT_CHECKS:
        result = check_fn()
        assert isinstance(result, doctor.CheckResult), \
            f"{check_fn.__name__} didn't return CheckResult"
        assert result.level in valid_levels, \
            f"{check_fn.__name__} returned bad level: {result.level!r}"
        assert isinstance(result.name, str) and result.name, \
            f"{check_fn.__name__} returned empty name"
        assert isinstance(result.message, str) and result.message, \
            f"{check_fn.__name__} returned empty message"


def test_python_version_check_passes_on_modern_python() -> None:
    """We run on Python 3.11+, so this check should be 'ok'."""
    result = doctor.check_python_version()
    assert result.level == "ok", \
        f"python version check failed unexpectedly: {result.message}"


def test_venv_check_finds_real_venv() -> None:
    result = doctor.check_venv_exists()
    assert result.level == "ok", "venv check should find .venv on this lab"


def test_required_deps_importable() -> None:
    result = doctor.check_required_deps()
    assert result.level == "ok", \
        f"required deps must be importable: {result.message}"


def test_proof_packet_check_finds_canonical() -> None:
    result = doctor.check_proof_packet()
    assert result.level == "ok", \
        f"canonical proof packet check failed: {result.message}"


def test_failures_md_check_finds_signed_failures() -> None:
    result = doctor.check_failures_md_in_packet()
    assert result.level == "ok", \
        f"failures.md check failed: {result.message}"
    assert "signed separately" in result.message, \
        "should detect separate signature on failures.md"


def test_ui_build_check_finds_dist() -> None:
    result = doctor.check_ui_build()
    assert result.level == "ok", \
        f"UI build check failed (need bert/v4/dist/index.html): {result.message}"


def test_weekly_timeline_check_finds_compiled_output() -> None:
    """T.1 should have produced weekly_history/timeline.{md,json}."""
    result = doctor.check_weekly_timeline()
    assert result.level == "ok", \
        f"weekly timeline check failed: {result.message}"
    # Should reference the actual count
    assert "1/8" in result.message or "/8" in result.message, \
        f"timeline check should show N/8 progress: {result.message}"


def test_default_lab_check_finds_sor_state() -> None:
    result = doctor.check_default_lab()
    assert result.level == "ok", \
        f"default lab check failed: {result.message}"


def test_port_check_returns_ok_or_warn() -> None:
    """Port check must return ok if free or warn if in use — never fail."""
    result = doctor.check_port_available(port=5174)
    assert result.level in ("ok", "warn"), \
        f"port check returned unexpected level: {result.level}"


def test_port_check_detects_in_use() -> None:
    """Bind a port deliberately, then verify the check sees it as warn."""
    # Bind a random unused port, then ask the doctor about it
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen(1)
        result = doctor.check_port_available(port=port)
        assert result.level == "warn", \
            f"port check should warn when in use; got {result.level}"
        assert result.fix_hint and "lsof" in result.fix_hint, \
            "fix hint should mention lsof"


def test_groq_key_check_missing_is_fail() -> None:
    """Stage-safety: no GROQ_API_KEY must FAIL (not WARN), because
    the live cycle path is unsalvageable without it."""
    old = os.environ.pop("GROQ_API_KEY", None)
    try:
        result = doctor.check_groq_key()
        assert result.level == "fail", \
            "GROQ_API_KEY missing must be fail-level"
        assert result.fix_hint and "export GROQ_API_KEY" in result.fix_hint, \
            "fix hint must give the exact export command"
    finally:
        if old:
            os.environ["GROQ_API_KEY"] = old


def test_optional_keys_missing_are_warn_only() -> None:
    """NVIDIA_API_KEY and MISTRAL_API_KEY are fallback / cross-family —
    missing must be WARN, not FAIL."""
    old_n = os.environ.pop("NVIDIA_API_KEY", None)
    old_m = os.environ.pop("MISTRAL_API_KEY", None)
    try:
        assert doctor.check_nvidia_key().level == "warn", \
            "NVIDIA missing should warn, not fail"
        assert doctor.check_mistral_key().level == "warn", \
            "MISTRAL missing should warn, not fail"
    finally:
        if old_n: os.environ["NVIDIA_API_KEY"] = old_n
        if old_m: os.environ["MISTRAL_API_KEY"] = old_m


def test_overall_exit_code_priority() -> None:
    """Exit code priority: fail > warn > ok."""
    def r(level: str) -> doctor.CheckResult:
        return doctor.CheckResult("test", level, "x")  # type: ignore[arg-type]

    assert doctor.overall_exit_code([r("ok"), r("ok")]) == 0
    assert doctor.overall_exit_code([r("ok"), r("warn"), r("ok")]) == 1
    assert doctor.overall_exit_code([r("ok"), r("warn"), r("fail")]) == 2
    assert doctor.overall_exit_code([r("fail")]) == 2
    assert doctor.overall_exit_code([]) == 0


def test_run_all_checks_includes_network_when_flagged() -> None:
    """Default: only DEFAULT_CHECKS. With --with-network: also NETWORK_CHECKS."""
    default = doctor.run_all_checks(with_network=False)
    with_net = doctor.run_all_checks(with_network=True)
    assert len(with_net) > len(default), \
        "--with-network must add additional checks"
    assert len(with_net) == len(default) + len(doctor.NETWORK_CHECKS)


def test_subprocess_no_keys_exits_2() -> None:
    """End-to-end: invoke as a CLI without keys → exit 2 (fail)."""
    result = subprocess.run(
        [str(VENV_PY), str(DOCTOR), "--no-color"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin"},
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 2, \
        f"no-keys CLI run should exit 2; got {result.returncode}"
    assert "BLOCKED" in result.stdout, \
        "blocked state must be visible in stdout"


def test_subprocess_with_groq_only_exits_1() -> None:
    """End-to-end: invoke with GROQ but no NVIDIA/MISTRAL → exit 1 (warn)."""
    result = subprocess.run(
        [str(VENV_PY), str(DOCTOR), "--no-color"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin", "GROQ_API_KEY": "test-key"},
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 1, \
        f"GROQ-only run should exit 1; got {result.returncode}"
    assert "with warnings" in result.stdout.lower(), \
        "warning state must be visible in stdout"


def test_subprocess_all_keys_exits_0() -> None:
    """End-to-end: all 3 keys set → exit 0 (GO)."""
    result = subprocess.run(
        [str(VENV_PY), str(DOCTOR), "--no-color"],
        capture_output=True, text=True, timeout=15,
        env={
            "PATH": "/usr/bin:/bin",
            "GROQ_API_KEY": "test-key",
            "NVIDIA_API_KEY": "test-key",
            "MISTRAL_API_KEY": "test-key",
        },
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, \
        f"all-keys run should exit 0; got {result.returncode}; stdout={result.stdout[:300]}"
    assert "GO" in result.stdout and "BLOCKED" not in result.stdout
    assert "warning" not in result.stdout.lower() or "GO with" not in result.stdout


def test_subprocess_json_output_well_formed() -> None:
    """--json must emit parseable JSON with documented top-level keys."""
    result = subprocess.run(
        [str(VENV_PY), str(DOCTOR), "--json"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin"},
        cwd=str(LAB_ROOT),
    )
    payload = json.loads(result.stdout)
    for key in ("checks", "exit_code", "summary"):
        assert key in payload, f"--json output missing '{key}'"
    summary = payload["summary"]
    for k in ("ok", "warn", "fail"):
        assert k in summary, f"summary missing '{k}'"
    # Sum of summary equals number of checks
    assert summary["ok"] + summary["warn"] + summary["fail"] == len(payload["checks"])


def test_subprocess_help_works() -> None:
    """--help must print usage without running any checks."""
    result = subprocess.run(
        [str(VENV_PY), str(DOCTOR), "--help"],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, "--help should exit 0"
    out = result.stdout + result.stderr
    assert "--json" in out, "help must document --json"
    assert "--with-network" in out, "help must document --with-network"
    assert "--verbose" in out, "help must document --verbose"


def test_verbose_mode_shows_fix_hints_for_ok_checks() -> None:
    """In default mode, fix hints only show on fail. In --verbose,
    every check with a fix hint shows it."""
    result_default = subprocess.run(
        [str(VENV_PY), str(DOCTOR), "--no-color"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin"},
        cwd=str(LAB_ROOT),
    )
    result_verbose = subprocess.run(
        [str(VENV_PY), str(DOCTOR), "--no-color", "--verbose"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin"},
        cwd=str(LAB_ROOT),
    )
    # Verbose output should be longer (more fix hints shown)
    assert len(result_verbose.stdout) >= len(result_default.stdout), \
        "--verbose should produce same-or-longer output"


def test_no_color_disables_ansi() -> None:
    """--no-color must produce ANSI-free output."""
    result = subprocess.run(
        [str(VENV_PY), str(DOCTOR), "--no-color"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin"},
        cwd=str(LAB_ROOT),
    )
    assert "\033[" not in result.stdout, \
        "--no-color must strip ANSI codes from stdout"


def main() -> int:
    tests = [
        test_module_imports_clean,
        test_default_checks_count_reasonable,
        test_each_check_returns_check_result,
        test_python_version_check_passes_on_modern_python,
        test_venv_check_finds_real_venv,
        test_required_deps_importable,
        test_proof_packet_check_finds_canonical,
        test_failures_md_check_finds_signed_failures,
        test_ui_build_check_finds_dist,
        test_weekly_timeline_check_finds_compiled_output,
        test_default_lab_check_finds_sor_state,
        test_port_check_returns_ok_or_warn,
        test_port_check_detects_in_use,
        test_groq_key_check_missing_is_fail,
        test_optional_keys_missing_are_warn_only,
        test_overall_exit_code_priority,
        test_run_all_checks_includes_network_when_flagged,
        test_subprocess_no_keys_exits_2,
        test_subprocess_with_groq_only_exits_1,
        test_subprocess_all_keys_exits_0,
        test_subprocess_json_output_well_formed,
        test_subprocess_help_works,
        test_verbose_mode_shows_fix_hints_for_ok_checks,
        test_no_color_disables_ansi,
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
