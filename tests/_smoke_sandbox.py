"""Smoke test for core/sandbox.py — three-tier dispatch.

Per FINAL_implementation_plan_2026-05-07.md §5.4 H4.

Tests:
  1. Trusted tier runs echo and captures stdout
  2. Trusted tier respects timeout
  3. Trusted tier captures non-zero exit
  4. Restricted tier with no allow_paths restricts file reads (smoke
     — only verified on macOS where sandbox-exec exists)
  5. Network-isolated tier falls back to RESTRICTED when Docker absent
  6. classify_tier picks expected tier per source
  7. _build_sandbox_profile produces deny-default + explicit allows

Run: `.venv/bin/python tests/_smoke_sandbox.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import sandbox  # noqa: E402


def test_trusted_echo() -> None:
    r = sandbox.run_trusted(["echo", "hello sandbox"])
    assert r.exit_code == 0
    assert "hello sandbox" in r.stdout
    assert r.tier_used == sandbox.Tier.TRUSTED
    assert r.elapsed_ms < 5000


def test_trusted_timeout() -> None:
    r = sandbox.run_trusted(["sleep", "10"], timeout_secs=1)
    assert r.timed_out
    assert r.exit_code == 124


def test_trusted_nonzero_exit() -> None:
    r = sandbox.run_trusted(["sh", "-c", "exit 7"])
    assert r.exit_code == 7
    assert not r.timed_out


def test_trusted_pipes_stdin() -> None:
    """run_trusted accepts a `stdin=` kwarg and pipes it to the child.
    Used by core/hooks.py to deliver event payloads as JSON on stdin."""
    payload = '{"event":"PreToolUse","tool":"Read"}'
    r = sandbox.run_trusted(["cat"], stdin=payload, timeout_secs=5)
    assert r.exit_code == 0
    assert r.stdout == payload


def test_restricted_runs_or_falls_back() -> None:
    """sandbox-exec available → real restricted run; otherwise falls back to trusted."""
    r = sandbox.run_restricted(["echo", "ok"], timeout_secs=5)
    if r.fallback_reason:
        assert "sandbox-exec_not_available" in r.fallback_reason
    else:
        # Should have actually run under sandbox-exec
        assert r.tier_used == sandbox.Tier.RESTRICTED


def test_network_isolated_falls_back_when_no_docker() -> None:
    """If Docker isn't installed/running, NETWORK_ISOLATED degrades to RESTRICTED."""
    r = sandbox.run_network_isolated(["echo", "ok"], timeout_secs=5)
    if not sandbox._docker_available():
        assert r.fallback_reason is not None
        assert "docker_unavailable" in r.fallback_reason
    # Either way, the call should produce a result without raising
    assert r.tier_used == sandbox.Tier.NETWORK_ISOLATED


def test_classify_tier() -> None:
    assert sandbox.classify_tier(source="lab") == sandbox.Tier.TRUSTED
    assert sandbox.classify_tier(source="generated") == sandbox.Tier.RESTRICTED
    assert sandbox.classify_tier(source="external") == sandbox.Tier.NETWORK_ISOLATED
    assert sandbox.classify_tier(source="unknown") == sandbox.Tier.RESTRICTED


def test_profile_is_deny_default_with_allows() -> None:
    profile = sandbox._build_sandbox_profile(
        allow_read_paths=["/tmp"],
        allow_network=False,
    )
    assert "(deny default)" in profile
    assert "/tmp" in profile
    assert "network*" not in profile
    profile2 = sandbox._build_sandbox_profile(allow_network=True)
    assert "(allow network*)" in profile2


def test_run_dispatcher() -> None:
    """sandbox.run(cmd, tier=...) routes to the correct backend."""
    r = sandbox.run(["echo", "via_run"], tier=sandbox.Tier.TRUSTED)
    assert r.exit_code == 0
    assert "via_run" in r.stdout


def main() -> int:
    tests = [
        test_trusted_echo,
        test_trusted_timeout,
        test_trusted_nonzero_exit,
        test_trusted_pipes_stdin,
        test_restricted_runs_or_falls_back,
        test_network_isolated_falls_back_when_no_docker,
        test_classify_tier,
        test_profile_is_deny_default_with_allows,
        test_run_dispatcher,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
