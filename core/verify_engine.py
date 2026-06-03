"""Python-native artifact verification.

Replaces the shell-based verification_command pattern (which f-string-
interpolated output_path into a bash command — shell injection vector
once features start parameterizing output paths in v1.0).

A VerificationSpec is a structured dict declaring what to check. The
engine runs the checks in Python, never shells out except for the
explicit `pytest_command` case (where a list-form argv prevents
injection).

Sprint 1 commit 2 (v1.0 — per the canonical spec section 11
"per-role verification in role template frontmatter").

Lifecycle:
  1. tools/bert_run.py:_build_spec attaches a `verification_spec` dict
     to the DispatchSpec (alongside the legacy `verification_command`
     shell string for backward compatibility)
  2. core/subagent.py:_run_verification checks for verification_spec
     first; falls back to shell-based verification_command otherwise
  3. core/verify_engine.verify_artifact() runs the structured checks
     and returns a VerifyResult

Per-role customization comes via role template frontmatter (Sprint 1
commit 3: role_registry.py reads each role's `verification:` field
from core/library/agents/*.md).
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger("bert.verify_engine")


@dataclass
class VerifyResult:
    """Outcome of running a VerificationSpec against an artifact."""
    ok: bool
    exit_code: int                    # 0 = pass, 1 = fail (mirrors shell convention)
    elapsed_ms: int
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def reason(self) -> str:
        """Human-readable summary of why the spec passed or failed."""
        if self.ok:
            return f"all {len(self.checks_passed)} checks passed"
        return f"{len(self.checks_failed)} check(s) failed: {'; '.join(self.checks_failed[:3])}"


# Default verification spec — equivalent to today's shell command in
# tools/bert_run.py:_build_spec, but Python-native.
DEFAULT_SPEC: dict[str, Any] = {
    "output_required": True,
    "min_chars": 1500,
    "required_headers": [
        {"level": 1, "count": 1},
        {"level": 2, "count": 3},
    ],
    "required_patterns": [
        {
            "description": "at least one citation (URL / arxiv / doi / github / Author-et-al)",
            "pattern": r"https?://|arxiv:|arXiv:|doi\.org|github\.com|[A-Z][a-z]+ et al",
        },
    ],
    "forbidden_patterns": [
        {
            "description": "no placeholder URLs (example.com / .org / .net)",
            "pattern": r"example\.(com|org|net)",
        },
        {
            "description": "no placeholder markers (TBD / XXX / placeholder)",
            "pattern": r"\bTBD\b|\bXXX\b|\bplaceholder\b",
        },
    ],
}


def verify_artifact(
    spec: dict[str, Any],
    output_path: Path,
    *,
    timeout_secs: float = 120.0,
) -> VerifyResult:
    """Run a VerificationSpec against the artifact at `output_path`.

    Args:
      spec: structured verification dict (see DEFAULT_SPEC for shape)
      output_path: absolute path to the artifact to verify
      timeout_secs: max wall-clock for the whole verification

    Returns a VerifyResult. Never raises on bad spec (logs + fails the
    relevant check); only raises on filesystem errors during read.
    """
    start = time.monotonic()
    deadline = start + timeout_secs
    passed: list[str] = []
    failed: list[str] = []

    # ── Existence check ───────────────────────────────────────────
    if spec.get("output_required", True):
        if not output_path.exists():
            failed.append(f"output_required: missing file {output_path}")
            return VerifyResult(
                ok=False, exit_code=1,
                elapsed_ms=int((time.monotonic() - start) * 1000),
                checks_passed=passed, checks_failed=failed,
            )
        if output_path.stat().st_size == 0:
            failed.append(f"output_required: empty file {output_path}")
            return VerifyResult(
                ok=False, exit_code=1,
                elapsed_ms=int((time.monotonic() - start) * 1000),
                checks_passed=passed, checks_failed=failed,
            )
        passed.append("output_required: file exists + non-empty")

    # Read content for textual checks
    content: str | None = None
    if output_path.exists():
        try:
            content = output_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            failed.append(f"read failed: {e}")
            return VerifyResult(
                ok=False, exit_code=1,
                elapsed_ms=int((time.monotonic() - start) * 1000),
                checks_passed=passed, checks_failed=failed,
            )

    # ── min_chars check ───────────────────────────────────────────
    min_chars = spec.get("min_chars")
    if min_chars is not None and content is not None:
        if len(content) < min_chars:
            failed.append(f"min_chars: {len(content)} < {min_chars}")
        else:
            passed.append(f"min_chars: {len(content)} >= {min_chars}")

    # ── required_headers check ───────────────────────────────────
    for hdr in spec.get("required_headers", []):
        if content is None:
            failed.append("required_headers: no content to check")
            break
        level = hdr["level"]
        need = hdr.get("count", 1)
        pattern = "^" + "#" * level + " "
        # Count lines matching the header marker
        matches = sum(1 for line in content.splitlines() if re.match(pattern, line))
        if matches < need:
            failed.append(
                f"required_headers: H{level} count {matches} < {need}"
            )
        else:
            passed.append(f"required_headers: H{level} >= {need}")

    # ── required_patterns check ──────────────────────────────────
    for req in spec.get("required_patterns", []):
        if content is None:
            failed.append(f"required_pattern: no content for {req['description']}")
            break
        pat = req["pattern"]
        desc = req.get("description", pat)
        if not re.search(pat, content):
            failed.append(f"required_pattern missing: {desc}")
        else:
            passed.append(f"required_pattern present: {desc}")

    # ── forbidden_patterns check ─────────────────────────────────
    for forb in spec.get("forbidden_patterns", []):
        if content is None:
            break
        pat = forb["pattern"]
        desc = forb.get("description", pat)
        m = re.search(pat, content)
        if m:
            failed.append(
                f"forbidden_pattern present: {desc} "
                f"(matched: {m.group(0)[:60]!r})"
            )
        else:
            passed.append(f"forbidden_pattern absent: {desc}")

    # ── pytest_command (for build / code missions) ───────────────
    pytest_cmd = spec.get("pytest_command")
    if pytest_cmd:
        result = _run_pytest(pytest_cmd, deadline)
        if result["ok"]:
            passed.append(f"pytest: {result['summary']}")
        else:
            failed.append(f"pytest: {result['summary']}")

    # ── gaps.md disclosure check (Sprint 1 commit 3) ─────────────
    # "Honest gap disclosure" is non-negotiable per memory constraint.
    # Every dispatch MUST write a {output_stem}_gaps.md alongside the
    # main artifact with ≥N bullets covering what couldn't be done.
    # Empty gaps.md is suspicious — high-stakes work always has gaps;
    # pretending otherwise is dishonest. Required by v1.0 launch
    # criterion #9 ("gaps.md non-empty signed separately").
    gaps_check = spec.get("gaps_required", {})
    if gaps_check.get("enabled", False):
        gaps_min_bullets = int(gaps_check.get("min_bullets", 3))
        gaps_path = output_path.parent / f"{output_path.stem}_gaps.md"
        if not gaps_path.exists():
            failed.append(
                f"gaps_required: missing {gaps_path.name} "
                f"(honest disclosure of acknowledged gaps required)"
            )
        else:
            try:
                gaps_content = gaps_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                failed.append(f"gaps_required: read failed: {e}")
                gaps_content = ""
            # Count markdown bullets (lines starting with "-" or "*")
            bullets = sum(
                1 for line in gaps_content.splitlines()
                if line.strip().startswith(("-", "*"))
            )
            if bullets < gaps_min_bullets:
                failed.append(
                    f"gaps_required: only {bullets} bullets in {gaps_path.name} "
                    f"(need ≥{gaps_min_bullets} — empty/sparse disclosure suspicious)"
                )
            else:
                passed.append(
                    f"gaps_required: {bullets} bullets disclosed in {gaps_path.name}"
                )

    # ── Final assembly ───────────────────────────────────────────
    elapsed = int((time.monotonic() - start) * 1000)
    ok = not failed
    return VerifyResult(
        ok=ok, exit_code=0 if ok else 1,
        elapsed_ms=elapsed,
        checks_passed=passed, checks_failed=failed,
    )


def _run_pytest(cmd: str | list[str], deadline: float) -> dict[str, Any]:
    """Run pytest in subprocess via list-argv (no shell injection).

    Accepts cmd as a string (split via shlex) or list. Times out if
    deadline reached.
    """
    import shlex
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {"ok": False, "summary": "deadline expired before pytest start"}

    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)

    try:
        r = subprocess.run(
            argv,
            capture_output=True, text=True,
            timeout=min(remaining, 120),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "summary": "pytest timed out"}
    except (FileNotFoundError, OSError) as e:
        return {"ok": False, "summary": f"pytest exec failed: {e}"}

    return {
        "ok": r.returncode == 0,
        "summary": (
            f"exit={r.returncode}" +
            (" passing" if r.returncode == 0 else " FAILING")
        ),
        "stdout_tail": r.stdout[-500:],
        "stderr_tail": r.stderr[-500:],
    }


def spec_from_role_template(role_template, output_path: str) -> dict[str, Any]:
    """Derive a verification spec from a role template's `verification:`
    frontmatter field.

    If the role template has no `verification:` field, returns DEFAULT_SPEC.

    Sprint 1 commit 3: role_registry.py will read role template
    frontmatter and pass the parsed verification dict here. For now
    this is a stub that always returns DEFAULT_SPEC.
    """
    # Future: read role_template.verification when role_registry lands
    return dict(DEFAULT_SPEC)
