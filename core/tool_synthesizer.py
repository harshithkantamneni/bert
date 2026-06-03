"""Sandboxed tool synthesis (Sprint 6 #30 — organic growth).

The lab can propose a NEW tool when a mission needs a capability the registry
lacks. AC: generated tools are sandboxed + reviewed before use. Pipeline:

  1. synthesize(spec)       — a cross-family LLM writes the tool source + a smoke
                              test from a structured spec (provider cascade).
  2. static_safety_scan()   — AST defense-in-depth: flag eval/exec/__import__,
                              dangerous imports, os.system. This is NOT the
                              primary control — it only surfaces foot-guns for PI
                              attention; the sandbox is the real containment, so
                              a conservative scan that over-flags is acceptable.
  3. sandbox_validate()     — run source+smoke inside core.sandbox (RESTRICTED by
                              default for generated code; Docker NETWORK_ISOLATED
                              for untrusted). The exit code is the real signal.
  4. propose()              — write the candidate to state/tools_pending_pi.md.
                              The tool is NOT registered or callable yet (mirrors
                              creator.propose_promotion for skills).
  5. install()              — ONLY after a PI blessing: write the source to
                              core/library/tools/<name>.py and register it in
                              core.tool_registry (signed, advisory).

Failure discipline (mirrors lineage/grader/creator): an LLM outage degrades to a
method="unavailable" candidate (empty source), never a crash or a fabricated
tool. Nothing reaches the registry without install(), which is PI-gated.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core import log

LOG = log.get_logger("bert.tool_synthesizer")

LAB_ROOT = Path(__file__).resolve().parent.parent
LIB_TOOLS_DIR = LAB_ROOT / "core" / "library" / "tools"
PROPOSALS_PATH = LAB_ROOT / "state" / "tools_pending_pi.md"
# Machine-readable sidecar dir: the markdown is for human review, but the
# generated source + schema must survive in a parseable form so activate()
# can install the approved tool (Sprint 7 — fixes the source-discard bug).
PENDING_DIR = LAB_ROOT / "state" / "tools_pending"

try:
    from core.grader import DEFAULT_CASCADE
except Exception:  # noqa: BLE001
    DEFAULT_CASCADE = [("groq", "llama-3.3-70b-versatile")]

# Tool names become both a filename (<name>.py) and a Python function name, so
# they MUST be strict snake_case identifiers — this blocks path traversal
# (../, /) and code-injection via the name. Enforced at every entry point.
_NAME_RE = re.compile(r"[a-z_][a-z0-9_]{0,63}\Z")


def is_valid_tool_name(name: str) -> bool:
    return isinstance(name, str) and bool(_NAME_RE.match(name))


# Defense-in-depth deny-list (the sandbox is the primary control).
_DANGEROUS_CALLS = frozenset({"eval", "exec", "compile", "__import__"})
_DANGEROUS_IMPORTS = frozenset({"subprocess", "socket", "ctypes", "pty", "shutil"})
_DANGEROUS_ATTRS = frozenset({"system", "popen", "spawn", "spawnl", "spawnv",
                              "execv", "execve", "fork", "remove", "rmdir", "unlink"})


@dataclass
class ToolSpec:
    name: str
    description: str
    params_schema: dict
    returns: str = ""
    implementation_hint: str = ""


@dataclass
class ScanResult:
    safe: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class SynthesisCandidate:
    spec: ToolSpec
    source: str
    smoke_test: str
    scan: ScanResult
    sandbox: dict | None = None      # SandboxResult-derived summary (or None)
    method: str = "llm-v1"           # "llm-v1" | "unavailable"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "params_schema": self.spec.params_schema,
            "source": self.source,
            "smoke_test": self.smoke_test,
            "scan": {"safe": self.scan.safe, "violations": self.scan.violations},
            "sandbox": self.sandbox,
            "method": self.method,
            "error": self.error,
        }


# ── static safety scan (pure AST) ────────────────────────────────────


def static_safety_scan(source: str) -> ScanResult:
    """AST walk flagging obvious foot-guns. Defense-in-depth only — the sandbox
    is the real containment, so over-flagging here is acceptable (it just routes
    the candidate to extra PI scrutiny; it does not gate the pipeline)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ScanResult(safe=False, violations=[f"syntax error: {e}"])
    violations: list[str] = []
    for node in ast.walk(tree):
        # bare dangerous builtins: eval(...), exec(...), __import__(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_CALLS:
                violations.append(f"dangerous call: {node.func.id}()")
        # attribute sinks: os.system(...), os.popen(...), shutil.rmtree, ...
        if isinstance(node, ast.Attribute) and node.attr in _DANGEROUS_ATTRS:
            base = node.value.id if isinstance(node.value, ast.Name) else "?"
            violations.append(f"dangerous attribute: {base}.{node.attr}")
        # dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DANGEROUS_IMPORTS:
                    violations.append(f"dangerous import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _DANGEROUS_IMPORTS:
                violations.append(f"dangerous import: from {node.module}")
    return ScanResult(safe=not violations, violations=violations)


# ── parse the generated payload (pure) ───────────────────────────────


def _parse_generated(obj: dict) -> tuple[str, str] | None:
    """Pull (source, smoke_test) from the judge's JSON. source is required and
    non-empty; smoke_test defaults to empty string."""
    src = obj.get("source")
    if not isinstance(src, str) or not src.strip():
        return None
    smoke = obj.get("smoke_test")
    smoke = smoke if isinstance(smoke, str) else ""
    return src, smoke


# ── synthesize (provider cascade) ────────────────────────────────────


def _llm_json(messages: list[dict],
              cascade: list[tuple[str, str | None]]) -> dict | None:
    from core import provider as _prov
    for prov_name, model in cascade:
        try:
            resp = _prov.call(prov_name, messages, model=model, max_tokens=2000,
                              temperature=0.1, response_format={"type": "json_object"},
                              timeout=45.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("tool_synth: lane %s raised: %s", prov_name, exc)
            continue
        if resp.finish_reason == "error" or resp.text.startswith("[bert]"):
            LOG.warning("tool_synth: lane %s failed: %s", prov_name, resp.text[:120])
            continue
        try:
            parsed = json.loads(resp.text)
        except (json.JSONDecodeError, TypeError):
            LOG.warning("tool_synth: lane %s unparseable", prov_name)
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


_SYNTH_SYS = (
    "You write a single self-contained Python tool function for an autonomous "
    "lab. The function MUST be named exactly as requested and take **kwargs, "
    "returning a JSON-serializable dict. Use only the Python standard library; "
    "no network, no subprocess, no filesystem writes. Also write a smoke test "
    "that calls the function and asserts a known result, printing 'ok' on "
    "success. Return ONLY JSON: {\"source\": \"<def ...>\", \"smoke_test\": "
    "\"<assert ...>\"}."
)


def synthesize(spec: ToolSpec, *,
               cascade: list[tuple[str, str | None]] | None = None) -> SynthesisCandidate:
    """Generate a tool source + smoke test from `spec` and scan it. Does NOT run
    the sandbox or register anything — call sandbox_validate() + propose() next."""
    if not is_valid_tool_name(spec.name):
        return SynthesisCandidate(
            spec=spec, source="", smoke_test="", scan=ScanResult(safe=False),
            method="unavailable",
            error=f"invalid tool name {spec.name!r} (must be snake_case identifier)")
    lanes = cascade if cascade is not None else DEFAULT_CASCADE
    obj = _llm_json([
        {"role": "system", "content": _SYNTH_SYS},
        {"role": "user", "content": (
            f"FUNCTION NAME: {spec.name}\nDESCRIPTION: {spec.description}\n"
            f"PARAMS SCHEMA: {json.dumps(spec.params_schema)}\n"
            f"RETURNS: {spec.returns}\nHINT: {spec.implementation_hint}\n\n"
            f"Write the tool + smoke test. JSON only.")},
    ], lanes)
    if obj is None:
        return SynthesisCandidate(
            spec=spec, source="", smoke_test="", scan=ScanResult(safe=False),
            method="unavailable",
            error="tool synthesis failed (all provider lanes)")
    parsed = _parse_generated(obj)
    if parsed is None:
        return SynthesisCandidate(
            spec=spec, source="", smoke_test="", scan=ScanResult(safe=False),
            method="unavailable",
            error="generated payload missing a non-empty source")
    source, smoke = parsed
    return SynthesisCandidate(spec=spec, source=source, smoke_test=smoke,
                              scan=static_safety_scan(source), method="llm-v1")


# ── sandbox validation (real containment) ────────────────────────────


def sandbox_validate(source: str, smoke_test: str, *, name: str,
                     tier=None, work_dir: Path | None = None,
                     timeout_secs: int = 30):
    """Run source + smoke_test together inside core.sandbox. Returns the
    SandboxResult — exit_code 0 means the generated tool passed its own smoke.

    Default tier classifies `generated` source -> RESTRICTED (no home dir / net).
    Tests pin TRUSTED for speed + portability; production passes the default."""
    from core import sandbox
    if not is_valid_tool_name(name):
        raise ValueError(f"invalid tool name {name!r} (must be snake_case identifier)")
    if tier is None:
        tier = sandbox.classify_tier(source="generated")
    wd = Path(work_dir) if work_dir is not None else LAB_ROOT / "state" / "tool_synth_tmp"
    wd.mkdir(parents=True, exist_ok=True)
    script = wd / f"{name}_candidate.py"
    # Defense in depth: the validated name can't traverse, but verify anyway.
    if not script.resolve().is_relative_to(wd.resolve()):
        raise ValueError(f"resolved script path escapes work dir: {script}")
    script.write_text(source + "\n\n# ── generated smoke test ──\n" + smoke_test)
    return sandbox.run(["python3", str(script)], tier=tier,
                       timeout_secs=timeout_secs, cwd=wd)


# ── review gate (PI-blessed, mirrors creator.propose_promotion) ──────


def propose(candidate: SynthesisCandidate, *,
            proposals_path: Path | None = None,
            pending_dir: Path | None = None) -> str:
    """Append a PI-approval-pending entry for a synthesized tool. Returns the
    proposal id. The tool is NOT registered — install() does that after blessing.

    Writes TWO artifacts: the human-review markdown (`proposals_path`) AND a
    machine-readable sidecar `<pending_dir>/<proposal_id>.json` carrying the full
    candidate (incl. source + params_schema) so activate() can install the
    approved tool. Paths resolve at call time so the defaults stay patchable."""
    if proposals_path is None:
        proposals_path = PROPOSALS_PATH
    if pending_dir is None:
        pending_dir = PENDING_DIR
    spec = candidate.spec
    digest = hashlib.sha256(candidate.source.encode()).hexdigest()[:10]
    proposal_id = f"tool-{spec.name}-{digest}"
    pending_dir.mkdir(parents=True, exist_ok=True)
    (pending_dir / f"{proposal_id}.json").write_text(
        json.dumps({"proposal_id": proposal_id, **candidate.to_dict()}, indent=2))
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    scan = candidate.scan
    sandbox_line = "(not run)"
    if candidate.sandbox is not None:
        sandbox_line = (f"exit={candidate.sandbox.get('exit_code')} "
                        f"tier={candidate.sandbox.get('tier_used')}")
    entry = (
        f"\n## {proposal_id}\n"
        f"\n- **tool:** {spec.name}\n"
        f"- **description:** {spec.description}\n"
        f"- **proposed_at:** {datetime.now(UTC).isoformat()}\n"
        f"- **method:** {candidate.method}\n"
        f"- **scan_safe:** {scan.safe}"
        + (f" (violations: {'; '.join(scan.violations)})" if scan.violations else "")
        + "\n"
        f"- **sandbox:** {sandbox_line}\n"
        f"- **status:** pending\n"
        f"- **approve_command:** `/approve {proposal_id}` or POST /api/approve/{proposal_id}\n"
        f"\n```python\n{spec.name}\n# source recorded; review before /approve\n```\n"
    )
    with proposals_path.open("a") as f:
        f.write(entry)
    LOG.info("tool_synth: proposed %s (scan_safe=%s)", proposal_id, scan.safe)
    return proposal_id


# ── install (post-blessing only) ─────────────────────────────────────


def install(name: str, source: str, description: str, params_schema: dict, *,
            lib_dir: Path = LIB_TOOLS_DIR):
    """Write a blessed tool's source to the lib dir and register it in the tool
    registry. Caller is responsible for verifying the PI blessing first (same
    contract as creator.promote). Signs the manifest (advisory).

    The blessed source is exec'd in an isolated namespace to obtain the handler
    callable — this code path runs ONLY on PI-approved, sandbox-validated source,
    never on raw model output."""
    from core import tool_registry
    from core.types import PermissionMode
    if not is_valid_tool_name(name):
        raise ValueError(f"invalid tool name {name!r} (must be snake_case identifier)")
    lib_dir.mkdir(parents=True, exist_ok=True)
    path = lib_dir / f"{name}.py"
    path.write_text(source)
    ns: dict = {}
    exec(compile(source, str(path), "exec"), ns)  # noqa: S102  # nosec B102 — runs ONLY on PI-blessed, sandbox-validated source, never raw model output
    handler = ns.get(name)
    if not callable(handler):
        raise ValueError(f"installed source defines no callable named {name!r}")
    tool_registry.register_function(
        name=name, description=description, parameters_schema=params_schema,
        handler=handler, permission_mode=PermissionMode.DEFAULT, source="creator")
    try:
        from core import signing
        sig = signing.sign_skill_manifest(path)
        signing.append_to_local_rekor(sig)
    except Exception as e:  # noqa: BLE001
        LOG.warning("tool_synth: signing failed (advisory): %s", e)
    LOG.info("tool_synth: installed + registered %s at %s", name, path)
    return path


# ── activation (PI approved a proposal) ──────────────────────────────


def read_pending(proposal_id: str, *, pending_dir: Path | None = None) -> dict | None:
    """Load the machine-readable candidate sidecar for a proposal id, or None."""
    if pending_dir is None:
        pending_dir = PENDING_DIR
    sidecar = pending_dir / f"{proposal_id}.json"
    # Containment: a traversing proposal id must not read outside pending_dir.
    try:
        if not sidecar.resolve().is_relative_to(pending_dir.resolve()):
            LOG.warning("tool_synth: rejected traversing proposal id %r", proposal_id)
            return None
    except (OSError, ValueError):
        return None
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def activate(proposal_id: str, *, pending_dir: Path | None = None,
             lib_dir: Path = LIB_TOOLS_DIR) -> dict:
    """Install an approved synthesized tool from its sidecar. Idempotent: if the
    tool is already registered, returns {ok, already}. Caller (proposal_activate)
    is responsible for confirming the PI blessing first."""
    from core import tool_registry
    rec = read_pending(proposal_id, pending_dir=pending_dir)
    if rec is None:
        return {"ok": False, "error": f"no pending tool record for {proposal_id!r}"}
    name = rec.get("name", "")
    if not is_valid_tool_name(name):
        return {"ok": False, "error": f"invalid tool name in sidecar: {name!r}"}
    source = rec.get("source", "")
    if not isinstance(source, str) or not source.strip():
        return {"ok": False, "error": f"missing/empty source in sidecar for {proposal_id!r}"}
    if tool_registry.get(name) is not None:
        return {"ok": True, "name": name, "already": True}
    path = install(name, source, rec.get("description", ""),
                   rec.get("params_schema") or {"type": "object"}, lib_dir=lib_dir)
    return {"ok": True, "name": name, "path": str(path)}
