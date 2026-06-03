"""bert doctor — pre-flight environment + readiness check.

Run before a stage demo, before installing on a partner machine, or as
part of CI. Catches the "broke before it started" failure mode that's
the single most embarrassing demo experience in a 2026 investor pitch.

Usage:
  .venv/bin/python tools/bert_doctor.py
  .venv/bin/python tools/bert_doctor.py --json           # machine output
  .venv/bin/python tools/bert_doctor.py --with-network   # ping providers
  .venv/bin/python tools/bert_doctor.py --verbose        # extra context

Exit codes:
  0  all checks GO (warnings are fine)
  1  one or more WARN, no FAIL
  2  one or more FAIL — demo blocked, run the suggested fixes
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

LAB_ROOT = Path(__file__).resolve().parent.parent
# Sprint 1 commit 11: make `core.*` importable when bert_doctor is run
# as a standalone script (the new host_detection + model_cards checks
# import from core).
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
VENV_DIR = LAB_ROOT / ".venv"
VENV_PY = VENV_DIR / "bin" / "python"
CANONICAL_PACKET = LAB_ROOT / "findings" / "proof_packets" / "cycle-0400.tar.gz"
UI_BUILD = LAB_ROOT / "bert" / "v4" / "dist" / "index.html"
WEEKLY_TIMELINE = LAB_ROOT / "findings" / "weekly_history" / "timeline.md"
DAILY_TIMELINE = LAB_ROOT / "findings" / "daily_history" / "timeline.md"
DEFAULT_LAB = LAB_ROOT / "lab"
DEFAULT_DEMO_PORT = 5174
MIN_PYTHON = (3, 11)
REQUIRED_DEPS = ("fastapi", "uvicorn", "httpx")  # core import-without-side-effects deps

Level = Literal["ok", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    level: Level
    message: str
    fix_hint: str | None = None
    details: dict = field(default_factory=dict)


# ── Individual checks ───────────────────────────────────────────────

def check_python_version() -> CheckResult:
    info = sys.version_info
    actual = f"{info.major}.{info.minor}.{info.micro}"
    if info >= MIN_PYTHON:
        return CheckResult("python version", "ok",
                           f"{actual} (≥{MIN_PYTHON[0]}.{MIN_PYTHON[1]} required)")
    return CheckResult("python version", "fail",
                       f"{actual} (need ≥{MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
                       fix_hint="install a newer Python, then recreate .venv")


def check_venv_exists() -> CheckResult:
    if VENV_PY.exists():
        return CheckResult(".venv", "ok", f"{VENV_PY.relative_to(LAB_ROOT)}")
    return CheckResult(".venv", "fail", "missing",
                       fix_hint=f"python3 -m venv {VENV_DIR.relative_to(LAB_ROOT)}")


def check_required_deps() -> CheckResult:
    missing: list[str] = []
    for mod in REQUIRED_DEPS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        return CheckResult("python deps", "ok",
                           f"{', '.join(REQUIRED_DEPS)} importable")
    return CheckResult("python deps", "fail",
                       f"missing: {', '.join(missing)}",
                       fix_hint=".venv/bin/pip install -r requirements.txt")


def check_groq_key() -> CheckResult:
    if os.environ.get("GROQ_API_KEY"):
        return CheckResult("GROQ_API_KEY", "ok", "set")
    # BB.4 — fall back to WARN if another provider key is available;
    # the routing fabric can still operate. Only FAIL when zero keys
    # are present (no live cycle is runnable at all).
    other_keys = [k for k in ("NVIDIA_API_KEY", "MISTRAL_API_KEY",
                              "OPENROUTER_API_KEY", "CEREBRAS_API_KEY",
                              "GOOGLE_AI_API_KEY", "GOOGLE_API_KEY")
                  if os.environ.get(k)]
    if other_keys:
        return CheckResult(
            "GROQ_API_KEY", "warn",
            f"not set — falling back to {', '.join(other_keys)}",
            fix_hint="export GROQ_API_KEY=gsk_... for fastest inference "
                     "(but the routing fabric will operate without it)"
        )
    return CheckResult(
        "GROQ_API_KEY", "fail",
        "not set in env — no provider keys available",
        fix_hint="export GROQ_API_KEY=gsk_... (free tier at console.groq.com)"
    )


def check_nvidia_key() -> CheckResult:
    if os.environ.get("NVIDIA_API_KEY"):
        return CheckResult("NVIDIA_API_KEY", "ok", "set (fallback ready)")
    return CheckResult("NVIDIA_API_KEY", "warn", "not set (optional fallback)",
                       fix_hint="export NVIDIA_API_KEY=nvapi-... if you want NVIDIA NIM fallback")


def check_mistral_key() -> CheckResult:
    if os.environ.get("MISTRAL_API_KEY"):
        return CheckResult("MISTRAL_API_KEY", "ok", "set (judge family ready)")
    return CheckResult("MISTRAL_API_KEY", "warn", "not set (optional judge family)",
                       fix_hint="export MISTRAL_API_KEY=... if you want Mistral cross-family judging")


def check_port_available(port: int = DEFAULT_DEMO_PORT) -> CheckResult:
    """Return ok if the port is bindable, warn if it's already in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
        return CheckResult(f"port {port}", "ok", "available")
    except OSError as e:
        return CheckResult(f"port {port}", "warn",
                           f"in use ({e.strerror})",
                           fix_hint=f"lsof -ti:{port} | xargs kill   OR   DEMO_PORT=5189 ./demo_run.sh")


def check_proof_packet() -> CheckResult:
    if CANONICAL_PACKET.exists():
        size_kb = CANONICAL_PACKET.stat().st_size // 1024
        return CheckResult("proof packet", "ok",
                           f"{CANONICAL_PACKET.relative_to(LAB_ROOT)} ({size_kb} KB)")
    return CheckResult("proof packet", "fail",
                       f"missing: {CANONICAL_PACKET.relative_to(LAB_ROOT)}",
                       fix_hint=".venv/bin/python -c 'from core import proof_packet; proof_packet.build_packet(cycle_id=400)'")


def check_ui_build() -> CheckResult:
    if UI_BUILD.exists():
        return CheckResult("UI build", "ok",
                           f"{UI_BUILD.relative_to(LAB_ROOT)}")
    return CheckResult("UI build", "fail",
                       f"missing: {UI_BUILD.relative_to(LAB_ROOT)}",
                       fix_hint="cd bert/v4 && npm install && npm run build")


def check_weekly_timeline() -> CheckResult:
    if WEEKLY_TIMELINE.exists():
        # Parse the JSON twin for the actual week count
        json_twin = WEEKLY_TIMELINE.with_suffix(".json")
        if json_twin.exists():
            try:
                data = json.loads(json_twin.read_text())
                n = data.get("weeks_recorded", 0)
                expected = data.get("expected_weeks", 8)
                msg = f"{n}/{expected} weeks recorded"
                level: Level = "ok" if n >= 1 else "warn"
                return CheckResult("weekly timeline", level, msg)
            except (OSError, json.JSONDecodeError):
                pass
        return CheckResult("weekly timeline", "ok", "compiled")
    return CheckResult("weekly timeline", "warn", "not compiled",
                       fix_hint=".venv/bin/python tools/weekly_history_compile.py")


def check_daily_timeline() -> CheckResult:
    if DAILY_TIMELINE.exists():
        json_twin = DAILY_TIMELINE.with_suffix(".json")
        if json_twin.exists():
            try:
                data = json.loads(json_twin.read_text())
                n = data.get("days_recorded", 0)
                expected = data.get("expected_days", 30)
                msg = f"{n}/{expected} days recorded"
                level: Level = "ok" if n >= 1 else "warn"
                return CheckResult("daily timeline", level, msg)
            except (OSError, json.JSONDecodeError):
                pass
        return CheckResult("daily timeline", "ok", "compiled")
    return CheckResult("daily timeline", "warn", "not compiled",
                       fix_hint=".venv/bin/python tools/daily_history_compile.py")


def check_bert_run_present() -> CheckResult:
    """W.4 — confirm the autonomous cycle runner is on disk + executable
    as a python module. Without bert_run.py, 'start the lab with a
    mission' has no implementation."""
    bert_run = LAB_ROOT / "tools" / "bert_run.py"
    if not bert_run.exists():
        return CheckResult("bert run", "fail",
                           "tools/bert_run.py missing",
                           fix_hint="reinstall or git-restore tools/bert_run.py")
    # The module should import without raising
    import subprocess
    result = subprocess.run(
        [str(VENV_PY) if (VENV_PY := LAB_ROOT / ".venv" / "bin" / "python").exists()
         else "python3",
         str(bert_run), "--dry-run", "--max-cycles", "0"],
        capture_output=True, text=True, timeout=5,
        cwd=str(LAB_ROOT),
    )
    # rc=2 when there's no seed (default lab) is fine — we only care
    # that the script ran without crashing during arg parsing / import.
    if result.returncode in (0, 2):
        return CheckResult("bert run", "ok",
                           "tools/bert_run.py importable + executable")
    return CheckResult("bert run", "fail",
                       f"bert_run.py crashed (rc={result.returncode}): {result.stderr[:120]}",
                       fix_hint="check Python path + .venv health")


def check_default_lab() -> CheckResult:
    sor = DEFAULT_LAB / "sor"
    state = DEFAULT_LAB / "state"
    if sor.exists() and state.exists():
        return CheckResult("default lab", "ok",
                           f"{DEFAULT_LAB.relative_to(LAB_ROOT)}/sor + state present")
    return CheckResult("default lab", "warn",
                       "incomplete (sor/ or state/ missing)",
                       fix_hint=".venv/bin/python tools/bert_init.py to scaffold")


def check_failures_md_in_packet() -> CheckResult:
    """The packet's failures.md must be present + signed. This is the
    honesty-discipline check — a packet without failures.md is suspect."""
    if not CANONICAL_PACKET.exists():
        return CheckResult("failures.md (packet)", "fail",
                           "can't check — packet missing",
                           fix_hint="see 'proof packet' check above")
    import tarfile
    try:
        with tarfile.open(CANONICAL_PACKET) as tf:
            names = tf.getnames()
        has_md = any(n.endswith("/failures.md") for n in names)
        has_sig = any(n.endswith("/failures.sigstore") for n in names)
        if has_md and has_sig:
            return CheckResult("failures.md (packet)", "ok",
                               "present + signed separately")
        if has_md:
            return CheckResult("failures.md (packet)", "warn",
                               "present but not separately signed",
                               fix_hint="rebuild packet with current proof_packet.build_packet()")
        return CheckResult("failures.md (packet)", "fail",
                           "missing — packet is suspect",
                           fix_hint="rebuild packet via core.proof_packet")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("failures.md (packet)", "fail",
                           f"tar read failed: {exc}",
                           fix_hint="packet corrupt — rebuild")


def check_groq_reachable() -> CheckResult:
    """Network check — only run with --with-network.

    AA.6 — Groq's edge (Cloudflare-fronted) rejects requests with the
    default `Python-urllib/X.Y` User-Agent as bot traffic, returning
    403 Forbidden. Sending an explicit UA mirroring curl gets through.
    Without this fix, the doctor was wrongly reporting "Groq unreachable"
    on a working key, which is itself a stage-safety failure.
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return CheckResult("Groq /v1/models", "warn",
                           "skipped (no GROQ_API_KEY)")
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {key}",
                # Mirror curl's UA — Groq's edge rejects Python-urllib/*.
                "User-Agent": "bert-doctor/1.0 (+https://bert.dev)",
            },
        )
        import time
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=5) as r:
            elapsed = (time.monotonic() - t0) * 1000
            if r.status == 200:
                return CheckResult("Groq /v1/models", "ok",
                                   f"reachable ({elapsed:.0f}ms)")
            return CheckResult("Groq /v1/models", "warn",
                               f"unexpected HTTP {r.status}")
    except urllib.error.HTTPError as exc:
        # Distinguish key-invalid (401) from network/edge issues (else)
        if exc.code == 401:
            return CheckResult("Groq /v1/models", "fail",
                               "401 Unauthorized — key invalid or revoked",
                               fix_hint="rotate the key at console.groq.com")
        if exc.code == 403:
            return CheckResult("Groq /v1/models", "warn",
                               "403 Forbidden — edge rejected request "
                               "(rare; usually means a User-Agent issue)")
        return CheckResult("Groq /v1/models", "warn",
                           f"HTTP {exc.code} {exc.reason}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Groq /v1/models", "fail",
                           f"unreachable: {type(exc).__name__}",
                           fix_hint="check internet + GROQ_API_KEY validity")


# ── Aggregator ──────────────────────────────────────────────────────

def check_credentials_mode() -> CheckResult:
    """Sprint 1 commit 5 (v1.0 spec S-6): credentials.json must be
    mode 600 (owner read/write only). Mode 644 leaks API keys to
    every other process on the machine."""
    cred_path = Path.home() / ".bert-lab" / "credentials.json"
    if not cred_path.exists():
        return CheckResult(
            "credentials.json mode", "warn",
            "no credentials.json found (BYO keys via env vars OK)",
            fix_hint="run `bert init` to create credentials.json",
        )
    try:
        st = cred_path.stat()
        # Extract POSIX permission bits (last 9 bits of mode)
        perms = st.st_mode & 0o777
        if perms == 0o600:
            return CheckResult(
                "credentials.json mode", "ok",
                "mode 600 (owner-only, as required)",
            )
        # 0o400 (read-only) is also fine for hardened setups
        if perms == 0o400:
            return CheckResult(
                "credentials.json mode", "ok",
                "mode 400 (owner read-only, hardened)",
            )
        return CheckResult(
            "credentials.json mode", "fail",
            f"mode {oct(perms)} — API keys readable by other processes",
            fix_hint=f"chmod 600 {cred_path}",
        )
    except OSError as e:
        return CheckResult(
            "credentials.json mode", "warn",
            f"stat failed: {e}",
        )


def check_host_detection() -> CheckResult:
    """Sprint 1 commit 11: detect MCP host context (Claude Code / Cursor /
    Codex / standalone) + report per-tier model availability."""
    try:
        from core import host_detector
        ctx = host_detector.detect()
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "host detection", "warn",
            f"detection raised: {type(e).__name__}: {e}",
        )
    host_detector.summarize(ctx)
    # Compress to one summary line for the table; full lines for verbose
    short = (
        f"host={ctx.host_name} "
        f"tier1_models={len(ctx.tier1_models_available)} "
        f"byo_keys={len(ctx.byo_keys_present)}"
    )
    return CheckResult(
        "host detection", "ok", short,
        fix_hint=(None if ctx.host_name != "standalone"
                  else "no MCP host detected; bert will use free-tier providers"),
    )


def check_model_cards_present() -> CheckResult:
    """Sprint 1 commit 9: model_cards.yaml must be loadable + have ≥10 cards."""
    try:
        from core import model_cards
        cards = model_cards.load_all()
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "model cards", "fail",
            f"load failed: {type(e).__name__}: {e}",
            fix_hint="check core/library/model_cards.yaml syntax",
        )
    if len(cards) < 10:
        return CheckResult(
            "model cards", "fail",
            f"only {len(cards)} cards loaded (need ≥10)",
            fix_hint="check core/library/model_cards.yaml for parse errors",
        )
    pending = model_cards.cards_with_pending_deprecation(within_days=7)
    if pending:
        return CheckResult(
            "model cards", "warn",
            f"{len(cards)} loaded, but {len(pending)} deprecating within 7 days "
            f"({[c.id for c in pending]})",
            fix_hint="update model_cards.yaml + retest routing",
        )
    return CheckResult("model cards", "ok", f"{len(cards)} loaded, no pending deprecations")


def check_lab_schema_parseable() -> CheckResult:
    """Sprint 1 commit 5: if the default lab has a lab_schema.json,
    verify it parses + matches the LabSchema shape. A corrupt schema
    would cause every cycle to re-classify (slow + flaky)."""
    schema_path = DEFAULT_LAB / "lab_schema.json"
    if not schema_path.exists():
        return CheckResult(
            "lab schema", "ok",
            "no persisted lab_schema.json yet (will synthesize on first run)",
        )
    try:
        data = json.loads(schema_path.read_text())
        required_fields = {"profile_id", "rule_id", "roster_initial", "workflow"}
        missing = required_fields - set(data.keys())
        if missing:
            return CheckResult(
                "lab schema", "fail",
                f"missing required fields: {sorted(missing)}",
                fix_hint=f"rm {schema_path} (will re-synthesize on next run)",
            )
        roster = data.get("roster_initial", [])
        if not roster:
            return CheckResult(
                "lab schema", "warn",
                f"roster_initial is empty (rule={data.get('rule_id')})",
            )
        return CheckResult(
            "lab schema", "ok",
            f"rule={data.get('rule_id')} roster_size={len(roster)}",
        )
    except (json.JSONDecodeError, OSError) as e:
        return CheckResult(
            "lab schema", "fail",
            f"corrupt: {e}",
            fix_hint=f"rm {schema_path} (will re-synthesize)",
        )


DEFAULT_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_python_version,
    check_venv_exists,
    check_required_deps,
    check_credentials_mode,      # Sprint 1 commit 5 — S-6 security check
    check_host_detection,        # Sprint 1 commit 11 — MCP host context
    check_model_cards_present,   # Sprint 1 commit 9 — model_cards.yaml registry
    check_groq_key,
    check_nvidia_key,
    check_mistral_key,
    check_port_available,
    check_proof_packet,
    check_ui_build,
    check_weekly_timeline,
    check_daily_timeline,
    check_bert_run_present,
    check_default_lab,
    check_lab_schema_parseable,  # Sprint 1 commit 5 — schema integrity
    check_failures_md_in_packet,
)
NETWORK_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_groq_reachable,
)


def run_all_checks(*, with_network: bool = False) -> list[CheckResult]:
    results = [check() for check in DEFAULT_CHECKS]
    if with_network:
        results.extend(check() for check in NETWORK_CHECKS)
    return results


def overall_exit_code(results: list[CheckResult]) -> int:
    if any(r.level == "fail" for r in results):
        return 2
    if any(r.level == "warn" for r in results):
        return 1
    return 0


# ── Rendering ───────────────────────────────────────────────────────

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

GLYPHS = {"ok": "✓", "warn": "⚠", "fail": "✗"}
COLORS = {"ok": GREEN, "warn": YELLOW, "fail": RED}


def render_text(results: list[CheckResult], *, verbose: bool = False,
                use_color: bool = True) -> str:
    def c(level: Level, s: str) -> str:
        if not use_color:
            return s
        return f"{COLORS[level]}{s}{RESET}"

    rc = overall_exit_code(results)
    lines = [f"{BOLD if use_color else ''}[bert doctor]{RESET if use_color else ''}"]
    name_width = max(len(r.name) for r in results) + 2
    for r in results:
        glyph = c(r.level, GLYPHS[r.level])
        name = r.name.ljust(name_width)
        lines.append(f"  {glyph}  {name}{r.message}")
        if verbose and r.fix_hint:
            lines.append(f"     {DIM if use_color else ''}↳ fix: {r.fix_hint}{RESET if use_color else ''}")
        elif r.level == "fail" and r.fix_hint:
            # Always surface fix hints for failures
            lines.append(f"     {DIM if use_color else ''}↳ fix: {r.fix_hint}{RESET if use_color else ''}")

    lines.append("")
    n_ok = sum(1 for r in results if r.level == "ok")
    n_warn = sum(1 for r in results if r.level == "warn")
    n_fail = sum(1 for r in results if r.level == "fail")
    summary = f"{n_ok} ok, {n_warn} warn, {n_fail} fail"
    if rc == 0:
        lines.append(c("ok", f"  Demo readiness: GO  ({summary})"))
    elif rc == 1:
        lines.append(c("warn", f"  Demo readiness: GO with warnings  ({summary})"))
    else:
        lines.append(c("fail", f"  Demo readiness: BLOCKED  ({summary})"))
    return "\n".join(lines)


def render_json(results: list[CheckResult]) -> str:
    return json.dumps({
        "checks": [asdict(r) for r in results],
        "exit_code": overall_exit_code(results),
        "summary": {
            "ok": sum(1 for r in results if r.level == "ok"),
            "warn": sum(1 for r in results if r.level == "warn"),
            "fail": sum(1 for r in results if r.level == "fail"),
        },
    }, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of human text.")
    ap.add_argument("--with-network", action="store_true",
                    help="Also ping providers (Groq /v1/models).")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show fix hints for every check, not just failures.")
    ap.add_argument("--no-color", action="store_true",
                    help="Disable ANSI color codes.")
    args = ap.parse_args()

    results = run_all_checks(with_network=args.with_network)
    if args.json:
        print(render_json(results))
    else:
        use_color = not args.no_color and sys.stdout.isatty()
        print(render_text(results, verbose=args.verbose, use_color=use_color))
    return overall_exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
