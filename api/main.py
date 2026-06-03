"""Bert v4 API server.

FastAPI on 127.0.0.1:5174. The vite dev server (5173) proxies /api/*
to here.

Read endpoints expose bert's event stream + roster + diagnostics.
Write endpoints record PI intent into lab/state/*.jsonl so bert's
orchestrator can drain them at the next cycle start.

Run with:
    .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 5174 --reload
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

LAB_ROOT = Path(__file__).resolve().parent.parent
# L.4 — BERT_LAB_PATH env var lets the API server view a lab OTHER than
# bert-lab's own (e.g. a scaffolded user lab at ~/.bert/labs/X/). Default
# is bert-lab's lab/ which keeps current behavior intact.
import contextlib
import os as _os

_lab_override = _os.environ.get("BERT_LAB_PATH")
LAB_PATH = Path(_lab_override).expanduser().resolve() if _lab_override else LAB_ROOT / "lab"
EVENTS_PATH = LAB_PATH / "sor" / "events.jsonl"
STATE_DIR = LAB_PATH / "state"


# N.2 — per-request lab routing. Endpoints that read events/state may
# accept ?lab=<name> to point at ~/.bert/labs/<name>/ instead of the
# default LAB_PATH. This lets the bert UI switch labs without a
# uvicorn restart.
def _resolve_lab_path(lab: str | None) -> tuple[Path, Path, Path]:
    """Return (lab_path, events_path, state_dir) for the given lab name.

    If `lab` is None or empty, return the default (BERT_LAB_PATH or
    bert-lab/lab/). Otherwise look up ~/.bert/labs/<lab>/. Raises 404
    if the named scaffolded lab is missing the bert-lab structure.
    """
    if not lab:
        return LAB_PATH, EVENTS_PATH, STATE_DIR
    home_labs = Path(_os.path.expanduser("~/.bert/labs"))
    cand = home_labs / lab
    if not cand.exists():
        raise HTTPException(404, f"lab {lab!r} not found at ~/.bert/labs/{lab}")
    cand_events = cand / "sor" / "events.jsonl"
    if not cand_events.exists():
        raise HTTPException(404, f"lab {lab!r} missing sor/events.jsonl")
    return cand, cand_events, cand / "state"
# GG-A-prep — module-level globals retained as DEFAULT (supervisor lab)
# for backwards compatibility. Endpoints accepting `lab:` should call
# `_state_paths(lab)` instead so writes scope to the user's chosen
# lab. Pre-GG these were the only state paths; multi-lab writes went
# to the supervisor lab's state regardless of which lab the user
# thought they were operating on (the "pause the wrong lab" bug).
PI_ACTIONS = STATE_DIR / "pi_actions.jsonl"
BLESSINGS = STATE_DIR / "blessings.jsonl"
VETOES = STATE_DIR / "vetoes.jsonl"
STEERS = STATE_DIR / "steers.jsonl"
ASKS = STATE_DIR / "asks.jsonl"
PI_OVERRIDES = STATE_DIR / "pi_overrides.json"
PAUSED_FLAG = STATE_DIR / "paused"
NOTES_DIR = STATE_DIR / "notes"
APPROVALS_DIR = STATE_DIR / "approvals"
LETTERS_DIR = STATE_DIR / "director_letters"


class StatePaths:
    """GG-A-prep — per-lab state paths.

    Built from a resolved lab_path so every endpoint can write to
    `<lab_path>/state/*` instead of the module-global supervisor
    state. The same file/dir names are kept so any tool that
    inspects `state/` on disk continues to work; only the parent
    directory changes per lab.
    """
    __slots__ = ("lab_path", "state_dir", "pi_actions", "blessings",
                  "vetoes", "steers", "asks", "pi_overrides",
                  "paused_flag", "notes_dir", "approvals_dir",
                  "letters_dir", "voice_steers_dir", "dev_pending")

    def __init__(self, lab_path: Path):
        self.lab_path = lab_path
        self.state_dir = lab_path / "state"
        self.pi_actions = self.state_dir / "pi_actions.jsonl"
        self.blessings = self.state_dir / "blessings.jsonl"
        self.vetoes = self.state_dir / "vetoes.jsonl"
        self.steers = self.state_dir / "steers.jsonl"
        self.asks = self.state_dir / "asks.jsonl"
        self.pi_overrides = self.state_dir / "pi_overrides.json"
        self.paused_flag = self.state_dir / "paused"
        self.notes_dir = self.state_dir / "notes"
        self.approvals_dir = self.state_dir / "approvals"
        self.letters_dir = self.state_dir / "director_letters"
        self.voice_steers_dir = self.state_dir / "voice_steers"
        self.dev_pending = self.state_dir / "dev_pending.jsonl"


def _state_paths(lab: str | None = None) -> StatePaths:
    """Resolve the per-lab StatePaths bundle. `lab=None` → repo's
    own lab/ (supervisor). Raises HTTPException(404) via
    _resolve_lab_path if the user-named lab doesn't exist."""
    lab_path, _events, _state_dir = _resolve_lab_path(lab)
    return StatePaths(lab_path)

app = FastAPI(
    title="bert v4 api",
    description="read-write surface for bert · for Dominus",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Sleep-time compute startup hook (G.2) ────────────────────────────


@app.on_event("startup")
def _start_idle_loop() -> None:
    """Start the background pre-warm thread when the FastAPI app boots.

    Disabled when BERT_DISABLE_IDLE_COMPUTE=1 (for tests). The loop is
    a daemon thread; uvicorn shutdown kills it automatically.
    """
    import os as _os
    if _os.environ.get("BERT_DISABLE_IDLE_COMPUTE"):
        return
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    try:
        from core import idle_compute
        idle_compute.start_idle_loop(interval_secs=30, stale_secs=90,
                                      deep_every=10)
    except Exception:  # noqa: BLE001
        # observability is best-effort; if it can't start, the lab
        # continues without pre-warming.
        pass


_SUPERVISOR_TASK: asyncio.Task | None = None  # held ref — see _reap_orphan_runs


@app.on_event("startup")
def _reap_orphan_runs() -> None:
    """Sprint 4 B — on boot, mark any run-cycle subprocess whose pid is gone
    as orphaned (criterion 21: async runner survives parent crash). Then start
    an asyncio supervisor task that re-reaps every 30s for runs that die later.
    """
    try:
        import logging

        from core import run_registry
        reaped = run_registry.reap_orphans()
        if reaped:
            logging.getLogger("bert.api").info(
                "run_registry: reaped %d orphan run(s) on startup: %s",
                len(reaped), reaped)

        async def _supervise() -> None:
            while True:
                await asyncio.sleep(30)
                try:
                    run_registry.reap_orphans()
                except Exception:  # noqa: BLE001
                    pass
        # Hold a module-level reference — asyncio keeps only a WEAK ref to
        # tasks, so a fire-and-forget create_task() can be GC'd mid-flight and
        # silently stop the periodic reaping. (recheck 2026-05-28)
        global _SUPERVISOR_TASK
        _SUPERVISOR_TASK = asyncio.create_task(_supervise())
    except Exception:  # noqa: BLE001
        pass


# ── helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    seed = f"{prefix}|{time.time_ns()}"
    # SHA1 as deterministic id (collision-resistance is plenty for the
    # short prefix-space we use). usedforsecurity=False mutes B324.
    return f"{prefix}_{hashlib.sha1(seed.encode(), usedforsecurity=False).hexdigest()[:12]}"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_overrides(paths: StatePaths | None = None) -> dict[str, list[str]]:
    """GG-A-prep — `paths=None` reads the supervisor's overrides
    (backwards compat). Pass `_state_paths(lab)` for per-lab reads."""
    p = (paths.pi_overrides if paths else PI_OVERRIDES)
    if not p.exists():
        return {"pinned": [], "suppressed": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {
            "pinned": list(data.get("pinned", []) or []),
            "suppressed": list(data.get("suppressed", []) or []),
        }
    except (OSError, json.JSONDecodeError):
        return {"pinned": [], "suppressed": []}


def _save_overrides(overrides: dict[str, list[str]],
                     paths: StatePaths | None = None) -> None:
    """GG-A-prep — `paths=None` writes to supervisor's overrides."""
    p = (paths.pi_overrides if paths else PI_OVERRIDES)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(overrides, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _read_events(since_ts: str | None = None, limit: int = 200,
                 events_path: Path | None = None) -> list[dict[str, Any]]:
    """Read events from `events_path` (or default EVENTS_PATH)."""
    path = events_path or EVENTS_PATH
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_ts is not None and ev.get("ts", "") <= since_ts:
                continue
            events.append(ev)
    return events[-limit:]


def _read_event_by_id(event_id: str,
                      events_path: Path | None = None) -> dict[str, Any] | None:
    """Look up an event by id. Searches the events stream (per-lab when
    `events_path` provided; otherwise the default EVENTS_PATH), then
    falls back to dev_pending fixtures so bless / veto / ask / note
    work transparently on preview decisions too."""
    path = events_path or EVENTS_PATH
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("id") == event_id:
                    return ev
    # Fallback: dev fixtures
    for ev in _read_jsonl(STATE_DIR / "dev_pending.jsonl"):
        if ev.get("id") == event_id:
            return ev
    return None


def _count_lines(p: Path) -> int:
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ── pydantic request models ──────────────────────────────────────────


class BlessRequest(BaseModel):
    rationale: str | None = None


class VetoRequest(BaseModel):
    reason: str | None = None


class AskRequest(BaseModel):
    target_id: str
    question: str = Field(..., min_length=1, max_length=2000)


class SteerRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    modality: Literal["typed", "whisper"] = "typed"


class PauseRequest(BaseModel):
    reason: str | None = None


class ApproveRequest(BaseModel):
    choice: str = Field(..., min_length=1, max_length=64)
    rationale: str | None = None


class NoteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class CredentialTestRequest(BaseModel):
    provider: str = Field(..., min_length=1, max_length=64)
    key: str = Field(..., min_length=1, max_length=512)


class SaveCredentialsRequest(BaseModel):
    """GG-A.0.2 — Persist validated provider keys to
    ~/.bert-lab/credentials.json (mode 600).

    The frontend tests each key via /api/onboarding/test-credential
    before calling this endpoint, but this endpoint re-validates
    every key against its provider before persisting — never trust
    the frontend's claim that a key passed validation.
    """
    credentials: dict[str, str] = Field(
        ..., description=(
            "Map of env_var → key. Keys are env var names "
            "(GROQ_API_KEY, NVIDIA_API_KEY, etc.); values are the "
            "API key strings. Only validated keys are persisted; "
            "the response reports which were saved and which failed."
        ),
    )


class FirstMissionRequest(BaseModel):
    mission: str = Field(..., min_length=1, max_length=4000)


class CreateLabRequest(BaseModel):
    """GG-A.1 — Create a customer lab via the UI.

    `name` is the human-facing slug (will be lowercased + space→_
    by bert_init). `mission` is the seed_brief.md content.
    `archetype` picks the template (research/product/strategy).
    `focus_areas` is optional; when omitted, the archetype's default
    set is used (research → methodology/evidence/synthesis/...).
    """
    name: str = Field(..., min_length=2, max_length=40,
                       pattern=r"^[A-Za-z][A-Za-z0-9 _-]{1,39}$")
    mission: str = Field(..., min_length=10, max_length=8000)
    archetype: str = Field("research", pattern=r"^(research|product|strategy)$")
    focus_areas: list[str] | None = Field(
        default=None, description=(
            "3-7 lab-specific focus areas. When omitted, the archetype's "
            "default set is used. The string 'unspecified' is auto-"
            "appended if missing."
        ),
    )
    provider: str = Field("groq", max_length=40)
    autonomy: str = Field("collaborator", pattern=r"^(assistant|collaborator|pilot)$")


# ── Onboarding (F.2) ─────────────────────────────────────────────────


@app.post("/api/onboarding/test-credential")
def test_credential(req: CredentialTestRequest) -> dict[str, Any]:
    """Validate a provider API key by hitting its /models endpoint.

    Used by the onboarding wizard's "test" button. Returns ok=True if
    the provider responds 200. Does NOT persist the credential —
    that's a separate flow (PI writes to ~/.bert-lab/credentials.json
    out-of-band or via a future /api/onboarding/save-credential
    endpoint that's permission-gated).
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    try:
        from core import provider as _prov
        spec = _prov.PROVIDERS.get(req.provider.lower())
        if spec is None:
            return {"ok": False, "reason": f"unknown provider: {req.provider}"}
        # Probe with the supplied key.
        import httpx
        url = f"{spec.base_url}/models"
        headers = {"Authorization": f"Bearer {req.key}"}
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(url, headers=headers)
            ok = 200 <= r.status_code < 300
            return {"ok": ok, "status_code": r.status_code,
                    "reason": "" if ok else r.text[:120]}
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            return {"ok": False, "reason": str(e)[:160]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": str(e)[:160]}


@app.post("/api/onboarding/save-credentials")
def save_credentials(req: SaveCredentialsRequest) -> dict[str, Any]:
    """GG-A.0.2 — Persist validated provider keys to
    ~/.bert-lab/credentials.json (mode 600).

    Re-validates every supplied key against its provider's /models
    endpoint before persisting. Keys that fail validation are NOT
    written. Keys already in credentials.json are preserved if the
    new request omits them (additive merge).

    Returns: per-key status + count saved/skipped, plus the canonical
    path so the UI can surface where keys land. Never echoes back
    the key values themselves.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    from core import config as _cfg
    from core import provider as _prov

    # Map env_var → provider_id for the validation probe (the
    # test-credential endpoint takes provider_id, not env_var).
    ENV_TO_PROVIDER = {
        "NVIDIA_API_KEY":      "nvidia",
        "GROQ_API_KEY":        "groq",
        "CEREBRAS_API_KEY":    "cerebras",
        "MISTRAL_API_KEY":     "mistral",
        "GOOGLE_AI_API_KEY":   "gemini",
        "GOOGLE_API_KEY":      "gemini",
        "OPENROUTER_API_KEY":  "openrouter",
        "HF_TOKEN":            "huggingface",
    }

    import httpx
    per_key: dict[str, dict[str, Any]] = {}
    validated: dict[str, str] = {}
    for env_var, key in req.credentials.items():
        env_var_upper = env_var.upper().strip()
        if not key or not key.strip():
            per_key[env_var_upper] = {"saved": False, "reason": "empty key"}
            continue
        provider_id = ENV_TO_PROVIDER.get(env_var_upper)
        if not provider_id:
            per_key[env_var_upper] = {"saved": False,
                                       "reason": f"unknown env var {env_var_upper!r}"}
            continue
        spec = _prov.PROVIDERS.get(provider_id)
        if spec is None:
            per_key[env_var_upper] = {"saved": False,
                                       "reason": f"unknown provider {provider_id!r}"}
            continue
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(f"{spec.base_url}/models",
                                headers={"Authorization": f"Bearer {key.strip()}"})
            if 200 <= r.status_code < 300:
                validated[env_var_upper] = key.strip()
                per_key[env_var_upper] = {"saved": True, "status_code": r.status_code}
            else:
                per_key[env_var_upper] = {
                    "saved": False,
                    "status_code": r.status_code,
                    "reason": (r.text or "")[:120],
                }
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            per_key[env_var_upper] = {"saved": False,
                                       "reason": f"network: {str(e)[:120]}"}
        except Exception as e:  # noqa: BLE001
            per_key[env_var_upper] = {"saved": False,
                                       "reason": str(e)[:160]}

    # Additive merge with existing credentials.json
    cred_path = _cfg.CRED_PATH
    cred_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing: dict[str, str] = {}
    if cred_path.exists():
        try:
            existing = json.loads(cred_path.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
    merged = {**existing, **validated}

    # Write with mode 600 — owner-only read/write.
    cred_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    with contextlib.suppress(OSError):
        cred_path.chmod(0o600)

    # Bust the config cache so the next config.load() picks up the
    # newly-written keys without a restart.
    _cfg._cached = None

    return {
        "ok": True,
        "path": str(cred_path),
        "saved_count": sum(1 for r in per_key.values() if r.get("saved")),
        "skipped_count": sum(1 for r in per_key.values() if not r.get("saved")),
        "per_key": per_key,
        "total_credentials_after_save": len(merged),
    }


@app.get("/api/onboarding/credentials-status")
def credentials_status() -> dict[str, Any]:
    """GG-A.0.2 — Report which credential env vars are currently
    available (from ~/.bert-lab/credentials.json + env vars). Used by
    the UI to decide whether to redirect to /onboard or show the
    normal FirstLight surface.

    Never echoes key values — only env-var names + boolean presence.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    from core import config as _cfg
    cfg = _cfg.load(reload=True)
    present = sorted(k for k, v in cfg.credentials.items() if v)
    return {
        "present": present,
        "count": len(present),
        "has_any_provider": any(
            k in cfg.credentials and cfg.credentials[k]
            for k in ("NVIDIA_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY",
                       "MISTRAL_API_KEY", "GOOGLE_AI_API_KEY",
                       "OPENROUTER_API_KEY", "HF_TOKEN")
        ),
    }


@app.post("/api/labs")
def create_lab(req: CreateLabRequest) -> dict[str, Any]:
    """GG-A.1 — Scaffold a new customer lab from the UI.

    Thin wrapper over `tools.bert_init._scaffold_lab(answers,
    user_provided_seed=True)`. Returns the new lab's path + the
    URL-safe name (the path slug bert_run.py and /api/run-cycle
    accept via `?lab=`). Conflicts (lab dir already exists with
    content) → 409.

    Honest disclosure: this also writes a FF-A-aware lab.yaml with
    role:standard + share_with_supervisor:true + archetype-aware
    focus_areas. The supervisor lab (role:supervisor) is created
    by hand, NOT through this endpoint — there's only one
    supervisor per engine.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    from tools import bert_init as _bi

    # Slugify the same way _scaffold_lab will, so we can detect
    # conflicts before mutating the filesystem.
    slug = req.name.replace(" ", "_").lower()
    target = _bi.LABS_DIR / slug
    if target.exists() and any(target.iterdir()):
        raise HTTPException(409,
            f"lab {slug!r} already exists at {target}; pick a "
            f"different name or delete the existing dir.")

    answers = {
        "name": req.name,
        "archetype": req.archetype.capitalize(),  # _scaffold_lab lowercases
        "provider": req.provider,
        "autonomy": req.autonomy.capitalize(),
        "seed": req.mission,
        "focus_areas": req.focus_areas,  # passed through to lab.yaml
    }
    # No template — `_scaffold_lab` writes a minimal seed_brief.md
    # from answers["seed"] when neither from_template nor a prior
    # seed_brief.md exists.
    try:
        lab_dir = _bi._scaffold_lab(answers, user_provided_seed=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"scaffold failed: {e}") from e

    return {
        "ok": True,
        "name": slug,
        "path": str(lab_dir),
        "archetype": req.archetype,
        "mission_preview": req.mission[:200],
    }


@app.post("/api/onboarding/first-mission")
def first_mission(req: FirstMissionRequest) -> dict[str, Any]:
    """Write the onboarding wizard's first-mission text to
    `memories/mission.md`.

    Format: a brief header + the supplied mission text. If
    mission.md already exists, this appends a "Previously" section
    rather than overwriting (preserves any prior mission state).
    """
    mission_path = LAB_ROOT / "memories" / "mission.md"
    mission_path.parent.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    if mission_path.exists():
        prior = mission_path.read_text()
        body = (
            f"# Mission\n\n*Set via onboarding wizard at {ts}.*\n\n"
            f"{req.mission.strip()}\n\n"
            f"---\n\n## Previously\n\n{prior}\n"
        )
    else:
        body = f"# Mission\n\n*Set via onboarding wizard at {ts}.*\n\n{req.mission.strip()}\n"
    mission_path.write_text(body)
    return {"ok": True, "path": "memories/mission.md", "ts": ts}


# ── Memory tier budget (H.2) ─────────────────────────────────────────


@app.get("/api/memory-tiers")
def memory_tier_status() -> dict[str, Any]:
    """Core-tier budget compliance + tier distribution.

    Production rubric: core tier ≤ 2K tokens. Above that, attention-
    dilution degrades long-context retrieval (MemoryAgentBench ICLR
    2026). This endpoint shows the headroom + overflow count.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import memory_tiers
        raw = memory_tiers.core_budget_status()
        # Y.1 — translate Python-side field names to the names the
        # TypeScript MemoryTierStatus interface (api/client.ts) expects.
        # Without this mapping the Diagnostics surface crashes with
        # 'Cannot read properties of undefined' on .toLocaleString().
        core_budget = {
            "used_tokens": raw.get("token_total_unenforced", 0),
            "budget_tokens": raw.get("token_budget", 2000),
            "headroom_tokens": raw.get("headroom_tokens", 0),
            "overflow_emitted": bool(raw.get("overflow_items", 0)),
        }
        return {
            "ts": _now_iso(),
            "core_budget": core_budget,
            "tier_distribution": memory_tiers.stats(),
        }
    except Exception as e:
        raise HTTPException(500, f"memory_tiers stats unavailable: {e}") from e


# ── MCP replay protection (H.1) ──────────────────────────────────────


@app.get("/api/mcp-replay")
def mcp_replay_stats() -> dict[str, Any]:
    """Active nonce count + per-tool replay-protection telemetry.

    Used by the Diagnostics surface to confirm replay protection is
    seeing traffic + zero rejected duplicates means either no replay
    attacks OR no nonces being sent (both are PI-visible from this
    endpoint's by_tool breakdown vs /api/events tool_call volume).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import mcp_replay
        return {
            "ts": _now_iso(),
            **mcp_replay.stats(),
        }
    except Exception as e:
        raise HTTPException(500, f"mcp_replay stats unavailable: {e}") from e


# ── Signing + local Rekor (G.4) ──────────────────────────────────────


@app.get("/.well-known/agent.json.sig")
def agent_card_signature() -> dict[str, Any]:
    """Signed attestation over /.well-known/agent.json.

    External verifiers:
      1. GET /.well-known/agent.json → canonical JSON
      2. GET /.well-known/agent.json.sig → this response
      3. Verify the signature with the embedded pubkey_pem
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import signing
        # Pass lab=None explicitly: agent_card's default is FastAPI's
        # Query(None) sentinel, which (a) isn't a str and (b) isn't JSON-
        # serializable — calling agent_card() bare leaks it into the card
        # dict and signing then fails. This is an internal call, not HTTP.
        card_resp = agent_card(lab=None)
        sig = signing.sign_agent_card(card_resp)
        signing.append_to_local_rekor(sig)
        return sig.to_dict()
    except Exception as e:
        raise HTTPException(500, f"signing unavailable: {e}") from e


@app.get("/api/signing/local-rekor")
def local_rekor_tail(limit: int = Query(50, ge=1, le=500),
                      artifact_kind: str | None = Query(None)) -> dict[str, Any]:
    """Tail the local Rekor-shaped append-only attestation log."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import signing
        entries = signing.read_local_rekor(
            limit=limit, artifact_kind=artifact_kind,
        )
        return {
            "ts": _now_iso(),
            "count": len(entries),
            "entries": entries,
        }
    except Exception as e:
        raise HTTPException(500, f"local_rekor unavailable: {e}") from e


@app.post("/api/signing/checkpoint-merkle")
def signing_checkpoint() -> dict[str, Any]:
    """Compute the events.jsonl Merkle root, sign it, append to local
    Rekor. Intended to run at backup time (nightly cron)."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import merkle, signing
        events_path = LAB_PATH / "sor" / "events.jsonl"
        if not events_path.exists():
            return {"ok": False, "reason": "events.jsonl not found"}
        root_hex = merkle.file_root_hex(events_path)
        line_count = sum(1 for _ in events_path.open("rb"))
        sig = signing.sign_merkle_root(
            root_hex,
            events_path=str(events_path.relative_to(LAB_ROOT)),
            line_count=line_count,
        )
        log_id = signing.append_to_local_rekor(sig)
        return {
            "ok": True,
            "log_id": log_id,
            "merkle_root_hex": root_hex,
            "line_count": line_count,
            "ts": sig.ts,
        }
    except Exception as e:
        raise HTTPException(500, f"checkpoint failed: {e}") from e


# ── Sleep-time compute (G.2) ─────────────────────────────────────────


@app.get("/api/idle-compute")
def idle_compute_stats() -> dict[str, Any]:
    """Per-day idle-compute pass count + duration histogram.

    Surfaces telemetry for the sleep-time-compute layer. The pre-warm
    thread (started at FastAPI app boot) runs whenever bert is idle.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import idle_compute as _ic
        return {
            "ts": _now_iso(),
            "stats": _ic.idle_stats(),
            "is_idle_now": _ic.is_idle(),
        }
    except Exception as e:
        raise HTTPException(500, f"idle_compute stats unavailable: {e}") from e


# ── Semantic cache (F.12) ────────────────────────────────────────────


@app.get("/api/semantic-cache")
def semantic_cache_stats() -> dict[str, Any]:
    """Per-role semantic-cache hit rates over the last 24h.

    Used by the bert Diagnostics surface to show whether the
    semantic cache is actually saving dispatches vs serving stale
    answers — load-bearing telemetry for the role-discipline gate.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import semantic_cache as _sc
        stats = _sc.cache_stats()
        return {
            "ts": _now_iso(),
            "roles": [
                {
                    "role": s.role,
                    "rows": s.rows,
                    "hits_24h": s.hits_24h,
                    "misses_24h": s.misses_24h,
                    "hit_rate": s.hit_rate,
                    "avg_similarity_on_hit": s.avg_similarity_on_hit,
                }
                for s in stats
            ],
            "cacheable_roles": sorted(_sc.CACHEABLE_ROLES),
        }
    except Exception as e:
        raise HTTPException(500, f"semantic_cache stats unavailable: {e}") from e


# ── A2A external interop (E.5 / F.9) ─────────────────────────────────


class A2ASendRequest(BaseModel):
    """A2A v0.1 task envelope (subset bert supports)."""
    task: dict[str, Any] = Field(default_factory=dict)
    message: dict[str, Any] = Field(default_factory=dict)
    skill_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.post("/a2a/v0/tasks/send")
def a2a_send(req: A2ASendRequest) -> dict[str, Any]:
    """A2A v0.1 task endpoint.

    Per amendment §A2: A2A is a wire-format surface on top of bert's
    L-09 MCP layer. This endpoint:
      1. Receives an A2A task envelope.
      2. Looks up the skill (= MCP server) by skill_id.
      3. Dispatches to the corresponding bert-* MCP server via
         core.mcp_installer.spawn().
      4. Returns the task lifecycle response.

    For now we return a stub "accepted" response that includes the
    matched skill_id; full task lifecycle (working/completed/failed
    states + streaming responses) lands when bert has its first
    real cross-agent dispatch need. The Agent Card already
    advertises bert's skills; this endpoint completes the round-trip.
    """
    skill_id = req.skill_id
    if not skill_id:
        raise HTTPException(400, "skill_id required")
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import mcp_installer
        spec = mcp_installer.load_spec(skill_id)
        if spec is None:
            raise HTTPException(404, f"skill {skill_id!r} not configured")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"installer error: {e}") from e
    # Stub task lifecycle: accept + echo back; full dispatch on
    # external-agent first contact (see amendment §A2).
    return {
        "task_id": _new_id("a2a"),
        "skill_id": skill_id,
        "state": "accepted",
        "ts": _now_iso(),
        "note": "bert-lab v0.1 A2A acceptance — dispatch deferred to "
                "first external-agent contact per amendment §A2",
        "message_echo": req.message,
    }


@app.get("/a2a/v0/tasks/{task_id}")
def a2a_task_status(task_id: str) -> dict[str, Any]:
    """A2A task status. Stub: returns 'completed' for any task id —
    actual lifecycle tracking lands when first external dispatch fires."""
    return {
        "task_id": task_id,
        "state": "completed",
        "ts": _now_iso(),
        "note": "bert-lab stub: no persistence layer yet for A2A tasks",
    }


@app.get("/.well-known/agent.json")
def agent_card(lab: str | None = Query(None)) -> dict[str, Any]:
    """A2A Agent Card per a2a-protocol.org/latest/specification.

    Discovery endpoint that lets external A2A-speaking agents enumerate
    bert's capabilities. Per FINAL plan amendment §A2: A2A activation
    requires no new harness code — it's a config + Agent Card surface
    on top of the L-09 MCP layer (which is the actual capability
    plumbing).

    N.4 — when ?lab=<name> is passed, the card reflects the routed
    lab's identity (name, archetype, cycle count) instead of the
    default bert-lab metadata. External agents that discover labs
    via /api/labs can then request a per-lab agent card.
    """
    try:
        from core import mcp_installer
        configured = mcp_installer.list_configured()
    except Exception:
        configured = []
    skills = [
        {
            "id": skill,
            "name": skill,
            "description": (
                "MCP server registered in bert's mcp_installer "
                "(state/mcp_servers.json)"
            ),
        }
        for skill in configured
    ]
    # N.4 — surface the active lab's identity if routed
    lab_name = "bert-lab"
    lab_description = (
        "Autonomous R&D-to-production lab. Operates a Quaker-style "
        "discernment pipeline (threshing → clearness → seasoning) "
        "across 8 free-tier LLM providers with cross-family "
        "adversarial review (P-VS-02)."
    )
    # isinstance guard: agent_card() is also called INTERNALLY by
    # agent_card_signature(), where `lab` is FastAPI's unresolved
    # Query(None) sentinel (truthy, but not a str). Without this guard
    # `home_labs / lab` raises TypeError → the .sig endpoint 500s.
    if lab and isinstance(lab, str):
        # Look up the scaffolded lab's metadata
        home_labs = Path(_os.path.expanduser("~/.bert/labs"))
        cand = home_labs / lab
        if cand.exists() and (cand / "lab.yaml").exists():
            lab_name = f"bert · {lab}"
            archetype = "unknown"
            template_origin = None
            try:
                for line in (cand / "lab.yaml").read_text().splitlines():
                    line = line.strip()
                    if line.startswith("archetype:"):
                        archetype = line.split(":", 1)[1].strip()
                    elif line.startswith("template_origin:"):
                        template_origin = line.split(":", 1)[1].strip()
            except OSError:
                pass
            cycle_count = (
                len(list((cand / "cycles").glob("*")))
                if (cand / "cycles").exists() else 0
            )
            lab_description = (
                f"Scaffolded {archetype} lab '{lab}'"
                + (f" from template {template_origin}" if template_origin else "")
                + f". {cycle_count} cycles on disk."
            )
        else:
            raise HTTPException(404, f"lab {lab!r} not found")
    card = {
        "name": lab_name,
        "description": lab_description,
        "url": "http://127.0.0.1:5174",
        "version": "0.1",
        "protocolVersion": "0.1",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "skills": skills,
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "documentation": "https://github.com/example/bert-lab",
    }
    if lab:
        card["lab"] = lab
    # G.3 — merge AGNTCY extensions (identity / observability / SLIM /
    # governance) so AGNTCY-speaking agents discover bert's full surface.
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import agntcy
        card.update(agntcy.agent_card_agntcy_extensions(skills=skills))
    except Exception:
        pass
    return card


@app.get("/.well-known/agntcy-directory.json")
def agntcy_directory() -> dict[str, Any]:
    """AGNTCY directory entry for bert. PI submits this to an AGNTCY
    directory (or self-hosts via docs.agntcy.org tooling) so other
    AGNTCY-speaking agents can discover bert without knowing the URL.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    from core import agntcy
    return agntcy.agntcy_directory_entry()


@app.post("/a2a/v0/observability/event")
def agntcy_observability_in(payload: dict[str, Any]) -> dict[str, Any]:
    """Inbound AGNTCY observability event from another agent. Stores
    the event in state/observability/agntcy_event.jsonl for the
    canvas + downstream consumers."""
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    from core import agntcy
    event_class = payload.get("event_class", "external")
    correlation_id = payload.get("correlation_id")
    agntcy.emit_agntcy_event(
        event_class, correlation_id=correlation_id,
        payload={k: v for k, v in payload.items()
                  if k not in ("event_class", "correlation_id")},
    )
    return {"ok": True, "received_at": _now_iso()}


# ── READ endpoints ───────────────────────────────────────────────────


@app.get("/api/status")
def status(lab: str | None = Query(None)) -> dict[str, Any]:
    # N.2 — per-request lab routing
    _lab_path, events_path, state_dir = _resolve_lab_path(lab)
    paused_flag = state_dir / "paused"
    if not events_path.exists():
        return {"ok": False, "reason": "events.jsonl not found"}
    stat = events_path.stat()
    last_ts = None
    cycle_current = None
    with events_path.open("rb") as f:
        f.seek(max(0, stat.st_size - 64 * 1024))
        tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.split("\n")):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                last_ts = ev.get("ts")
                cycle_current = ev.get("cycle")
                break
            except json.JSONDecodeError:
                continue
    return {
        "ok": True,
        "ts": _now_iso(),
        "lab": lab or "(default)",
        "last_event_ts": last_ts,
        "cycle_current": cycle_current,
        "events_total": _count_lines(events_path),
        "paused": paused_flag.exists(),
    }


@app.get("/api/events")
def list_events(
    since: str | None = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    lab: str | None = Query(None),
) -> dict[str, Any]:
    # N.2 — per-request lab routing
    _lab_path, events_path, _state_dir = _resolve_lab_path(lab)
    events = _read_events(since_ts=since, limit=limit, events_path=events_path)
    return {"count": len(events), "events": events, "lab": lab or "(default)"}


@app.get("/api/events/stream")
async def events_stream() -> StreamingResponse:
    return StreamingResponse(_sse_generator(), media_type="text/event-stream")


async def _sse_generator() -> AsyncIterator[str]:
    position = EVENTS_PATH.stat().st_size if EVENTS_PATH.exists() else 0
    yield ":connected\n\n"
    last_tick = time.monotonic()
    while True:
        new_events: list[dict[str, Any]] = []
        if EVENTS_PATH.exists():
            stat = EVENTS_PATH.stat()
            if stat.st_size > position:
                with EVENTS_PATH.open("rb") as f:
                    f.seek(position)
                    chunk = f.read().decode("utf-8", errors="replace")
                    position = stat.st_size
                    for line in chunk.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            new_events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            elif stat.st_size < position:
                position = stat.st_size
        for ev in new_events:
            yield f"data: {json.dumps({'type': 'event', 'event': ev})}\n\n"
        now = time.monotonic()
        if now - last_tick >= 5.0:
            last_tick = now
            yield f"data: {json.dumps({'type': 'tick', 'ts': _now_iso()})}\n\n"
        await asyncio.sleep(0.5)


@app.get("/api/events/{event_id}")
def get_event(event_id: str) -> dict[str, Any]:
    ev = _read_event_by_id(event_id)
    if ev is None:
        raise HTTPException(404, f"event {event_id!r} not found")
    return ev


@app.get("/api/agents")
def list_agents(lab: str | None = Query(None)) -> dict[str, Any]:
    # N.2 — per-request lab routing
    _lab_path, events_path, _state_dir = _resolve_lab_path(lab)
    events = _read_events(limit=10_000, events_path=events_path)
    counter: Counter[str] = Counter()
    last_ts: dict[str, str] = {}
    for ev in events:
        a = ev.get("agent")
        if a:
            counter[a] += 1
            ts = ev.get("ts", "")
            if ts > last_ts.get(a, ""):
                last_ts[a] = ts
    return {
        "count": len(counter),
        "lab": lab or "(default)",
        "agents": [
            {"agent": a, "count": n, "last_ts": last_ts.get(a)}
            for a, n in counter.most_common()
        ],
    }


@app.get("/api/choreography")
def get_choreography(bucket_minutes: int = Query(30, ge=5, le=240), buckets: int = Query(48, ge=8, le=288)) -> dict[str, Any]:
    """Agent roster as a playbill: each agent with their total events,
    last appearance, and an activity sparkline binned into `buckets`
    intervals of `bucket_minutes` each, anchored at that agent's most
    recent event (not wall-clock now). Anchoring lets the rhythm read
    even when the lab has been idle for days."""
    from collections import defaultdict

    events = _read_events(limit=20_000)
    per_agent_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        a = ev.get("agent")
        if not a:
            continue
        per_agent_events[a].append(ev)

    agents_out: list[dict[str, Any]] = []
    bucket_seconds = bucket_minutes * 60
    window_seconds = buckets * bucket_seconds
    for name, evs in per_agent_events.items():
        # Most recent timestamp anchors the window
        timestamps = [_ts_sort_key(e.get("ts")) for e in evs]
        timestamps = [t for t in timestamps if t > 0]
        if not timestamps:
            continue
        last = max(timestamps)
        window_start = last - window_seconds
        rhythm = [0] * buckets
        for t in timestamps:
            if t < window_start:
                continue
            idx = int((t - window_start) / bucket_seconds)
            if 0 <= idx < buckets:
                rhythm[idx] += 1
        agents_out.append({
            "name": name,
            "count": len(evs),
            "last_ts": evs[-1].get("ts") if evs else None,
            "rhythm": rhythm,
            "rhythm_anchor_ts": evs[-1].get("ts") if evs else None,
            "bucket_minutes": bucket_minutes,
        })
    agents_out.sort(key=lambda a: -a["count"])
    return {"ts": _now_iso(), "buckets": buckets, "bucket_minutes": bucket_minutes, "agents": agents_out}


@app.get("/api/quota")
def quota_stats(provider: str | None = Query(None)) -> dict[str, Any]:
    """Per-provider quota rollup from `core/quota.py:stats()`. Powers the
    bert Diagnostics surface (URGENT U-AC-Q3). 1ms read against the
    quota.db SQLite log."""
    try:
        import sys
        sys.path.insert(0, str(LAB_ROOT))
        from core import quota as _quota  # type: ignore[import-not-found]

        return {
            "ts": _now_iso(),
            "providers": _quota.stats(provider),
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"quota stats unavailable: {e}") from e


@app.get("/api/diagnostics")
def diagnostics() -> dict[str, Any]:
    events = _read_events(limit=2000)
    providers: Counter[str] = Counter()
    cost_today_usd = 0.0
    today_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
    for ev in events:
        provider = ev.get("gen_ai.system") or ev.get("judge_provider")
        if provider:
            providers[provider] += 1
        cost = ev.get("gen_ai.usage.cost_usd")
        if isinstance(cost, (int, float)) and ev.get("ts", "").startswith(today_prefix):
            cost_today_usd += float(cost)
    return {
        "ts": _now_iso(),
        "cost_today_usd": round(cost_today_usd, 4),
        "providers": dict(providers.most_common()),
    }


@app.get("/api/pending")
def list_pending(include_dev: bool = Query(False),
                  lab: str | None = Query(None)) -> dict[str, Any]:
    """Decisions escalated to Dominus. Rare by design — most cycles
    produce zero pending decisions. Dev fixtures live in
    <lab_path>/state/dev_pending.jsonl and are excluded unless
    include_dev=true so they don't surface in First Light's needs-you
    block."""
    paths = _state_paths(lab)
    blessed_ids = {b["decision_id"] for b in _read_jsonl(paths.blessings)}
    vetoed_ids = {v["decision_id"] for v in _read_jsonl(paths.vetoes)}
    pending: list[dict[str, Any]] = []
    for ev in _read_events(limit=10_000):
        if ev.get("bless_status") == "pending":
            eid = ev.get("id")
            if eid not in blessed_ids and eid not in vetoed_ids:
                pending.append(ev)
    if include_dev:
        for ev in _read_jsonl(paths.dev_pending):
            eid = ev.get("id")
            if eid and eid not in blessed_ids and eid not in vetoed_ids:
                ev["is_dev"] = True
                pending.append(ev)
    return {"count": len(pending), "pending": pending}


@app.post("/api/dev/seed-decision")
def dev_seed_decision() -> dict[str, Any]:
    """Idempotent: write a small set of fixture decisions to
    lab/state/dev_pending.jsonl for use by /meeting in preview mode.
    Excluded from /api/pending unless include_dev=true."""
    fixture_path = STATE_DIR / "dev_pending.jsonl"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh ids per seed so repeat-blessings during preview don't leave
    # the room permanently empty.
    fixtures = [
        {
            "id": _new_id("dec"),
            "ts": _now_iso(),
            "event_class": "decision",
            "agent": "researcher",
            "cycle": 412,
            "bless_status": "pending",
            "content": (
                "## Promote the cross-family judge ensemble to the default verdict path?\n\n"
                "The matter: bert's verdicts have been single-family for the last 18 cycles. "
                "A trial of cross-family judging (anthropic + openai + google in panel) ran "
                "twice this cycle — both passes converged on safety verdicts, neither converged "
                "on capability verdicts. Cost is +$0.04 per verdict."
            ),
            "voices": [
                {"agent": "researcher", "stance": "concur", "note": "evidence is intact across both passes; cross-family convergence on safety is the signal we wanted."},
                {"agent": "strategist", "stance": "concur", "note": "concur with caveat — set a cost ceiling at $0.06 per call."},
                {"agent": "falsifier", "stance": "object", "note": "objects on cost trajectory if the panel scales to four families. asks for a hard cap."},
                {"agent": "evaluator", "stance": "stand_aside", "note": "the cost is within budget but the panel is structurally novel. stand aside, no block."},
            ],
            "lineage": [],
            "tags": ["matter", "verdict-path"],
            "verdict": None,
            "severity_grade": "medium",
            "source_path": "fixtures/dev",
        },
        {
            "id": _new_id("dec"),
            "ts": _now_iso(),
            "event_class": "decision",
            "agent": "strategist",
            "cycle": 412,
            "bless_status": "pending",
            "content": (
                "## Cap subagent recursion depth at four, with explicit director override?\n\n"
                "The matter: cycle 407 produced a subagent recursion that reached depth seven "
                "before the orchestrator intervened. The implementer team proposes a hard cap at "
                "four. Falsifier asks whether this is the right number or merely a number."
            ),
            "voices": [
                {"agent": "implementer", "stance": "concur", "note": "four covers every observed legitimate case; deeper recursions in our trace were all loops."},
                {"agent": "falsifier", "stance": "object", "note": "four is a number, not a principle. propose: cap by reasoning-depth, not subagent-count."},
                {"agent": "strategist", "stance": "stand_aside", "note": "either path is defensible. let dominus decide which discipline this cap teaches."},
            ],
            "lineage": [],
            "tags": ["matter", "safety", "orchestration"],
            "verdict": None,
            "severity_grade": "high",
            "source_path": "fixtures/dev",
        },
    ]
    with fixture_path.open("w", encoding="utf-8") as f:
        for fx in fixtures:
            f.write(json.dumps(fx, separators=(",", ":")) + "\n")
    return {"ok": True, "count": len(fixtures), "decisions": fixtures}


@app.post("/api/dev/clear-decisions")
def dev_clear_decisions() -> dict[str, Any]:
    fixture_path = STATE_DIR / "dev_pending.jsonl"
    if fixture_path.exists():
        fixture_path.unlink()
    return {"ok": True}


@app.get("/api/approvals")
def list_approvals(include_dev: bool = Query(False),
                    lab: str | None = Query(None)) -> dict[str, Any]:
    """Interrupt-driven approval requests from bert.
    Dev-fixture approvals are filtered out unless include_dev=true so
    First Light doesn't surface a "needs you" block for fake data."""
    paths = _state_paths(lab)
    if not paths.approvals_dir.exists():
        return {"count": 0, "approvals": []}
    out: list[dict[str, Any]] = []
    for p in sorted(paths.approvals_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status", "pending") != "pending":
            continue
        if data.get("is_dev") and not include_dev:
            continue
        out.append(data)
    return {"count": len(out), "approvals": out}


@app.get("/api/overrides")
def get_overrides(lab: str | None = Query(None)) -> dict[str, list[str]]:
    return _load_overrides(_state_paths(lab))


@app.get("/api/notes/{event_id}")
def get_note(event_id: str,
              lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    path = paths.notes_dir / f"{event_id}.md"
    if not path.exists():
        return {"event_id": event_id, "text": None}
    return {"event_id": event_id, "text": path.read_text(encoding="utf-8")}


@app.get("/api/asks/{target_id}")
def list_asks_for(target_id: str,
                    lab: str | None = Query(None)) -> dict[str, Any]:
    """Q&A history for a specific event."""
    paths = _state_paths(lab)
    history = [a for a in _read_jsonl(paths.asks) if a.get("target") == target_id]
    return {"count": len(history), "asks": history}


@app.get("/api/loom")
def get_loom(min_count: int = Query(1, ge=1)) -> dict[str, Any]:
    """Citation threads — group editorial findings by what they cite.
    Each thread is a source (an arXiv id, a file path, a memory) and
    the findings that drew from it. Threads sort by citation count
    desc, then by most-recent citation."""
    from collections import defaultdict

    threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in _read_events(limit=10_000):
        if ev.get("event_class") != "finding":
            continue
        if not _is_editorial(ev):
            continue
        lineage = ev.get("lineage") or []
        if not lineage:
            continue
        title = _extract_title(ev.get("content") or "")
        for src in lineage:
            if not isinstance(src, str) or not src.strip():
                continue
            threads[src.strip()].append({
                "id": ev.get("id"),
                "ts": ev.get("ts"),
                "agent": ev.get("agent"),
                "cycle": ev.get("cycle"),
                "title": title,
            })

    out: list[dict[str, Any]] = []
    for src, cites in threads.items():
        cites.sort(key=lambda c: c.get("ts") or "")
        out.append({
            "source": src,
            "count": len(cites),
            "first_ts": cites[0]["ts"] if cites else None,
            "last_ts": cites[-1]["ts"] if cites else None,
            "kind": _classify_source(src),
            "citations": cites,
        })
    out = [t for t in out if t["count"] >= min_count]
    out.sort(key=lambda t: (-t["count"], -(_ts_sort_key(t["last_ts"]))))
    return {"count": len(out), "threads": out}


def _classify_source(src: str) -> str:
    s = src.lower()
    if s.startswith("arxiv") or "arxiv.org" in s:
        return "paper"
    if s.startswith("http://") or s.startswith("https://"):
        return "web"
    if s.endswith(".md") or "/findings/" in s or "/memories/" in s:
        return "note"
    if "/schemas/" in s or s.endswith(".json"):
        return "schema"
    if "/logs/" in s or s.endswith(".jsonl"):
        return "log"
    if s.startswith("d-n:") or s.startswith("d-"):
        return "decision"
    return "other"


def _extract_title(content: str) -> str:
    """Pick a useful title line out of a finding's content.
    Skips:
      - blank lines
      - italic stamp lines (e.g. `_Generated YYYY-MM-DD via tools/foo.py_`)
    Prefers, in order: the first `#`-heading; the first plain prose
    line after a stamp; the first non-stamp line. Falls back to
    `(untitled)` only when there is genuinely nothing to show.
    """
    def clean(s: str) -> str:
        return s.replace("**", "").replace("`", "").strip("_*").strip()

    saw_heading = False
    second_choice: str | None = None
    for line in content.split("\n"):
        l = line.strip()
        if not l:
            continue
        # Italic single-line stamp — skip
        if (
            (l.startswith("_") and (l.endswith("_") or l.endswith("_.")))
            and " " in l
        ):
            continue
        # Markdown heading — best title
        if l.startswith("#"):
            saw_heading = True
            return clean(l.lstrip("#")) or "(untitled)"
        # Otherwise remember the first prose line and continue,
        # in case a later heading shows up
        cleaned = clean(l)
        if not cleaned:
            continue
        if second_choice is None:
            second_choice = cleaned[:120]
    if saw_heading:
        return "(untitled)"
    return second_choice or "(untitled)"


def _ts_sort_key(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


@app.get("/api/findings")
def list_findings(
    limit: int = Query(60, ge=1, le=500),
    offset: int = Query(0, ge=0),
    lab: str | None = Query(None),
) -> dict[str, Any]:
    """Editorial findings sorted by ts desc — the corpus the Manuscript
    surface reads from. Filters to events whose first content line looks
    like prose (skips events tagged 'finding' but carrying raw metadata
    such as `role=x · verdict=y`)."""
    # N.2 — per-request lab routing
    _lab_path, events_path, _state_dir = _resolve_lab_path(lab)
    findings: list[dict[str, Any]] = []
    for ev in _read_events(limit=10_000, events_path=events_path):
        if ev.get("event_class") != "finding":
            continue
        if not _is_editorial(ev):
            continue
        findings.append(ev)
    # newest first
    findings.sort(key=lambda e: e.get("ts", ""), reverse=True)
    page = findings[offset : offset + limit]
    return {"count": len(findings), "offset": offset, "limit": limit,
            "lab": lab or "(default)", "findings": page}


@app.get("/api/findings/{finding_id}")
def get_finding(finding_id: str,
                lab: str | None = Query(None)) -> dict[str, Any]:
    _lab_path, events_path, _state_dir = _resolve_lab_path(lab)
    ev = _read_event_by_id(finding_id, events_path=events_path)
    if ev is None or ev.get("event_class") != "finding":
        raise HTTPException(404, f"finding {finding_id!r} not found")
    # neighbours for prev/next navigation — only over editorial findings
    siblings: list[dict[str, Any]] = []
    for sib in _read_events(limit=10_000, events_path=events_path):
        if sib.get("event_class") != "finding":
            continue
        if not _is_editorial(sib):
            continue
        siblings.append(sib)
    siblings.sort(key=lambda e: e.get("ts", ""), reverse=True)
    index = next((i for i, s in enumerate(siblings) if s.get("id") == finding_id), -1)
    prev_id = siblings[index + 1]["id"] if 0 <= index < len(siblings) - 1 else None
    next_id = siblings[index - 1]["id"] if index > 0 else None
    return {
        "finding": ev,
        "index": index,
        "total": len(siblings),
        "prev_id": prev_id,
        "next_id": next_id,
    }


def _is_editorial(ev: dict[str, Any]) -> bool:
    """Server-side mirror of the editorial heuristic the Tide uses, so
    /api/findings doesn't surface raw key=value events tagged 'finding'."""
    content = (ev.get("content") or "").strip()
    if not content:
        return False
    first = content.split("\n", 1)[0].strip()
    if first.startswith("#"):
        return True
    if any(p in first for p in ".!?"):
        return True
    if first[:32].lower().count("=") >= 1 and "·" in first:
        return False
    return len(first) >= 24


@app.get("/api/letters/latest")
def get_latest_letter(lab: str | None = Query(None)) -> dict[str, Any]:
    """Return bert's most recent director letter, or a fallback fixture
    if the cron hasn't yet written one. Letters live as JSON files at
    <lab_path>/state/director_letters/letter_YYYY-MM-DD.json so they
    sort lexically by filename and the latest comes off the end of
    the list."""
    paths = _state_paths(lab)
    if paths.letters_dir.exists():
        files = sorted(paths.letters_dir.glob("letter_*.json"))
        if files:
            latest = files[-1]
            try:
                return json.loads(latest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return _fallback_letter()


def _fallback_letter() -> dict[str, Any]:
    """A hand-written fixture in voice direction B (direct/unceremonial),
    reflecting the actual recent lab state. Used until the midnight cron
    is wired up. Local time is read from the server's TZ — for the lab's
    PI in Madison, WI that's America/Chicago."""
    # Local time = server local time (assumed PI's machine)
    now_local = datetime.now().astimezone()
    weekday = now_local.strftime("%A")
    date_long = now_local.strftime("%-d %B %Y")
    time_short = now_local.strftime("%H:%M")
    # Find the current cycle from the event tail
    cycle = None
    if EVENTS_PATH.exists():
        stat = EVENTS_PATH.stat()
        with EVENTS_PATH.open("rb") as f:
            f.seek(max(0, stat.st_size - 64 * 1024))
            tail = f.read().decode("utf-8", errors="replace")
            for line in reversed(tail.split("\n")):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    cycle = ev.get("cycle")
                    break
                except json.JSONDecodeError:
                    continue
    cycle_str = f"cycle {cycle} in keeping" if cycle is not None else "no cycle running"
    return {
        "id": f"letter_fallback_{now_local.date().isoformat()}",
        "voice": "B",
        "is_fallback": True,
        "ts_local": now_local.isoformat(timespec="seconds"),
        "weekday": weekday,
        "date_long": date_long,
        "time_short": time_short,
        "cycle": cycle,
        "kicker": f"{weekday}, {date_long} · {time_short} — {cycle_str}",
        "salutation": "Dominus,",
        "body": [
            (
                "Quiet through the night. The lab held its rhythm — researcher "
                "and implementer kept their usual cadence; clearness was never "
                "called, nothing went to seasoning, no one stood aside."
            ),
            (
                "Nothing this morning needs you. The pending shelf is empty. "
                "When you're ready, the surfaces are below — the meeting, the "
                "tide, the manuscript. I'll be here."
            ),
        ],
        "signed": "— bert, director",
        "needs_dominus": False,
    }


# ── WRITE endpoints (Dominus → bert) ─────────────────────────────────


@app.post("/api/bless/{decision_id}")
def bless(decision_id: str, req: BlessRequest,
           lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    ev = _read_event_by_id(decision_id)
    if ev is None:
        raise HTTPException(404, f"decision {decision_id!r} not found")
    ts = _now_iso()
    action_id = _new_id("act")
    _append_jsonl(paths.pi_actions, {
        "id": action_id,
        "ts": ts,
        "action": "bless",
        "target": decision_id,
        "text": req.rationale,
        "applied_at_cycle": None,
        "applied_ts": None,
    })
    _append_jsonl(paths.blessings, {
        "decision_id": decision_id,
        "blessed_at": ts,
        "blessed_at_cycle": ev.get("cycle"),
        "rationale": req.rationale,
    })
    # I.1 — PI blessing is the strongest acceptance signal. Emit
    # artifact_accepted to anchor the §9 north-star metric.
    try:
        import sys as _sys
        _sys.path.insert(0, str(LAB_ROOT))
        from core import artifact_acceptance
        artifact_acceptance.emit_artifact_accepted(
            artifact_id=decision_id,
            source_dispatch_id=ev.get("source_path") or ev.get("dispatch_id"),
            cycle=ev.get("cycle") or 0,
            acceptance_kind=artifact_acceptance.KIND_PI_BLESSING,
            artifact_type=artifact_acceptance.TYPE_DECISION,
            rationale=req.rationale,
            role=ev.get("agent"),
        )
    except Exception:  # noqa: BLE001
        # Observability is best-effort; bless flow must not fail on
        # acceptance emission.
        pass
    return {"ok": True, "action_id": action_id, "ts": ts, "decision_id": decision_id}


@app.post("/api/veto/{decision_id}")
def veto(decision_id: str, req: VetoRequest,
          lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    ev = _read_event_by_id(decision_id)
    if ev is None:
        raise HTTPException(404, f"decision {decision_id!r} not found")
    ts = _now_iso()
    action_id = _new_id("act")
    _append_jsonl(paths.pi_actions, {
        "id": action_id,
        "ts": ts,
        "action": "veto",
        "target": decision_id,
        "text": req.reason,
        "applied_at_cycle": None,
        "applied_ts": None,
    })
    _append_jsonl(paths.vetoes, {
        "decision_id": decision_id,
        "vetoed_at": ts,
        "vetoed_at_cycle": ev.get("cycle"),
        "reason": req.reason,
    })
    return {"ok": True, "action_id": action_id, "ts": ts, "decision_id": decision_id}


@app.post("/api/ask")
def ask(req: AskRequest,
         lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    # Verify the target event exists (for non-special targets)
    if not req.target_id.startswith("global"):
        ev = _read_event_by_id(req.target_id)
        if ev is None:
            raise HTTPException(404, f"target {req.target_id!r} not found")
    ts = _now_iso()
    ask_id = _new_id("ask")
    record = {
        "id": ask_id,
        "ts_asked": ts,
        "target": req.target_id,
        "question": req.question,
        "answer": None,
        "answered_ts": None,
        "model": None,
    }
    _append_jsonl(paths.asks, record)
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"),
        "ts": ts,
        "action": "ask",
        "target": req.target_id,
        "text": req.question,
        "applied_at_cycle": None,
        "applied_ts": None,
    })
    return {"ok": True, "ask_id": ask_id, "ts": ts}


@app.post("/api/steer")
def steer(req: SteerRequest,
           lab: str | None = Query(None)) -> dict[str, Any]:
    """GG-B — PI talk-to-lab channel.

    Writes the steer to THREE places (atomic at the line-append
    level; each file's open-append is independent so a crash between
    writes can leave the ledger inconsistent — that's acceptable for
    prototype phase, the events.jsonl write is the load-bearing one):

      1. <lab>/state/steers.jsonl — audit ledger (consumed_at_cycle
         semantics: a cycle marks the steer consumed when it reads
         it during gather_observation)
      2. <lab>/state/pi_actions.jsonl — PI-action audit (every PI
         interaction with the lab lands here)
      3. <lab>/sor/events.jsonl — pi_message event_class entry so
         the director's gather_observation picks it up via the
         existing _read_recent_events path (GG-B.2 added pi_message
         to keep_classes)

    Pre-GG-B only (1) and (2) happened, so steers landed in audit
    files no one read. The director loop didn't engage with PI
    intent unless the PI re-typed the steer into seed_brief.md by
    hand. Bug-shaped UX.
    """
    paths = _state_paths(lab)
    ts = _now_iso()
    steer_id = _new_id("stx")
    _append_jsonl(paths.steers, {
        "id": steer_id,
        "ts": ts,
        "text": req.text,
        "modality": req.modality,
        "consumed_at_cycle": None,
    })
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"),
        "ts": ts,
        "action": "steer",
        "target": None,
        "text": req.text,
        "modality": req.modality,
        "applied_at_cycle": None,
        "applied_ts": None,
    })
    # GG-B — append to the lab's events.jsonl as pi_message so the
    # director's observation sees it on next cycle.
    events_path = paths.lab_path / "sor" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(events_path, {
        "id": steer_id,
        "ts": ts,
        "event_class": "pi_message",
        "agent": "pi",
        "cycle": None,
        "content": req.text,
        "text": req.text,
        "modality": req.modality,
        "tags": ["pi-steer"],
        "lineage": [],
        "verdict": None,
        "severity_grade": None,
    })
    return {"ok": True, "steer_id": steer_id, "ts": ts, "lab": lab or "(default)"}


@app.post("/api/voice-steer")
async def voice_steer(audio: UploadFile = File(...),
                       lab: str | None = Query(None)) -> dict[str, Any]:
    """Voice steer endpoint. Accepts audio blob, will route to whisper.cpp
    in a later integration. For now: records the upload and returns a
    placeholder transcript so the UI flow can be exercised end-to-end."""
    paths = _state_paths(lab)
    blob = await audio.read()
    if not blob:
        raise HTTPException(400, "empty audio payload")
    ts = _now_iso()
    # Persist the audio for later transcription
    paths.voice_steers_dir.mkdir(parents=True, exist_ok=True)
    audio_id = _new_id("vsteer")
    audio_path = paths.voice_steers_dir / f"{audio_id}.webm"
    audio_path.write_bytes(blob)
    placeholder = f"[voice steer captured · {len(blob)} bytes · awaiting whisper.cpp transcription]"
    # Log as a steer with whisper modality so bert eventually consumes it
    _append_jsonl(paths.steers, {
        "id": audio_id,
        "ts": ts,
        "text": placeholder,
        "modality": "whisper",
        "audio_path": str(audio_path),
        "consumed_at_cycle": None,
    })
    # GG-B — same pi_message append as typed steer so the director
    # sees voice steers in observation just like typed ones.
    events_path = paths.lab_path / "sor" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(events_path, {
        "id": audio_id,
        "ts": ts,
        "event_class": "pi_message",
        "agent": "pi",
        "cycle": None,
        "content": placeholder,
        "text": placeholder,
        "modality": "whisper",
        "audio_path": str(audio_path),
        "tags": ["pi-steer", "voice"],
        "lineage": [],
        "verdict": None,
        "severity_grade": None,
    })
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"),
        "ts": ts,
        "action": "steer",
        "target": None,
        "text": placeholder,
        "modality": "whisper",
        "applied_at_cycle": None,
        "applied_ts": None,
    })
    return {"ok": True, "steer_id": audio_id, "ts": ts, "transcript": placeholder, "bytes": len(blob)}


@app.post("/api/pin/{event_id}")
def pin(event_id: str, lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    overrides = _load_overrides(paths)
    if event_id not in overrides["pinned"]:
        overrides["pinned"].append(event_id)
    if event_id in overrides["suppressed"]:
        overrides["suppressed"].remove(event_id)
    _save_overrides(overrides, paths)
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"), "ts": _now_iso(), "action": "pin",
        "target": event_id, "text": None,
        "applied_at_cycle": None, "applied_ts": None,
    })
    return {"ok": True, "overrides": overrides}


@app.post("/api/unpin/{event_id}")
def unpin(event_id: str, lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    overrides = _load_overrides(paths)
    if event_id in overrides["pinned"]:
        overrides["pinned"].remove(event_id)
    _save_overrides(overrides, paths)
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"), "ts": _now_iso(), "action": "unpin",
        "target": event_id, "text": None,
        "applied_at_cycle": None, "applied_ts": None,
    })
    return {"ok": True, "overrides": overrides}


@app.post("/api/suppress/{event_id}")
def suppress(event_id: str, lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    overrides = _load_overrides(paths)
    if event_id not in overrides["suppressed"]:
        overrides["suppressed"].append(event_id)
    if event_id in overrides["pinned"]:
        overrides["pinned"].remove(event_id)
    _save_overrides(overrides, paths)
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"), "ts": _now_iso(), "action": "suppress",
        "target": event_id, "text": None,
        "applied_at_cycle": None, "applied_ts": None,
    })
    return {"ok": True, "overrides": overrides}


@app.post("/api/unsuppress/{event_id}")
def unsuppress(event_id: str, lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    overrides = _load_overrides(paths)
    if event_id in overrides["suppressed"]:
        overrides["suppressed"].remove(event_id)
    _save_overrides(overrides, paths)
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"), "ts": _now_iso(), "action": "unsuppress",
        "target": event_id, "text": None,
        "applied_at_cycle": None, "applied_ts": None,
    })
    return {"ok": True, "overrides": overrides}


@app.post("/api/pause")
def pause(req: PauseRequest, lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    ts = _now_iso()
    paths.paused_flag.parent.mkdir(parents=True, exist_ok=True)
    paths.paused_flag.write_text(
        json.dumps({"ts": ts, "reason": req.reason}, indent=2),
    )
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"), "ts": ts, "action": "pause",
        "target": None, "text": req.reason,
        "applied_at_cycle": None, "applied_ts": None,
    })
    return {"ok": True, "ts": ts, "paused": True, "lab": lab or "(default)"}


@app.post("/api/resume")
def resume(lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    ts = _now_iso()
    if paths.paused_flag.exists():
        paths.paused_flag.unlink()
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"), "ts": ts, "action": "resume",
        "target": None, "text": None,
        "applied_at_cycle": None, "applied_ts": None,
    })
    return {"ok": True, "ts": ts, "paused": False, "lab": lab or "(default)"}


@app.post("/api/approve/{approval_id}")
def approve(approval_id: str, req: ApproveRequest,
             lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    path = paths.approvals_dir / f"{approval_id}.json"
    if not path.exists():
        raise HTTPException(404, f"approval {approval_id!r} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"could not read approval: {e}") from e
    options = data.get("options") or []
    if req.choice not in options:
        raise HTTPException(
            422,
            f"choice {req.choice!r} is not one of the offered options: {options}",
        )
    # Quaker process distinguishes three verdicts: bless (approve),
    # block (reject), or stand aside (neither — bert may proceed but
    # without my consent). Map choice → status accordingly.
    choice_lc = req.choice.lower()
    if choice_lc == "reject":
        new_status = "rejected"
    elif choice_lc == "stand aside":
        new_status = "stand_aside"
    else:
        new_status = "approved"
    data["status"] = new_status
    data["decided_ts"] = _now_iso()
    data["chosen_option"] = req.choice
    data["rationale"] = req.rationale
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"), "ts": data["decided_ts"], "action": "approve_checkpoint",
        "target": approval_id, "text": req.choice,
        "applied_at_cycle": None, "applied_ts": None,
    })
    return {"ok": True, "approval": data}


@app.post("/api/dev/seed-approval")
def dev_seed_approval() -> dict[str, Any]:
    """Idempotent: create or reset a fixture approval used by /dev/gestures.
    Routes only exist under /api/dev/* for development affordances —
    bert never produces these in production cycles."""
    APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
    fixture_id = "appr_dev_demo_001"
    path = APPROVALS_DIR / f"{fixture_id}.json"
    payload = {
        "id": fixture_id,
        "asked_ts": _now_iso(),
        "cycle": 999,
        "question": "Promote the cross-family judge ensemble to the default verdict path?",
        "options": ["approve", "approve with caveat", "stand aside", "reject"],
        "context": (
            "Researcher + strategist concur (0.83 / 0.79 confidence). "
            "Falsifier abstains: no counterexamples in current corpus. "
            "Evaluator stand-aside cites $0.04/call cost increase, calls it 'within budget but worth Dominus's eye'."
        ),
        "status": "pending",
        "decided_ts": None,
        "chosen_option": None,
        "rationale": None,
        "is_dev": True,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"ok": True, "approval": payload}


@app.post("/api/notes/{event_id}")
def write_note(event_id: str, req: NoteRequest,
                lab: str | None = Query(None)) -> dict[str, Any]:
    paths = _state_paths(lab)
    paths.notes_dir.mkdir(parents=True, exist_ok=True)
    path = paths.notes_dir / f"{event_id}.md"
    ts = _now_iso()
    header = f"<!-- last edited {ts} -->\n"
    path.write_text(header + req.text, encoding="utf-8")
    _append_jsonl(paths.pi_actions, {
        "id": _new_id("act"), "ts": ts, "action": "note",
        "target": event_id, "text": req.text[:200],
        "applied_at_cycle": None, "applied_ts": None,
    })
    return {"ok": True, "event_id": event_id, "ts": ts}


# ── J.1 H-phase surfaces ─────────────────────────────────────────────
#
# Six read-only endpoints that expose the H.3–H.8 infrastructure to
# bert. Each is tolerant of empty/missing data — returns zero
# counts rather than 404 so the UI can render "no signal yet" panels
# instead of error states.


@app.get("/api/graph")
def graph_summary(lab: str | None = Query(None)) -> dict[str, Any]:
    """Knowledge-graph state with validity-window aggregates (H.3).

    Atlas consumes this as its "fourth ring" — the subsurface seams
    of the lab. Returns node/edge totals, validity-window
    distributions (active vs invalidated), time depth, and a small
    sample of the most-recent edges for context.

    Lab-scoped: each lab has its own graph.db under <lab>/state/.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    try:
        import sqlite3

        from core import graph_store
        from core.lab_context import reset_active_lab_path, set_active_lab_path
        # Resolve lab → set active-lab context so graph_store routes
        # to the right db. Reset at the end so subsequent requests
        # don't inherit our context.
        lab_path, _events_path, _state_dir = _resolve_lab_path(lab)
        # Only set context for non-default labs (LAB_ROOT/lab is the
        # default and graph_store's DB_PATH already points there).
        token = None
        if lab_path != LAB_ROOT / "lab":
            token = set_active_lab_path(lab_path)
        try:
            counts = graph_store.count()
            db_path = graph_store._active_db_path()
        finally:
            if token is not None:
                reset_active_lab_path(token)
        # Validity-window aggregates — computed inline so graph_store
        # public API stays clean.
        active_edges = 0
        invalidated_edges = 0
        oldest_valid_from: float | None = None
        newest_valid_to: float | None = None
        recent: list[dict[str, Any]] = []
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                (active_edges,) = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE valid_to IS NULL",
                ).fetchone()
                (invalidated_edges,) = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE valid_to IS NOT NULL",
                ).fetchone()
                row = conn.execute(
                    "SELECT MIN(valid_from) AS oldest, MAX(valid_to) AS newest FROM edges"
                ).fetchone()
                if row:
                    oldest_valid_from = row["oldest"]
                    newest_valid_to = row["newest"]
                cur = conn.execute(
                    "SELECT src, dst, type, valid_from, valid_to "
                    "FROM edges ORDER BY valid_from DESC NULLS LAST LIMIT 10"
                )
                for r in cur.fetchall():
                    recent.append({
                        "src": r["src"], "dst": r["dst"], "type": r["type"],
                        "valid_from": r["valid_from"], "valid_to": r["valid_to"],
                    })
        except sqlite3.OperationalError:
            # Schema may pre-date validity-window columns; fall through.
            pass
        return {
            "ts": _now_iso(),
            "nodes_total": counts.get("nodes_total", 0),
            "edges_total": counts.get("edges_total", 0),
            "edges_active": active_edges,
            "edges_invalidated": invalidated_edges,
            "nodes_by_type": counts.get("nodes_by_type", {}),
            "edges_by_type": counts.get("edges_by_type", {}),
            "oldest_valid_from": oldest_valid_from,
            "newest_valid_to": newest_valid_to,
            "recent": recent,
        }
    except Exception as e:
        # Don't 500 on empty/uninitialized KG — return zeroed payload
        # so Atlas renders "the strata are still being laid" instead
        # of an error.
        return {
            "ts": _now_iso(),
            "nodes_total": 0, "edges_total": 0,
            "edges_active": 0, "edges_invalidated": 0,
            "nodes_by_type": {}, "edges_by_type": {},
            "oldest_valid_from": None, "newest_valid_to": None,
            "recent": [], "_note": f"graph store unavailable: {e}",
        }


@app.get("/api/retrieval")
def retrieval_health() -> dict[str, Any]:
    """Hybrid-retrieval availability + scaffold health (H.4).

    Reports which retrieval adapters are wired (vector / graph /
    cache), whether the reranker is reachable, and the RRF k
    parameter in use. Doesn't run a live retrieval — that's expensive
    and would slow Diagnostics polling.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    out: dict[str, Any] = {
        "ts": _now_iso(),
        "rrf_k": 60,
        "adapters": {"vector": False, "graph": False, "cache": False},
        "reranker_available": False,
    }
    try:
        from core import retrieval
        out["adapters"] = {
            "vector": hasattr(retrieval, "_vector_candidates"),
            "graph": hasattr(retrieval, "_graph_candidates"),
            "cache": hasattr(retrieval, "_cache_candidates"),
        }
        out["reranker_available"] = hasattr(retrieval, "default_cosine_reranker")
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"retrieval unavailable: {e}"
    return out


@app.get("/api/compaction")
def compaction_status() -> dict[str, Any]:
    """Compaction-pipeline state (H.5).

    Surfaces the 3-strike auto-compact killswitch + recent shaper
    activity. Used by Diagnostics to flag a lab approaching
    auto-compact-loop pathology.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    out: dict[str, Any] = {
        "ts": _now_iso(),
        "shapers_in_order": [
            "budget_reduce", "snip_stale_tool_results",
            "microcompact_oldest", "context_collapse", "auto_compact",
        ],
        "strike_threshold": 3,
        "strike_window_secs": 600,
        "active_strikes_by_cycle": {},
        "killswitch_armed": True,
    }
    try:
        from core import compact
        # Snapshot of the in-process strike state (one process per
        # cycle, so this is bounded). Convert lists to counts only.
        strikes = getattr(compact, "_AUTO_COMPACT_STRIKES", {})
        out["active_strikes_by_cycle"] = {
            str(cycle): len(ts_list) for cycle, ts_list in strikes.items()
        }
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"compaction state unavailable: {e}"
    return out


@app.get("/api/quality-report")
def quality_report_latest() -> dict[str, Any]:
    """Latest H.6 weekly quality report.

    Reads the most-recent findings/weekly_quality_report_*.json. If
    no run has produced one yet, returns a placeholder shape so
    Manuscript can render "no weekly report yet — first run pending".
    """
    findings_dir = LAB_ROOT / "findings"
    if not findings_dir.exists():
        return {"ts": _now_iso(), "available": False,
                "reason": "findings/ missing"}
    candidates = sorted(
        findings_dir.glob("weekly_quality_report_*.json"),
        reverse=True,
    )
    if not candidates:
        return {"ts": _now_iso(), "available": False,
                "reason": "no weekly_quality_report_*.json yet"}
    latest = candidates[0]
    try:
        report = json.loads(latest.read_text())
        md_path = latest.with_suffix(".md")
        # The report is flat: top-level keys are the 8 measurement
        # section names + `grades` + `ts` + `window_secs`. Surface the
        # sections under a structured "sections" map for the UI; pass
        # grades through unchanged.
        section_names = (
            "cross_family_agreement", "skill_curator", "cache_drift",
            "memory_tier_budget", "falsifier_baseline", "idle_compute",
            "mcp_replay", "delegation",
        )
        sections = {n: report.get(n) for n in section_names if n in report}
        grades: dict[str, str] = report.get("grades", {})
        # Composite "overall grade" = worst grade across sections.
        # A > A- > B > C; preserve C as worst.
        order = {"A": 4, "A-": 3, "B": 2, "C": 1, "N/A": 0}
        worst: str | None = None
        for v in grades.values():
            if worst is None or order.get(v, 0) < order.get(worst, 0):
                worst = v
        return {
            "ts": _now_iso(),
            "available": True,
            "path": str(latest.relative_to(LAB_ROOT)),
            "md_path": (
                str(md_path.relative_to(LAB_ROOT)) if md_path.exists() else None
            ),
            "generated_at": report.get("ts"),
            "window_days": (report.get("window_secs", 0) or 0) // 86400 or 7,
            "overall_grade": worst,
            "grades": grades,
            "sections": sections,
        }
    except Exception as e:
        raise HTTPException(500, f"quality report unreadable: {e}") from e


@app.get("/api/eval-scorecard")
def eval_scorecard() -> dict[str, Any]:
    """OWASP Top-10 + MemoryAgentBench structural pass counts (H.7).

    Runs the structural @check functions live (cheap — they're
    filesystem-presence checks, ~milliseconds each) and returns a
    pass/fail count per suite. The slow part — Inspect AI live eval
    — is offline-only; this endpoint is the bert-visible summary.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    out: dict[str, Any] = {
        "ts": _now_iso(),
        "owasp": {"passed": 0, "failed": 0, "details": []},
        "memoryagentbench": {"passed": 0, "failed": 0, "axes": []},
    }
    # OWASP — call each _check_llmXX()
    try:
        from evals.inspect import owasp_top10
        for i in range(1, 11):
            fn = getattr(owasp_top10, f"_check_llm{i:02d}", None)
            if not fn:
                continue
            result = fn()
            passed = bool(result.get("passed"))
            out["owasp"]["details"].append({
                "id": f"LLM{i:02d}",
                "passed": passed,
                "rationale": result.get("rationale", "")[:160],
            })
            if passed:
                out["owasp"]["passed"] += 1
            else:
                out["owasp"]["failed"] += 1
    except Exception as e:  # noqa: BLE001
        out["owasp"]["_note"] = f"owasp checks unavailable: {e}"
    # MemoryAgentBench — structural shape only; the live eval would
    # require a model and is offline-only.
    try:
        from evals.inspect import memoryagentbench as mab
        for axis_name in ("accurate_retrieval", "test_time_learning",
                          "long_range_understanding", "conflict_resolution"):
            fn = getattr(mab, axis_name, None)
            if not fn:
                continue
            t = fn()
            samples = list(t.dataset)
            axis = samples[0].metadata.get("axis") if samples else "?"
            out["memoryagentbench"]["axes"].append({
                "axis": axis,
                "task": axis_name,
                "sample_count": len(samples),
                "wired": True,
            })
            out["memoryagentbench"]["passed"] += 1
    except Exception as e:  # noqa: BLE001
        out["memoryagentbench"]["_note"] = f"mab checks unavailable: {e}"
    return out


@app.get("/api/labs")
def list_labs_unified() -> dict[str, Any]:
    """L.4 + GG-A.1 — list scaffolded user labs + the active one.

    Pre-GG-A.1 this returned only basic metadata (name, path,
    archetype, cycle_count, template_origin). GG-A.1 enriches each
    scaffolded entry with FF-A-aware fields (focus_areas, role,
    mission, share_with_supervisor) via core.lab_config so the UI
    can render the dashboard without a second roundtrip per lab.

    The `{active, scaffolded}` shape is preserved (L.4 contract) +
    a `labs` array is included for callers that want a unified
    flat list including the supervisor. Honest disclosure: the
    `cycle_count` field counts dirs under `cycles/`; for runtime
    activity, prefer `events_total`.

    Switching labs at runtime still requires `BERT_LAB_PATH=<path>`
    on uvicorn restart — see active.is_bert_lab_default.
    """
    import os as _os
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    from core import lab_config as _lc

    home_labs = Path(_os.path.expanduser("~/.bert/labs"))
    scaffolded: list[dict[str, Any]] = []
    if home_labs.exists():
        for d in sorted(home_labs.iterdir()):
            if not d.is_dir():
                continue
            cfg_file = d / "lab.yaml"
            if not cfg_file.exists():
                continue
            cfg = _lc.load(d)
            template_origin = None
            try:
                for line in cfg_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("template_origin:"):
                        template_origin = line.split(":", 1)[1].strip().strip("'\"")
            except OSError:
                pass
            events_path = d / "sor" / "events.jsonl"
            cycle_dir = d / "cycles"
            cycle_count = (len(list(cycle_dir.glob("*")))
                           if cycle_dir.exists() else 0)
            scaffolded.append({
                "name": d.name,
                "path": str(d),
                # L.4 contract fields
                "archetype": cfg.archetype,
                "cycle_count": cycle_count,
                "template_origin": template_origin,
                # GG-A.1 FF-A-aware enrichment
                "role": cfg.role,
                "mission": cfg.mission,
                "focus_areas": list(cfg.focus_areas),
                "share_with_supervisor": cfg.share_with_supervisor,
                "events_total": (_count_lines(events_path)
                                  if events_path.exists() else 0),
                "config_warnings": list(cfg.parse_warnings),
            })

    # Unified flat list including the supervisor for callers that
    # want every lab in one array (the dashboard).
    supervisor_path = LAB_ROOT / "lab"
    labs_flat: list[dict[str, Any]] = []
    if supervisor_path.exists():
        sup_cfg = _lc.load(supervisor_path)
        sup_events = supervisor_path / "sor" / "events.jsonl"
        labs_flat.append({
            "name": "(default)",
            "path": str(supervisor_path),
            "is_supervisor": sup_cfg.is_supervisor,
            "archetype": sup_cfg.archetype,
            "role": sup_cfg.role,
            "mission": sup_cfg.mission,
            "focus_areas": list(sup_cfg.focus_areas),
            "share_with_supervisor": sup_cfg.share_with_supervisor,
            "events_total": (_count_lines(sup_events)
                              if sup_events.exists() else 0),
            "config_warnings": list(sup_cfg.parse_warnings),
        })
    for s in scaffolded:
        labs_flat.append({**s, "is_supervisor": s["role"] == "supervisor"})

    return {
        "ts": _now_iso(),
        # L.4 contract — DO NOT remove these fields
        "active": {
            "path": str(LAB_PATH),
            "is_bert_lab_default": LAB_PATH == LAB_ROOT / "lab",
        },
        "scaffolded": scaffolded,
        # GG-A.1 unified-flat addition
        "labs": labs_flat,
        "count": len(labs_flat),
        "_note": (
            "To view a different lab, restart uvicorn with "
            "BERT_LAB_PATH=/path/to/lab. Runtime switching is deferred."
        ),
    }


@app.get("/api/demo-mode")
def demo_mode_status() -> dict[str, Any]:
    """I.7 — BERT_DEMO_MODE toggle status.

    Returns {"enabled": bool, "policy": {...}} so the frontend can
    suppress dev surfaces, replace error toasts with warm-amber
    weather pills, force FirstLight as default, and auto-load the
    note-cli demo lab.

    Enabled when:
      - env var BERT_DEMO_MODE=on (or BERT_DEMO_MODE=1)
      - OR the local flag file lab/state/demo_mode.on exists

    Honest: returns enabled=false otherwise. The frontend treats
    enabled=true as "investor in the room — be polished."
    """
    import os as _os
    env_val = (_os.environ.get("BERT_DEMO_MODE") or "").lower()
    flag_path = LAB_PATH / "state" / "demo_mode.on"
    enabled = env_val in ("on", "1", "true", "yes") or flag_path.exists()
    return {
        "ts": _now_iso(),
        "enabled": enabled,
        "policy": {
            # Surfaces to hide (matches project_bert_demo_mode_and_polish.md)
            "hide_surfaces": [
                "DevGestures", "KeyboardHelp", "Choreography",
                "Infrastructure",
            ],
            "hide_routes": ["/dev/*"],
            # Default landing surface when demo-mode is on
            "default_surface": "FirstLight",
            # Provider failover chain (Groq → NVIDIA → Ollama)
            "provider_chain": ["groq", "nvidia-prod", "ollama"],
            # Auto-load demo lab if no other lab is present
            "auto_load_lab": "note-cli",
            # Toast suppression: replace these substrings with weather pills
            "toast_suppress_patterns": ["undefined", "null", "error", "failed"],
        },
    }


@app.get("/api/artifact-acceptance")
def artifact_acceptance_status(window_days: int = Query(7, ge=1, le=90)) -> dict[str, Any]:
    """I.1 — §9 north-star metric: accepted artifacts per lab-week.

    Returns the acceptance rate (accepted / shippable-verdicts), the
    grade (A/A-/B/C/INSUFFICIENT_DATA), and breakdowns by acceptance
    kind, artifact type, and role. Diagnostics surface displays this
    as the lab's primary health number.
    """
    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    try:
        from core import artifact_acceptance
        window_secs = window_days * 86400
        g = artifact_acceptance.grade(window_secs=window_secs)
        return {"ts": _now_iso(), "window_days": window_days, **g}
    except Exception as e:
        # Honest degradation: return zeros + reason instead of 500.
        return {"ts": _now_iso(), "window_days": window_days,
                "letter": "INSUFFICIENT_DATA",
                "reason": f"artifact_acceptance unavailable: {e}",
                "accepted_n": 0, "shippable_verdicts_n": 0,
                "acceptance_rate": 0.0,
                "by_kind": {}, "by_type": {}, "by_role": {}}


@app.get("/api/token-redundancy")
def token_redundancy_latest() -> dict[str, Any]:
    """Latest H.8 token-redundancy measurement.

    Reads the most-recent findings/token_redundancy_*.json. Returns
    placeholder shape when no run has produced one yet.
    """
    findings_dir = LAB_ROOT / "findings"
    if not findings_dir.exists():
        return {"ts": _now_iso(), "available": False,
                "reason": "findings/ missing"}
    candidates = sorted(
        findings_dir.glob("token_redundancy_*.json"),
        reverse=True,
    )
    if not candidates:
        return {"ts": _now_iso(), "available": False,
                "reason": "no token_redundancy_*.json yet"}
    latest = candidates[0]
    try:
        report = json.loads(latest.read_text())
        return {
            "ts": _now_iso(),
            "available": True,
            "path": str(latest.relative_to(LAB_ROOT)),
            **report,  # method, roles, overall, grade
        }
    except Exception as e:
        raise HTTPException(500, f"token redundancy report unreadable: {e}") from e


# ── Mission input: seed_brief.md read/write (AA.1) ───────────────────


@app.get("/api/seed-brief")
def get_seed_brief(lab: str | None = Query(None)) -> dict[str, Any]:
    """Return the active lab's seed_brief.md content + mtime.

    The mtime is the optimistic-lock token the client passes back on
    PUT to detect concurrent edits.
    """
    lab_path, _events_path, _state_dir = _resolve_lab_path(lab)
    seed_file = lab_path / "seed_brief.md"
    if not seed_file.exists():
        return {
            "ts": _now_iso(),
            "lab": lab or "(default)",
            "path": str(seed_file.relative_to(LAB_ROOT)) if seed_file.is_relative_to(LAB_ROOT) else str(seed_file),
            "exists": False,
            "content": "",
            "mtime": None,
            "size_bytes": 0,
        }
    stat = seed_file.stat()
    return {
        "ts": _now_iso(),
        "lab": lab or "(default)",
        "path": str(seed_file.relative_to(LAB_ROOT)) if seed_file.is_relative_to(LAB_ROOT) else str(seed_file),
        "exists": True,
        "content": seed_file.read_text(),
        "mtime": stat.st_mtime,
        "size_bytes": stat.st_size,
    }


class SeedBriefWrite(BaseModel):
    content: str
    expected_mtime: float | None = None  # optimistic-lock token


@app.put("/api/seed-brief")
def put_seed_brief(
    req: SeedBriefWrite,
    lab: str | None = Query(None),
) -> dict[str, Any]:
    """Write the active lab's seed_brief.md.

    Optimistic-lock: if `expected_mtime` is provided AND the file's
    current mtime differs, returns 409 — someone else edited it. This
    prevents the UI's "save my mission" from silently clobbering a
    concurrent edit by another window or by the CLI.

    Validation: content must be non-empty and ≤ 32KB. Above 32KB the
    brief assembler trims aggressively anyway.
    """
    if not req.content or not req.content.strip():
        raise HTTPException(400, "seed_brief content must be non-empty")
    if len(req.content) > 32 * 1024:
        raise HTTPException(400,
            f"seed_brief content too large ({len(req.content)} bytes; "
            f"max 32KB). Aggressive trimming happens in brief_assembler anyway.")

    lab_path, _events_path, _state_dir = _resolve_lab_path(lab)
    seed_file = lab_path / "seed_brief.md"

    # Optimistic lock check
    if req.expected_mtime is not None and seed_file.exists():
        current_mtime = seed_file.stat().st_mtime
        # Tolerate small float-precision wobble (<1 ms)
        if abs(current_mtime - req.expected_mtime) > 0.001:
            raise HTTPException(
                409,
                f"concurrent edit detected: file mtime is {current_mtime} "
                f"but client expected {req.expected_mtime}. Reload the "
                f"brief and merge before saving."
            )

    # Ensure parent dir exists (defensive — should already)
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    seed_file.write_text(req.content)
    new_stat = seed_file.stat()
    return {
        "ts": _now_iso(),
        "lab": lab or "(default)",
        "path": str(seed_file.relative_to(LAB_ROOT)) if seed_file.is_relative_to(LAB_ROOT) else str(seed_file),
        "ok": True,
        "mtime": new_stat.st_mtime,
        "size_bytes": new_stat.st_size,
    }


# ── Mission input: run-cycle subprocess + SSE (AA.2) ─────────────────

# In-memory registry of run_id → subprocess.Popen handles. Sufficient
# for single-user laptop deployment. For multi-tenant we'd need
# a backing store (Redis, sqlite).
_RUN_REGISTRY: dict[str, dict[str, Any]] = {}


class RunCycleRequest(BaseModel):
    lab: str | None = None
    max_cycles: int = 1
    model: str | None = None
    dry_run: bool = False
    # GG-C — autonomous loop support (bugs #3 + #7).
    # Pre-GG /api/run-cycle capped at 5 cycles AND didn't pass
    # --autonomous to bert_run, so the UI could only fire short
    # manual batches. Post-GG: autonomous=True enables the
    # director-led loop and raises the cap to 50 (still capped to
    # prevent the UI from kicking off a multi-day run). CC.4
    # termination guardrails (3-strike, failure cascade, pending
    # threshold, IDLE) make runaway structurally prevented.
    autonomous: bool = False
    # Consent flag for max_cycles > 5. UI must surface a confirm
    # dialog before passing consent=True. Prevents accidental long
    # runs from a misclick.
    consent_long_run: bool = False


@app.post("/api/run-cycle")
def post_run_cycle(req: RunCycleRequest) -> dict[str, Any]:
    """Spawn bert_run.py as a subprocess; return a run_id the client
    can use to stream stdout via /api/run-cycle/{run_id}/stream.

    Stage-safety (GG-C):
      - max_cycles must be in [1, 50]
      - max_cycles > 5 REQUIRES consent_long_run=True (UI must
        surface a confirm dialog before passing this)
      - autonomous=True activates the director-led loop with CC.4
        termination guardrails; without it bert_run runs the
        manual researcher → strategist pipeline only
    """
    if req.max_cycles < 1 or req.max_cycles > 50:
        raise HTTPException(400,
            f"max_cycles must be 1..50 (got {req.max_cycles}). For "
            f"unbounded loops use `bert_run.py --watch` from CLI.")
    if req.max_cycles > 5 and not req.consent_long_run:
        raise HTTPException(400,
            f"max_cycles={req.max_cycles} > 5 requires "
            f"consent_long_run=True. UI must surface a confirm dialog "
            f"before firing this — autonomous cycles bill against "
            f"provider quotas even when bert's compute is free.")

    # Validate lab resolves
    lab_path, _events_path, _state_dir = _resolve_lab_path(req.lab)
    seed_file = lab_path / "seed_brief.md"
    if not seed_file.exists():
        raise HTTPException(400,
            f"lab {req.lab!r} has no seed_brief.md — write one via "
            f"PUT /api/seed-brief?lab={req.lab} first.")

    # Compose the subprocess command
    venv_py = LAB_ROOT / ".venv" / "bin" / "python"
    bert_run = LAB_ROOT / "tools" / "bert_run.py"
    cmd = [str(venv_py), str(bert_run),
           "--lab", str(lab_path),
           "--max-cycles", str(req.max_cycles)]
    if req.model:
        cmd.extend(["--model", req.model])
    if req.dry_run:
        cmd.append("--dry-run")
    # GG-C — pass --autonomous when set so bert_run engages the
    # director-led loop (CC phase) with episodic-feedback grading
    # (EE phase) and the per-lab pause-flag check (GG-A-prep).
    if req.autonomous:
        cmd.append("--autonomous")

    run_id = _new_id("run")
    import subprocess as _sp
    import threading
    proc = _sp.Popen(
        cmd,
        stdout=_sp.PIPE,
        stderr=_sp.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        cwd=str(LAB_ROOT),
    )
    entry: dict[str, Any] = {
        "proc": proc,
        "cmd": " ".join(cmd),
        "lab": req.lab or "(default)",
        "max_cycles": req.max_cycles,
        "dry_run": req.dry_run,
        "autonomous": req.autonomous,
        "started_ts": _now_iso(),
        "lines": [],
        "drained": False,  # set True when the reader thread exits
    }
    _RUN_REGISTRY[run_id] = entry

    # Sprint 4 B — mirror to the durable registry so a restarted API can
    # reap this subprocess if it orphans (criterion 21).
    try:
        from core import run_registry as _rr
        _rr.record_start(run_id, pid=proc.pid, lab=lab_path.name)
    except Exception:  # noqa: BLE001
        pass  # durable mirror is best-effort; in-memory registry still works

    # AA.2 fix — drain stdout in a background thread, regardless of
    # whether anyone connects to the SSE stream. Without this, status
    # polls return line_count=0 because nothing is reading the pipe.
    def _drain_stdout(ent: dict[str, Any]) -> None:
        try:
            for line in ent["proc"].stdout:
                ent["lines"].append(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            ent["proc"].wait()
            ent["drained"] = True
            ent["exit_code"] = ent["proc"].returncode
            try:
                from core import run_registry as _rr2
                _rr2.record_finish(run_id, exit_code=ent["proc"].returncode)
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_drain_stdout, args=(entry,), daemon=True).start()

    return {
        "ts": _now_iso(),
        "run_id": run_id,
        "lab": req.lab or "(default)",
        "max_cycles": req.max_cycles,
        "dry_run": req.dry_run,
        "autonomous": req.autonomous,
        "stream_url": f"/api/run-cycle/{run_id}/stream",
    }


@app.get("/api/run-cycle/{run_id}/stream")
def stream_run_cycle(run_id: str):
    """SSE stream of the subprocess's stdout (merged with stderr).

    Each line of the bert_run.py output becomes one SSE event. Closes
    when the subprocess exits.
    """
    import asyncio

    from fastapi.responses import StreamingResponse

    entry = _RUN_REGISTRY.get(run_id)
    if entry is None:
        raise HTTPException(404, f"run_id {run_id!r} not found")
    entry["proc"]

    async def gen():
        # AA.2 fix — the reader thread is the single owner of proc.stdout.
        # The SSE stream just tails entry["lines"] (cumulative) and waits
        # for `drained=True`. Avoids double-readers (which would deadlock
        # or split the stdout stream between the thread and the SSE coro).
        emitted = 0
        while True:
            cur_n = len(entry.get("lines", []))
            while emitted < cur_n:
                line = entry["lines"][emitted]
                yield f"data: {json.dumps({'line': line})}\n\n"
                emitted += 1
            if entry.get("drained"):
                rc = entry.get("exit_code")
                yield f"data: {json.dumps({'done': True, 'exit_code': rc})}\n\n"
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/run-cycle/{run_id}")
def get_run_status(run_id: str) -> dict[str, Any]:
    """Poll-friendly status: alive/exit_code, lines so far, started ts."""
    entry = _RUN_REGISTRY.get(run_id)
    if entry is None:
        # Sprint 4 B — not in the in-memory map (e.g. API restarted). Fall
        # back to the durable registry so status is still recoverable.
        try:
            from core import run_registry
            rec = run_registry.get(run_id)
        except Exception:  # noqa: BLE001
            rec = None
        if rec is not None:
            return {"ts": _now_iso(), "run_id": run_id, "lab": rec.lab,
                    "status": rec.status, "exit_code": rec.exit_code,
                    "alive": rec.status == "running", "from": "durable_registry"}
        raise HTTPException(404, f"run_id {run_id!r} not found")
    proc = entry["proc"]
    is_alive = proc.poll() is None
    lines = entry.get("lines", [])

    # Mission complete + cycle counter — derived from bert_run stdout so
    # the UI can show a live counter and a mission-complete receipt
    # without needing to also subscribe to /api/events/stream just for
    # this run. The bert_run prints lines like "[cycle N] ✓ success"
    # per completed cycle and "[stop] director declared MISSION
    # COMPLETE" on completion.
    cycles_completed = sum(
        1 for ln in lines
        if "] ✓ success" in ln or "] ✗ partial" in ln
    )
    mission_complete = any(
        "MISSION COMPLETE" in ln for ln in lines
    )
    return {
        "ts": _now_iso(),
        "run_id": run_id,
        "lab": entry["lab"],
        "max_cycles": entry["max_cycles"],
        "dry_run": entry["dry_run"],
        "started_ts": entry["started_ts"],
        "alive": is_alive,
        "exit_code": None if is_alive else proc.returncode,
        "line_count": len(lines),
        "lines_tail": lines[-20:],  # last 20 lines for quick view
        # GG-D — surface cancellation status separately from exit_code
        # so the UI knows whether a non-zero exit was a normal failure
        # or an operator cancel.
        "cancelled": bool(entry.get("cancelled")),
        "cancelled_ts": entry.get("cancelled_ts"),
        # Mission completion derivatives (added with the start/stop
        # autonomous-loop UI rework).
        "cycles_completed": cycles_completed,
        "mission_complete": mission_complete,
    }


@app.delete("/api/run-cycle/{run_id}")
def cancel_run_cycle(run_id: str) -> dict[str, Any]:
    """GG-D — Cancel a running bert_run subprocess.

    Sends SIGTERM first (gives the cycle a chance to finish its
    current dispatch and flush partial state cleanly via the
    autonomous loop's Ctrl-C handler). Idempotent: DELETE-ing an
    already-cancelled or already-finished run returns the same
    "cancelled" payload without re-killing.

    Honest disclosure: this cancels the orchestrator process, but
    in-flight provider HTTP calls land their responses to a dead
    process. The provider's quota counter ticks regardless — bert
    only refunds CPU, never tokens. The UI should make this clear
    in the confirm dialog.
    """
    import signal as _signal
    entry = _RUN_REGISTRY.get(run_id)
    if entry is None:
        raise HTTPException(404, f"run_id {run_id!r} not found")
    proc = entry["proc"]

    if entry.get("cancelled"):
        # Idempotent — already cancelled, return current state
        return {
            "ts": _now_iso(),
            "run_id": run_id,
            "cancelled": True,
            "cancelled_ts": entry.get("cancelled_ts"),
            "already_cancelled": True,
            "alive": proc.poll() is None,
            "exit_code": None if proc.poll() is None else proc.returncode,
        }

    is_alive = proc.poll() is None
    if not is_alive:
        # Already finished naturally — mark cancelled=False but echo
        # the terminal state. Distinguishes operator-cancel from
        # normal exit.
        return {
            "ts": _now_iso(),
            "run_id": run_id,
            "cancelled": False,
            "already_finished": True,
            "alive": False,
            "exit_code": proc.returncode,
        }

    # Send SIGTERM. The autonomous loop's signal handler treats this
    # like Ctrl-C — it sets interrupted["caught"] = True, finishes
    # the current dispatch, and exits cleanly.
    try:
        proc.send_signal(_signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as exc:
        raise HTTPException(500,
            f"could not send SIGTERM to run {run_id!r}: {exc}") from exc

    ts = _now_iso()
    entry["cancelled"] = True
    entry["cancelled_ts"] = ts
    return {
        "ts": ts,
        "run_id": run_id,
        "cancelled": True,
        "cancelled_ts": ts,
        "alive": proc.poll() is None,  # may still be draining; recheck via GET
        "exit_code": None if proc.poll() is None else proc.returncode,
    }


# ── GG-E: Proof packet outputs ───────────────────────────────────────


PROOF_PACKETS_DIR = LAB_ROOT / "findings" / "proof_packets"


def _peek_packet_metadata(tarball: Path) -> dict[str, Any] | None:
    """Read cycle.json + manifest.json from a packet .tar.gz without
    extracting the full tarball. Returns the metadata dict, or None
    if the packet is malformed."""
    import tarfile as _tar
    try:
        with _tar.open(tarball, "r:gz") as tar:
            cycle_json = None
            manifest = None
            for member in tar.getmembers():
                if member.name.endswith("/cycle.json") and cycle_json is None:
                    f = tar.extractfile(member)
                    if f:
                        cycle_json = json.loads(f.read().decode())
                if member.name.endswith("/manifest.json") and manifest is None:
                    f = tar.extractfile(member)
                    if f:
                        try:
                            manifest = json.loads(f.read().decode())
                        except json.JSONDecodeError:
                            manifest = None
                if cycle_json and manifest:
                    break
            if cycle_json is None:
                return None
            stat = tarball.stat()
            return {
                "cycle_id": cycle_json.get("cycleId") or tarball.stem,
                "lab_ref": cycle_json.get("labRef") or "",
                "schema_version": cycle_json.get("schemaVersion") or "",
                "started_at": cycle_json.get("startedAt"),
                "completed_at": cycle_json.get("completedAt"),
                "event_count": cycle_json.get("eventCount", 0),
                "artifact_count": cycle_json.get("artifactCount", 0),
                "claims_count": len(cycle_json.get("claims", []) or []),
                "limitations_count": len(
                    cycle_json.get("limitations", []) or []),
                "parent_cycle_id": cycle_json.get("parentCycleId"),
                "provider": cycle_json.get("provider"),
                "tarball_path": str(tarball.relative_to(LAB_ROOT)),
                "tarball_bytes": stat.st_size,
                "tarball_mtime": stat.st_mtime,
            }
    except (OSError, _tar.TarError, json.JSONDecodeError):
        return None


def _lab_ref_matches(lab_ref: str, lab_filter: str | None) -> bool:
    """Match a packet's labRef ('local://bert@cycle-0400') against a
    lab filter ('bert' or None). Pre-FF packets used a fixed 'bert'
    labRef regardless of which lab generated them; post-FF packets
    encode the lab name. We're permissive for prototype phase.
    """
    if lab_filter is None:
        return True
    # Common shapes: 'local://<name>@cycle-N' or '<name>/cycle-N'
    if "@" in lab_ref:
        name = lab_ref.split("//")[-1].split("@")[0]
    elif "/" in lab_ref:
        name = lab_ref.rsplit("/", 1)[0]
    else:
        name = lab_ref
    # The supervisor lab's labRef is "bert" (legacy) or the lab name
    if lab_filter in ("(default)", "bert-self", "bert"):
        return name in ("bert", "bert-self", "")
    return name == lab_filter


@app.get("/api/proof-packets")
def list_proof_packets(
    lab: str | None = Query(None),
) -> dict[str, Any]:
    """GG-E — List proof packets, filtered by lab.

    Reads findings/proof_packets/*.tar.gz, peeks each cycle.json
    (extracts only that one file, not the whole tarball), filters
    by `labRef`. Returns newest-first sorted by tarball mtime.

    Per the GG-A.1 decision: packets are flat-with-metadata, not
    per-lab subdirectory. Filtering happens at read time.
    """
    if not PROOF_PACKETS_DIR.exists():
        return {"count": 0, "packets": [], "lab_filter": lab}
    out: list[dict[str, Any]] = []
    for tarball in sorted(PROOF_PACKETS_DIR.glob("*.tar.gz")):
        meta = _peek_packet_metadata(tarball)
        if meta is None:
            continue
        if not _lab_ref_matches(meta["lab_ref"], lab):
            continue
        out.append(meta)
    # Newest first
    out.sort(key=lambda m: m.get("tarball_mtime", 0), reverse=True)
    return {
        "count": len(out),
        "packets": out,
        "lab_filter": lab,
    }


@app.get("/api/proof-packets/{cycle_id}")
def get_proof_packet(cycle_id: str) -> dict[str, Any]:
    """GG-E — Return detailed proof packet contents.

    Extracts cycle.json + adversarial.json + failures.md +
    self-eval.json into memory and returns them as a structured
    bundle. Does NOT serve the tarball itself — that's a separate
    download path (deferred until customer-pickup flow needs it).
    """
    # Tolerate either "cycle-0400" or "0400" or "400"
    candidates = [
        PROOF_PACKETS_DIR / f"{cycle_id}.tar.gz",
        PROOF_PACKETS_DIR / f"cycle-{cycle_id}.tar.gz",
    ]
    if cycle_id.isdigit():
        candidates.append(PROOF_PACKETS_DIR / f"cycle-{int(cycle_id):04d}.tar.gz")
    tarball = next((p for p in candidates if p.exists()), None)
    if tarball is None:
        raise HTTPException(404, f"proof packet {cycle_id!r} not found")

    import tarfile as _tar
    cycle_json: dict | None = None
    adversarial: dict | None = None
    self_eval: dict | None = None
    failures_md: str | None = None
    file_index: list[dict[str, Any]] = []
    try:
        with _tar.open(tarball, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                # Strip the top-level cycle-NNNN/ prefix
                rel = "/".join(member.name.split("/")[1:])
                file_index.append({"path": rel, "size": member.size})
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                if rel == "cycle.json":
                    cycle_json = json.loads(data.decode())
                elif rel == "eval/adversarial.json":
                    adversarial = json.loads(data.decode())
                elif rel == "eval/self-eval.json":
                    with contextlib.suppress(json.JSONDecodeError):
                        self_eval = json.loads(data.decode())
                elif rel == "failures.md":
                    failures_md = data.decode(errors="replace")
    except (_tar.TarError, OSError) as exc:
        raise HTTPException(500,
            f"could not read proof packet {cycle_id!r}: {exc}") from exc

    if cycle_json is None:
        raise HTTPException(500,
            f"proof packet {cycle_id!r} missing cycle.json")

    return {
        "cycle_id": cycle_json.get("cycleId") or cycle_id,
        "tarball_path": str(tarball.relative_to(LAB_ROOT)),
        "tarball_bytes": tarball.stat().st_size,
        "cycle_json": cycle_json,
        "adversarial": adversarial,
        "self_eval": self_eval,
        "failures_md": failures_md,
        "file_index": file_index,
    }


@app.post("/api/proof-packets/{cycle_id}/verify")
def verify_proof_packet(cycle_id: str,
                         fetch_rekor: bool = Query(False)) -> dict[str, Any]:
    """GG-E — Run the 8-check verification ladder on a proof packet.

    Wraps core.verify_packet.verify_packet. Returns the full ladder
    so the UI can render each check with its status + detail. Honest
    disclosure: the local-dev signing mode reports [3] Rekor and
    [4] RFC3161 as WARN (skipped) — see DD.2 retired one-flip
    framing. The verifier doesn't lie about what it didn't check.
    """
    candidates = [
        PROOF_PACKETS_DIR / f"{cycle_id}.tar.gz",
        PROOF_PACKETS_DIR / f"cycle-{cycle_id}.tar.gz",
    ]
    if cycle_id.isdigit():
        candidates.append(PROOF_PACKETS_DIR / f"cycle-{int(cycle_id):04d}.tar.gz")
    tarball = next((p for p in candidates if p.exists()), None)
    if tarball is None:
        raise HTTPException(404, f"proof packet {cycle_id!r} not found")

    import sys as _sys
    _sys.path.insert(0, str(LAB_ROOT))
    from core import verify_packet as _vp
    result = _vp.verify_packet(tarball, fetch_rekor=fetch_rekor)
    return {
        "cycle_id": cycle_id,
        "tarball_path": str(tarball.relative_to(LAB_ROOT)),
        "overall": result.overall,
        "claims_count": result.claims_count,
        "failures_count": result.failures_count,
        "checks": [
            {
                "name": c.name,
                "status": c.status.value,
                "detail": c.detail,
                "cosign_equivalent": c.cosign_equivalent,
            }
            for c in result.checks
        ],
    }
