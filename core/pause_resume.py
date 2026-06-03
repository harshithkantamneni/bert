"""needs_user_input envelope + lab_resume primitives.

Phase C1 of the v3 plan. Implements the collaborative-autonomous loop:
when bert hits a fork it can't (or shouldn't) resolve on its own, it
returns a NeedsUserInput envelope. The MCP host (Claude / Cursor)
surfaces the question to the user via natural conversation, gets the
answer, then calls `lab_resume(token, answer)` to continue.

Design choices:
  - Resume token = HMAC-signed JSON blob (no external state required)
  - Saved state lives at `<lab>/state/paused/<token_id>.json`
  - Expiry: 24 hours by default (configurable per envelope)
  - HMAC key derived from `~/.bert-lab/signing.key` (existing Ed25519
    key, reused as HMAC secret via hash)
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOG = logging.getLogger("bert.pause_resume")

LAB_ROOT = Path(__file__).resolve().parent.parent

# Default expiry — long enough that the user has time to reply (24h),
# short enough that abandoned tokens don't accumulate forever.
DEFAULT_EXPIRY_SECS = 24 * 3600


# ── Dataclasses ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Option:
    """One choice in a needs_user_input envelope."""
    value: str               # what the user types/says
    label: str               # human-readable label
    cost_usd_est: float | None = None
    risk_level: str | None = None  # 'low' | 'medium' | 'high'


@dataclass(frozen=True)
class NeedsUserInput:
    """Envelope returned when bert wants the user to resolve a fork.

    MCP host sees this in the response body, asks the user, then calls
    `lab_resume(token=resume_token, answer=user_choice)`.
    """
    status: str = "needs_user_input"     # constant; MCP-host's flag
    question: str = ""
    options: tuple[Option, ...] = ()
    rationale: str = ""                  # why bert is asking (so user understands)
    resume_token: str = ""               # HMAC-signed; used by lab_resume

    def to_envelope(self) -> dict:
        """Serialized for return through the MCP tool response."""
        return {
            "status": self.status,
            "question": self.question,
            "options": [asdict(o) for o in self.options],
            "rationale": self.rationale,
            "resume_token": self.resume_token,
        }


@dataclass
class PausedState:
    """Server-side state saved when bert pauses for user input."""
    lab: str
    cycle: int
    step_id: str                          # arbitrary; identifies what we paused at
    saved_state: dict = field(default_factory=dict)  # arbitrary payload
    created_at_ts: float = field(default_factory=time.time)
    expires_at_ts: float = 0.0            # set on save


# ── HMAC key (from signing.key) ──────────────────────────────────


def _hmac_key() -> bytes:
    """Derive an HMAC key from bert's signing key. Reused so users
    don't need a second secret."""
    key_path = Path.home() / ".bert-lab" / "signing.key"
    if key_path.exists():
        try:
            content = key_path.read_bytes()
            return hashlib.sha256(content).digest()
        except OSError:
            pass
    # Fallback: derive from a stable host-local secret. Not ideal but
    # better than refusing to operate; tokens are still unforgeable
    # against external attackers (they don't have the secret).
    fallback = LAB_ROOT / "state" / "_fallback_hmac.key"
    if not fallback.exists():
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_bytes(secrets.token_bytes(32))
    return hashlib.sha256(fallback.read_bytes()).digest()


def _sign(payload: bytes) -> str:
    """HMAC-SHA256 hex digest."""
    return hmac.new(_hmac_key(), payload, hashlib.sha256).hexdigest()


# ── Token mint + verify ──────────────────────────────────────────


def mint_resume_token(state: PausedState) -> str:
    """Create a signed resume token from a PausedState.

    Token format: base64-safe JSON {state...} + '.' + HMAC digest hex.
    """
    if state.expires_at_ts <= 0:
        # Caller didn't set expiry; default 24h
        object.__setattr__(state, "expires_at_ts",
                           state.created_at_ts + DEFAULT_EXPIRY_SECS)
    state_dict = asdict(state)
    # Add a random nonce to defeat replay across labs/cycles
    state_dict["nonce"] = secrets.token_hex(8)
    body = json.dumps(state_dict, separators=(",", ":"), sort_keys=True).encode()
    sig = _sign(body)
    import base64
    b64 = base64.urlsafe_b64encode(body).decode().rstrip("=")
    return f"{b64}.{sig}"


def verify_resume_token(token: str) -> PausedState | None:
    """Parse + verify a resume token. Returns the PausedState on
    success, None on failure (bad sig, expired, malformed)."""
    if not token or "." not in token:
        return None
    b64, sig = token.rsplit(".", 1)
    # Add padding back
    pad = "=" * (-len(b64) % 4)
    try:
        import base64
        body = base64.urlsafe_b64decode(b64 + pad)
    except (ValueError, TypeError):
        return None
    expected = _sign(body)
    if not hmac.compare_digest(sig, expected):
        LOG.warning("resume token signature mismatch")
        return None
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None
    if float(d.get("expires_at_ts", 0)) < time.time():
        LOG.warning("resume token expired")
        return None
    # Reconstruct PausedState (drop the nonce field that's only for replay protection)
    d.pop("nonce", None)
    try:
        return PausedState(
            lab=d["lab"],
            cycle=int(d["cycle"]),
            step_id=d["step_id"],
            saved_state=d.get("saved_state", {}),
            created_at_ts=float(d.get("created_at_ts", time.time())),
            expires_at_ts=float(d.get("expires_at_ts", 0)),
        )
    except (KeyError, ValueError, TypeError) as e:
        LOG.warning("malformed resume token: %s", e)
        return None


# ── On-disk persistence (mirror of the token, indexed by lab) ────


def _paused_dir(lab_path: Path) -> Path:
    d = lab_path / "state" / "paused"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_paused_state(lab_path: Path, state: PausedState) -> Path:
    """Write the paused state to disk for the lab so the user can
    inspect pending pauses + the runner can find them on next start."""
    if state.expires_at_ts <= 0:
        object.__setattr__(state, "expires_at_ts",
                           state.created_at_ts + DEFAULT_EXPIRY_SECS)
    d = _paused_dir(lab_path)
    fname = f"{int(state.created_at_ts)}_{state.step_id}.json"
    p = d / fname
    p.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))
    return p


def list_pending(lab_path: Path) -> list[dict]:
    """Return all pending paused states for a lab. Cleans up expired
    files as a side effect."""
    out = []
    now = time.time()
    for p in _paused_dir(lab_path).glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if float(d.get("expires_at_ts", 0)) < now:
            with contextlib.suppress(OSError):
                p.unlink()
            continue
        d["_file"] = str(p)
        out.append(d)
    return out


def clear_paused(lab_path: Path, step_id: str) -> bool:
    """Remove the on-disk file for a resolved pause."""
    for p in _paused_dir(lab_path).glob(f"*_{step_id}.json"):
        try:
            p.unlink()
            return True
        except OSError:
            pass
    return False


# ── High-level helpers ───────────────────────────────────────────


def build_envelope(
    *,
    lab: str,
    cycle: int,
    step_id: str,
    question: str,
    options: list[Option] | None = None,
    rationale: str = "",
    saved_state: dict | None = None,
    lab_path: Path | None = None,
    expires_in_secs: int = DEFAULT_EXPIRY_SECS,
) -> NeedsUserInput:
    """Construct a NeedsUserInput envelope + persist the paused state.

    Use this from director / agent code at fork points."""
    state = PausedState(
        lab=lab, cycle=cycle, step_id=step_id,
        saved_state=saved_state or {},
        created_at_ts=time.time(),
        expires_at_ts=time.time() + expires_in_secs,
    )
    token = mint_resume_token(state)
    if lab_path is not None:
        try:
            save_paused_state(lab_path, state)
        except OSError as e:
            LOG.warning("save_paused_state failed: %s", e)
    return NeedsUserInput(
        question=question,
        options=tuple(options or ()),
        rationale=rationale,
        resume_token=token,
    )


# ── CLI ─────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """python -m core.pause_resume mint <lab> <cycle> <step_id>
    python -m core.pause_resume verify <token>
    python -m core.pause_resume list <lab_path>
    """
    import sys
    if len(argv) < 2:
        print("usage: pause_resume mint|verify|list ...", file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "mint":
        if len(argv) < 5:
            print("usage: pause_resume mint <lab> <cycle> <step_id>",
                  file=sys.stderr)
            return 2
        st = PausedState(
            lab=argv[2], cycle=int(argv[3]), step_id=argv[4],
        )
        token = mint_resume_token(st)
        print(token)
        return 0
    if cmd == "verify":
        if len(argv) < 3:
            print("usage: pause_resume verify <token>", file=sys.stderr)
            return 2
        st = verify_resume_token(argv[2])
        if st is None:
            print("INVALID")
            return 1
        print(json.dumps(asdict(st), indent=2))
        return 0
    if cmd == "list":
        if len(argv) < 3:
            print("usage: pause_resume list <lab_path>", file=sys.stderr)
            return 2
        pending = list_pending(Path(argv[2]).expanduser())
        print(json.dumps(pending, indent=2))
        return 0
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv))
