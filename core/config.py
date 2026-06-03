"""Lab configuration + credentials loader.

Loads ~/.bert-lab/credentials.json (mode 600), env vars, runtime config.
Single source of truth for everything bert needs to authenticate to providers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
CRED_PATH = Path.home() / ".bert-lab" / "credentials.json"


@dataclass
class Config:
    """Runtime configuration."""
    lab_root: Path = LAB_ROOT
    credentials: dict[str, str | None] = field(default_factory=dict)
    permission_mode: str = "default"          # plan | default | auto | dontAsk
    spend_per_mission_cap: int = 5_000_000    # P-012
    spend_per_day_cap: int = 10_000_000       # P-012
    context_usage_cap: float = 0.70           # P-013 — cap usage at 70% of model window
    max_tokens_default: int = 8000            # thinking (Gemini up to ~6k) + reasoning + tool calls all share this budget; gemini-2.5-flash caps at 8192 output
    log_level: str = "INFO"
    telegram_user_id: int | None = None       # set via BERT_LAB_TG_USER_ID

    def has(self, key: str) -> bool:
        return bool(self.credentials.get(key))

    def get(self, key: str) -> str | None:
        return self.credentials.get(key)

    def require(self, key: str) -> str:
        v = self.credentials.get(key)
        if not v:
            raise RuntimeError(
                f"Required credential {key} missing from {CRED_PATH}. "
                f"Available keys: {sorted(k for k, v in self.credentials.items() if v)}"
            )
        return v


_cached: Config | None = None


def load(reload: bool = False) -> Config:
    """Load config from credentials.json + env. Cached after first call."""
    global _cached
    if _cached is not None and not reload:
        return _cached

    creds: dict[str, str | None] = {}
    if CRED_PATH.exists():
        try:
            creds = json.loads(CRED_PATH.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse {CRED_PATH}: {e}") from e

    # Env vars override file values (useful for one-off testing)
    for env_key in ("NVIDIA_API_KEY", "CEREBRAS_API_KEY", "GROQ_API_KEY",
                    "GOOGLE_AI_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY",
                    "HF_TOKEN", "TELEGRAM_BOT_TOKEN"):
        if os.environ.get(env_key):
            creds[env_key] = os.environ[env_key]

    # BB.7 — backward-compat alias for Google AI Studio. Google uses
    # `GOOGLE_API_KEY` as the conventional env var name for AI Studio;
    # bert's provider registry was written with `GOOGLE_AI_API_KEY` for
    # disambiguation. Accept either. If GOOGLE_API_KEY is set and
    # GOOGLE_AI_API_KEY is not, alias the former to the latter.
    if os.environ.get("GOOGLE_API_KEY") and not creds.get("GOOGLE_AI_API_KEY"):
        creds["GOOGLE_AI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

    cfg = Config(
        credentials=creds,
        permission_mode=os.environ.get("BERT_PERMISSION_MODE", "default"),
        log_level=os.environ.get("BERT_LOG_LEVEL", "INFO"),
        telegram_user_id=(int(os.environ["BERT_LAB_TG_USER_ID"])
                          if os.environ.get("BERT_LAB_TG_USER_ID") else None),
    )
    _cached = cfg
    return cfg


def credential_keys_present() -> list[str]:
    cfg = load()
    return [k for k, v in cfg.credentials.items() if v]
