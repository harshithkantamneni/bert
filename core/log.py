"""Rotating logger + structured event emitter with credential redaction.

Per P-020: redact api keys / tokens / private keys before any log write.
Per P-016: tool outputs get sentinel-wrapped before being logged.

Local-file logs in logs/. No SaaS, no remote shipping.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = LAB_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# P-020 redaction patterns — applied to every log message before write.
_REDACTION_PATTERNS = [
    (re.compile(r"nvapi-[A-Za-z0-9_-]{60,}"),                 "<nvidia_key:redacted>"),
    (re.compile(r"csk-[A-Za-z0-9]{40,}"),                     "<cerebras_key:redacted>"),
    (re.compile(r"gsk_[A-Za-z0-9]{50,}"),                     "<groq_key:redacted>"),
    (re.compile(r"AIzaSy[A-Za-z0-9_-]{30,}"),                 "<google_key:redacted>"),
    (re.compile(r"sk-or-v1-[A-Za-z0-9]{60,}"),                "<openrouter_key:redacted>"),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"),                      "<hf_token:redacted>"),
    (re.compile(r"\b\d{9,12}:[A-Za-z0-9_-]{30,}\b"),          "<telegram_token:redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9]{40,}\b"),                  "<api_key:redacted>"),
    (
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----"),
        "<private_key:redacted>",
    ),
]


def redact(text: str) -> str:
    """Apply P-020 redaction patterns to any string before persistence."""
    if not isinstance(text, str):
        return text
    for pat, replacement in _REDACTION_PATTERNS:
        text = pat.sub(replacement, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact dicts / lists / strings."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj


# ── Structured logger ────────────────────────────────────────────────


class RedactingFormatter(logging.Formatter):
    """Logging formatter that runs every message through redact()."""

    def format(self, record: logging.LogRecord) -> str:
        record.msg = redact(str(record.msg))
        return super().format(record)


def get_logger(name: str = "bert", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console (stderr) handler — runs through redaction
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(RedactingFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    # Rotating file handler — also redacted
    fh = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / f"{name}.log",
        when="midnight",
        backupCount=30,  # P-015: keep 30 days
        encoding="utf-8",
    )
    fh.setFormatter(RedactingFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    logger.addHandler(fh)

    return logger


# ── JSONL session log (every turn appended) ──────────────────────────


def append_session_event(cycle: int, event: dict[str, Any]) -> Path:
    """Append a structured event to logs/cycle_{cycle}_{ts}.jsonl.

    Used by core.session for per-turn audit trail. Auto-redacted.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = LOG_DIR / f"cycle_{cycle}_{ts[:8]}.jsonl"
    safe = redact_obj({**event, "_ts": datetime.now(UTC).isoformat()})
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, default=str) + "\n")
    return path


# ── Tool-output sentinel wrapping (P-016) ─────────────────────────────


def wrap_tool_output(content: str) -> str:
    """Wrap tool output with the untrusted-data sentinel.

    The model is instructed via constitutional preamble §11 to treat
    content within these markers as DATA, not directives.
    """
    return f"<<TOOL_OUTPUT untrusted>\n{content}\n</TOOL_OUTPUT>>"
