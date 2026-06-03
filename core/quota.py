"""Per-provider quota tracking with SQLite-backed event log.

Used by the router cascade and by core/agent.py before each provider
call to decide whether to dispatch or fall back. Each provider has a
declared free-tier ceiling (RPM, RPD, daily token cap, request-context
cap); this module records every dispatch and answers `check_quota` in
~1ms by counting events in a rolling window.

Free-tier ceilings come from May 2026 validation. Update when
provider-side ceilings change. The quota check is advisory — if a
provider returns 429 anyway, core/provider.py's retry-with-backoff
handles it; quota.py just minimizes 429s by pre-checking.

Schema (lab/state/quota.db):
  events(id, provider, ts, prompt_tokens, completion_tokens,
         status_code, latency_ms)
  probes(id, provider, ts, ok, latency_ms, error)

Both tables are append-only — no updates, no deletes. Pruning of old
rows is a separate maintenance job (P-014 nightly backup includes
quota.db; aging out is fine).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from core import log

LOG = log.get_logger("bert.quota")
LAB_ROOT = Path(__file__).resolve().parent.parent
QUOTA_DB = LAB_ROOT / "lab" / "state" / "quota.db"

_DB_LOCK = Lock()


def _emit_circuit_breaker(provider: str, ceiling: str, observed: int, limit: int) -> None:
    """P-023 circuit-breaker event: fires when a provider's quota
    ceiling is hit (RPM/RPD/daily_tokens). Advisory observability;
    never raises but logs at warning so debug-time the failure is
    visible. Silent failure was a quality-first audit finding —
    never log nothing."""
    try:
        from core import observability as _obs
        _obs.emit("circuit_breaker_event", {
            "provider": provider, "ceiling": ceiling,
            "observed": observed, "limit": limit, "kind": "quota",
        })
    except Exception as e:  # noqa: BLE001
        LOG.warning("quota: circuit_breaker emit failed (advisory): %s", e)


@dataclass(frozen=True)
class ProviderLimits:
    """Free-tier ceilings per provider. None = no enforced ceiling."""
    rpm: int | None = None
    rpd: int | None = None
    daily_tokens: int | None = None
    context_max: int | None = None


# Free-tier ceilings (May 2026). Update when changed.
PROVIDER_LIMITS: dict[str, ProviderLimits] = {
    "nvidia":     ProviderLimits(rpm=40),
    "cerebras":   ProviderLimits(rpm=30, daily_tokens=1_000_000, context_max=8192),
    "groq":       ProviderLimits(rpm=30, rpd=1000),
    "gemini":     ProviderLimits(rpm=60, rpd=1500),
    "mistral":    ProviderLimits(rpm=30),
    "openrouter": ProviderLimits(rpm=20),
    "hf_router":  ProviderLimits(rpm=30),
    "ollama":     ProviderLimits(),
}


def _connect() -> sqlite3.Connection:
    QUOTA_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(QUOTA_DB, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            ts REAL NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER NOT NULL DEFAULT 0,
            status_code INTEGER NOT NULL DEFAULT 0,
            latency_ms INTEGER NOT NULL DEFAULT 0
        )
    """)
    # Forward-compat: existing dbs created before cached_tokens column
    # get an ALTER TABLE on first connect. NOT NULL with DEFAULT 0 is
    # safe to add to an existing table.
    _cols = {r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
    if "cached_tokens" not in _cols:
        conn.execute("ALTER TABLE events ADD COLUMN cached_tokens INTEGER NOT NULL DEFAULT 0")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS probes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            ts REAL NOT NULL,
            ok INTEGER NOT NULL DEFAULT 0,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            error TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_provider_ts ON events(provider, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_probes_provider_ts ON probes(provider, ts)")
    conn.commit()
    return conn


def record_call(
    provider: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    status_code: int = 200,
    latency_ms: int = 0,
) -> None:
    """Log one dispatch. Called by core/provider.py after every HTTP call.

    cached_tokens is provider-reported prefix-cache hit count (Gemini 2.5+
    implicit cache + Groq GPT-OSS automatic cache surface this in
    `prompt_tokens_details.cached_tokens` / `cached_content_token_count`).
    Recording lets the Diagnostics surface show cache-hit % per provider.
    """
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO events(provider, ts, prompt_tokens, completion_tokens, "
            "cached_tokens, status_code, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (provider, time.time(), prompt_tokens, completion_tokens,
             cached_tokens, status_code, latency_ms),
        )


def record_probe(
    provider: str, *, ok: bool, latency_ms: int = 0, error: str | None = None
) -> None:
    """Log one health probe outcome. Called by the 60s scheduler."""
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO probes(provider, ts, ok, latency_ms, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (provider, time.time(), 1 if ok else 0, latency_ms, error),
        )


def check_quota(provider: str, *, prompt_tokens: int = 0) -> tuple[bool, str]:
    """Pre-flight quota check.

    Returns (ok, reason). ok=False means the cascade should skip this
    provider. reason explains which ceiling was hit (or "ok" on success).

    Counts events in rolling windows: 60s for RPM, 86400s for RPD,
    86400s for daily_tokens. Context_max is checked against the
    in-flight prompt_tokens (synchronous; doesn't query the db).
    """
    limits = PROVIDER_LIMITS.get(provider, ProviderLimits())

    if limits.context_max is not None and prompt_tokens > limits.context_max:
        return False, f"context_max {prompt_tokens}>{limits.context_max}"

    now = time.time()
    with _DB_LOCK, _connect() as conn:
        if limits.rpm is not None:
            (rpm_count,) = conn.execute(
                "SELECT COUNT(*) FROM events WHERE provider=? AND ts > ?",
                (provider, now - 60),
            ).fetchone()
            if rpm_count >= limits.rpm:
                _emit_circuit_breaker(provider, "rpm", rpm_count, limits.rpm)
                return False, f"rpm {rpm_count}>={limits.rpm}"

        if limits.rpd is not None:
            (rpd_count,) = conn.execute(
                "SELECT COUNT(*) FROM events WHERE provider=? AND ts > ?",
                (provider, now - 86400),
            ).fetchone()
            if rpd_count >= limits.rpd:
                _emit_circuit_breaker(provider, "rpd", rpd_count, limits.rpd)
                return False, f"rpd {rpd_count}>={limits.rpd}"

        if limits.daily_tokens is not None:
            row = conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) "
                "FROM events WHERE provider=? AND ts > ?",
                (provider, now - 86400),
            ).fetchone()
            tok_used = row[0] if row else 0
            if tok_used + prompt_tokens >= limits.daily_tokens:
                _emit_circuit_breaker(provider, "daily_tokens",
                                      tok_used + prompt_tokens, limits.daily_tokens)
                return False, f"daily_tokens {tok_used + prompt_tokens}>={limits.daily_tokens}"

    return True, "ok"


def stats(provider: str | None = None) -> dict:
    """Roll-up of recent activity. Used by /now page generator and ops scripts."""
    now = time.time()
    out: dict = {}
    with _DB_LOCK, _connect() as conn:
        providers = (
            [provider] if provider
            else [r[0] for r in conn.execute("SELECT DISTINCT provider FROM events").fetchall()]
        )
        for p in providers:
            row60 = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), "
                "COALESCE(SUM(completion_tokens),0), COALESCE(AVG(latency_ms),0) "
                "FROM events WHERE provider=? AND ts > ?",
                (p, now - 60),
            ).fetchone()
            row24h = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), "
                "COALESCE(SUM(completion_tokens),0), "
                "COALESCE(SUM(cached_tokens),0) "
                "FROM events WHERE provider=? AND ts > ?",
                (p, now - 86400),
            ).fetchone()
            err24h = conn.execute(
                "SELECT COUNT(*) FROM events WHERE provider=? AND ts > ? "
                "AND status_code >= 400",
                (p, now - 86400),
            ).fetchone()[0]
            probe24h = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(ok),0) FROM probes "
                "WHERE provider=? AND ts > ?",
                (p, now - 86400),
            ).fetchone()
            limits = PROVIDER_LIMITS.get(p, ProviderLimits())
            prompt_tok_24h = int(row24h[1])
            cached_tok_24h = int(row24h[3])
            # Cache-hit % = cached_tokens / prompt_tokens (cached are a
            # subset of prompt for providers that surface both).
            cache_hit_pct = (
                round(100.0 * cached_tok_24h / prompt_tok_24h, 1)
                if prompt_tok_24h > 0 else 0.0
            )
            # K.6 — cooldown signal for ProviderCooledBadge.
            # If the most recent event from this provider was a 429
            # within the last 60s, mark it cooled. The frontend renders
            # a warm-amber pill ("Groq cooled · NVIDIA warm") — errors-
            # as-weather, not errors-as-failure.
            recent_429 = conn.execute(
                "SELECT ts FROM events WHERE provider=? AND status_code=429 "
                "ORDER BY ts DESC LIMIT 1",
                (p,),
            ).fetchone()
            cooled_until_ts = None
            if recent_429:
                # Groq's typical 429 cooldown is ~60s; broader floor 120s
                # is a safer demo signal that doesn't flicker on/off.
                cooled_until_ts = recent_429[0] + 120
                if cooled_until_ts < now:
                    cooled_until_ts = None

            out[p] = {
                "rpm_60s": row60[0],
                "tokens_60s": int(row60[1] + row60[2]),
                "avg_latency_ms_60s": int(row60[3]),
                "calls_24h": row24h[0],
                "tokens_24h": int(row24h[1] + row24h[2]),
                "cached_tokens_24h": cached_tok_24h,
                "cache_hit_pct_24h": cache_hit_pct,
                "errors_24h": err24h,
                "probes_24h_total": probe24h[0],
                "probes_24h_ok": probe24h[1],
                "cooled_until_ts": cooled_until_ts,
                "limits": {
                    "rpm": limits.rpm, "rpd": limits.rpd,
                    "daily_tokens": limits.daily_tokens,
                    "context_max": limits.context_max,
                },
            }
    return out


def prune_old(days: int = 30) -> int:
    """Delete events + probes older than `days`. Returns rows deleted.
    Called from P-014 nightly backup; not a hot path."""
    cutoff = time.time() - days * 86400
    with _DB_LOCK, _connect() as conn:
        e = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,)).rowcount
        p = conn.execute("DELETE FROM probes WHERE ts < ?", (cutoff,)).rowcount
        conn.commit()
    LOG.info("prune_old days=%d deleted events=%d probes=%d", days, e, p)
    return e + p
