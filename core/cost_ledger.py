"""Per-model_call cost ledger (Sprint 4 C2, launch criteria 22 + 25).

Appends one priced row per provider response to
`state/observability/cost.jsonl`. Free-tier providers cost $0 but their
tokens are still tracked for quota + attribution; host/BYO models are priced
from `core/library/model_prices.yaml`. Thinking tokens are tracked as a
distinct column (criterion 25). `summarize()` rolls up per (provider, model).

Wired into `core.provider.call`'s success path (best-effort; never breaks the
call). The cost-estimate-with-CI surfaced to users (criterion 22) reads these
rows via `core.cost_estimator`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

LOG = logging.getLogger("bert.cost_ledger")
LAB_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = LAB_ROOT / "state" / "observability" / "cost.jsonl"
PRICES_PATH = LAB_ROOT / "core" / "library" / "model_prices.yaml"

_prices_cache: dict | None = None


def _load_prices() -> dict:
    global _prices_cache
    if _prices_cache is not None:
        return _prices_cache
    try:
        import yaml
        _prices_cache = yaml.safe_load(PRICES_PATH.read_text()) or {}
    except Exception as e:  # noqa: BLE001
        LOG.warning("cost_ledger: prices load failed (%s); treating all as free", e)
        _prices_cache = {}
    return _prices_cache


def _price_for(provider: str, model: str) -> tuple[float, float]:
    """(input_per_1k, output_per_1k) USD. Model-specific price wins; else
    provider-tier price; else the free default."""
    p = _load_prices()
    models = p.get("models") or {}
    if model in models:
        m = models[model]
        return (float(m.get("input", 0.0)), float(m.get("output", 0.0)))
    providers = p.get("providers") or {}
    if provider in providers:
        pr = providers[provider]
        return (float(pr.get("input", 0.0)), float(pr.get("output", 0.0)))
    d = p.get("default") or {}
    return (float(d.get("input", 0.0)), float(d.get("output", 0.0)))


def record(*, provider: str, model: str, input_tokens: int, output_tokens: int,
           cached_tokens: int = 0, thinking_tokens: int = 0,
           lab: str | None = None, cycle: int | None = None) -> dict:
    """Append one priced ledger row. Returns the row."""
    in_price, out_price = _price_for(provider, model)
    usd = round(input_tokens / 1000 * in_price + output_tokens / 1000 * out_price, 6)
    row = {
        "ts": time.time(),
        "lab": lab,
        "cycle": cycle,
        "provider": provider,
        "model": model,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cached_tokens": int(cached_tokens or 0),
        "thinking_tokens": int(thinking_tokens or 0),
        "usd_estimate": usd,
    }
    try:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError as e:
        LOG.warning("cost_ledger: write failed: %s", e)
    return row


def summarize(*, since_ts: float | None = None) -> dict:
    """Roll up the ledger: totals + per-(provider/model) breakdown."""
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
              "cached_tokens": 0, "thinking_tokens": 0, "usd_estimate": 0.0}
    by_model: dict[str, dict] = {}
    if not LEDGER_PATH.exists():
        return {"totals": totals, "by_model": by_model}
    for line in LEDGER_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since_ts is not None and r.get("ts", 0) < since_ts:
            continue
        totals["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cached_tokens", "thinking_tokens"):
            totals[k] += int(r.get(k, 0) or 0)
        totals["usd_estimate"] = round(totals["usd_estimate"] + float(r.get("usd_estimate", 0.0)), 6)
        key = f"{r.get('provider', '?')}/{r.get('model', '?')}"
        m = by_model.setdefault(key, {"calls": 0, "input_tokens": 0,
                                      "output_tokens": 0, "usd_estimate": 0.0})
        m["calls"] += 1
        m["input_tokens"] += int(r.get("input_tokens", 0) or 0)
        m["output_tokens"] += int(r.get("output_tokens", 0) or 0)
        m["usd_estimate"] = round(m["usd_estimate"] + float(r.get("usd_estimate", 0.0)), 6)
    return {"totals": totals, "by_model": by_model}


def _iter_rows(since_ts: float | None = None):
    if not LEDGER_PATH.exists():
        return
    for line in LEDGER_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since_ts is not None and r.get("ts", 0) < since_ts:
            continue
        yield r


def history_usd(*, provider: str | None = None, model: str | None = None,
                lab: str | None = None, since_ts: float | None = None) -> list[float]:
    """Per-call USD history, optionally filtered — the `history` list that
    `core.cost_estimator.estimate(point_usd, history=...)` consumes to compute
    a confidence interval (launch criterion 22). This is the wire between the
    ledger and the estimator: `estimate(point, history=history_usd(...))`.
    """
    out: list[float] = []
    for r in _iter_rows(since_ts):
        if provider is not None and r.get("provider") != provider:
            continue
        if model is not None and r.get("model") != model:
            continue
        if lab is not None and r.get("lab") != lab:
            continue
        out.append(float(r.get("usd_estimate", 0.0)))
    return out


def cache_hit_rate(*, since_ts: float | None = None) -> float:
    """Prompt-cache hit rate = sum(cached_tokens) / sum(input_tokens) across
    ledger rows (launch criterion 12's metric). Returns 0.0 with no rows."""
    cached = prompt = 0
    for r in _iter_rows(since_ts):
        cached += int(r.get("cached_tokens", 0) or 0)
        prompt += int(r.get("input_tokens", 0) or 0)
    return cached / prompt if prompt else 0.0
