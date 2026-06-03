"""Observability layer — bert-specific JSONL + OTel GenAI dual-emission.

Two emission paths, both optional and lazy:

  1. bert-specific JSONL at state/observability/{event_class}.jsonl
     — rich per-event detail for in-lab calibration queries

  2. OpenTelemetry GenAI semantic-convention spans per the official
     spec at https://opentelemetry.io/docs/specs/semconv/gen-ai/ —
     vendor-neutral interop with Datadog, Grafana Tempo, Honeycomb,
     LangFuse, Helicone, etc.

The two emit-paths are non-redundant: OTel spans use the standardized
gen_ai.* namespace (gen_ai.system, gen_ai.request.model,
gen_ai.usage.input_tokens, gen_ai.operation.name, ...) which is enough
for vendor dashboards but doesn't carry bert-specific calibration
state. The JSONL has the richer fields (position_swap_delta,
severity_grade, threshing summaries, etc.) that bert's own
calibration queries need.

Emission is best-effort: a write failure logs a warning and continues
without re-raising. Observability failure must NOT kill a cycle.

OTel SDK is an optional dependency. If opentelemetry-api/sdk isn't
installed, emit_otel_span is a no-op (logged once). bert ships with
JSONL emission unconditionally; OTel is opt-in.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

from core import log

LAB_ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = LAB_ROOT / "state" / "observability"

# Size at which a per-event_class JSONL file is archived. Tuned for
# ~50K typical events per file before rotation; total disk impact
# bounded since archives go to cold storage.
ROTATION_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10 MB
LOG = log.get_logger("bert.observability")

# Lazy OTel import — only fires if user has installed opentelemetry-{api,sdk}
_otel_tracer = None
_otel_warned = False


def _get_otel_tracer():
    """Return the OTel tracer, configuring the SDK + OTLP exporter on
    first call when OTEL_EXPORTER_OTLP_ENDPOINT is set.

    When the endpoint env var is unset, returns the API's default
    (no-op) tracer — spans are created but discarded. This keeps the
    cost of emit_otel_span() at ~free in dev without an OTel backend.

    When the endpoint IS set, we install a BatchSpanProcessor pointed
    at it via the OTLP HTTP exporter. One-time SDK setup is idempotent;
    subsequent calls reuse the cached tracer.
    """
    global _otel_tracer, _otel_warned
    if _otel_tracer is not None:
        return _otel_tracer
    try:
        from opentelemetry import trace  # type: ignore
    except ImportError:
        if not _otel_warned:
            LOG.info(
                "opentelemetry SDK not installed; OTel emission is a no-op. "
                "Install opentelemetry-api + opentelemetry-sdk to enable "
                "vendor-neutral observability."
            )
            _otel_warned = True
        return None

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource  # type: ignore
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

            service = os.environ.get("OTEL_SERVICE_NAME", "bert-lab")
            # Honeycomb/Jaeger/Tempo all accept /v1/traces; some collectors
            # want it explicit. If the endpoint already has the path, leave it.
            traces_url = endpoint.rstrip("/")
            if not traces_url.endswith("/v1/traces"):
                traces_url = traces_url + "/v1/traces"
            headers_str = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
            headers: dict[str, str] = {}
            for pair in headers_str.split(","):
                if "=" in pair:
                    k, _, v = pair.partition("=")
                    headers[k.strip()] = v.strip()
            exporter = OTLPSpanExporter(endpoint=traces_url,
                                         headers=headers or None)
            provider = TracerProvider(resource=Resource.create({
                "service.name": service,
                "service.version": "0.1",
            }))
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            LOG.info("OTel tracer configured: endpoint=%s service=%s",
                     traces_url, service)
        except Exception as e:  # noqa: BLE001
            if not _otel_warned:
                LOG.warning("OTel SDK setup failed (%s); falling back to no-op", e)
                _otel_warned = True
    _otel_tracer = trace.get_tracer("bert.observability", "1.0")
    return _otel_tracer


# ── Background-tool invocation tracking (v3+ Phase 1d) ──────────────


def emit_cycle_outcome(
    cycle_id: int,
    *,
    lab: str | None = None,
    success: bool,
    elapsed_secs: float | None = None,
    dispatches_total: int = 0,
    dispatches_valid: int = 0,
    verdicts: list[str] | None = None,
    findings_produced: int = 0,
    artifacts_accepted: int = 0,
    concerns_raised: int = 0,
    concerns_resolved: int = 0,
    extra: dict | None = None,
) -> None:
    """Emit a cycle_outcome rollup event capturing the per-cycle bottom
    line. Per v3+ Phase 1d — the missing downstream-causality signal
    that lets us evaluate retrieval utility (did the cycle produce
    accepted output? did it stall? did it accumulate concerns?).

    The data needed is already scattered across verdict.jsonl,
    subagent_finish.jsonl, artifact_accepted.jsonl, concern_*.jsonl —
    this is the aggregation point that makes per-cycle analysis cheap."""
    payload = {
        "cycle_id": cycle_id,
        "lab": lab,
        "success": success,
        "elapsed_secs": elapsed_secs,
        "dispatches": {
            "total": dispatches_total,
            "valid": dispatches_valid,
            "invalid": max(0, dispatches_total - dispatches_valid),
        },
        "verdicts": verdicts or [],
        "findings_produced": findings_produced,
        "artifacts_accepted": artifacts_accepted,
        "concerns": {
            "raised": concerns_raised,
            "resolved": concerns_resolved,
            "open": max(0, concerns_raised - concerns_resolved),
        },
    }
    if extra:
        payload["extra"] = extra
    try:
        emit("cycle_outcome", payload)
    except Exception as e:  # noqa: BLE001
        LOG.debug("cycle_outcome emit skipped (advisory): %s", e)


def emit_background_invocation(
    tool_name: str,
    *,
    args: dict | None = None,
    duration_ms: float | None = None,
    findings_produced: list[str] | None = None,
    success: bool = True,
    extra: dict | None = None,
) -> None:
    """Emit a background_invocation event for tools that produce
    findings outside the cycle-agent path (falsifier_baseline.py,
    weekly_quality_report.py, daily_quality_report.py, etc.).

    Per v3+ analysis: bert has TWO emission paths — cycle agents
    (instrumented via tool_call.jsonl) + background tools (not
    instrumented). This closes that observability gap.

    Best-effort; failures don't block the tool's actual work."""
    payload = {
        "tool": tool_name,
        "args": args or {},
        "duration_ms": duration_ms,
        "findings_produced": findings_produced or [],
        "success": success,
    }
    if extra:
        payload["extra"] = extra
    try:
        emit("background_invocation", payload)
    except Exception as e:  # noqa: BLE001
        LOG.debug("background_invocation emit skipped (advisory): %s", e)


# ── bert-specific JSONL emission ────────────────────────────────────


def emit(event_class: str, payload: dict[str, Any]) -> None:
    """Append `payload` to state/observability/{event_class}.jsonl with a
    UTC timestamp prefix.

    event_class is one of the 14 enum values:

      Wired and live (13 of 14):
        tool_call                       — agent.py per tool_call
        model_call                      — agent.py per provider response
        subagent_spawn / subagent_finish — subagent.run_subagent enter/exit
        verdict                         — subagent.run_subagent post-validate
        memory_write                    — agent.py for Write into memories/findings
        threshing_dispatch              — subagent for role=threshing_pass
        clearness_phase1_dispatch       — subagent for role=clearness_phase1
        clearness_phase2_dispatch       — subagent for role=clearness_phase2
        seasoning_entry                 — seasoning.season() helper
        seasoning_revive                — seasoning.revive() helper
        stand_aside_verdict             — subagent for verdict=APPROVE_WITH_CAVEATS
        circuit_breaker_event           — provider.py + quota.py on quota / retry-exhaust
        calibration_falsifier_check     — tools/falsifier_baseline.py per run

      Reserved for future implementation:
        position_swap_event             — P-VS-10 dual-judge (procedures.md
                                          P-VS-10 is FROZEN but the comparison-
                                          mode phase-2 dispatch hasn't landed;
                                          when it does, fire this event with
                                          {position_swap_delta, swap_a, swap_b})

    Best-effort: write failures log a warning, don't re-raise. The
    observability subsystem must NOT kill a cycle.
    """
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    path = OBS_DIR / f"{event_class}.jsonl"
    # Quality-first: rotate when this event_class file exceeds the
    # threshold. Keeps live event store bounded; archived files stay
    # readable for retrospective audit but don't block live writes.
    _maybe_rotate(path)
    record = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "event_class": event_class,
        **payload,
    }
    # Sprint 1 commit 7 (v1.0 Round-2 C-1 fix): atomic concurrent writes.
    # With async cycle runner (Sprint 4), multiple processes may emit
    # to the same JSONL concurrently. POSIX guarantees atomic appends
    # for writes ≤ PIPE_BUF (4096 bytes) when the file is opened with
    # O_APPEND. We use raw os.write + low-level fd to bypass Python's
    # io buffering layer (which can split a single write into multiple
    # syscalls). For lines > PIPE_BUF we fall back to an advisory lock.
    line = json.dumps(record, separators=(",", ":")) + "\n"
    line_bytes = line.encode("utf-8")
    _PIPE_BUF = 4096
    try:
        # Open with O_APPEND so concurrent writes don't truncate each other
        fd = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            if len(line_bytes) <= _PIPE_BUF:
                # Single atomic write (POSIX-guaranteed for ≤PIPE_BUF with O_APPEND)
                os.write(fd, line_bytes)
            else:
                # Oversized event — take an advisory lock to prevent interleaving
                try:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    try:
                        # Write in PIPE_BUF chunks; lock guarantees no interleave
                        offset = 0
                        while offset < len(line_bytes):
                            n = os.write(fd, line_bytes[offset:offset + _PIPE_BUF])
                            if n <= 0:
                                break
                            offset += n
                    finally:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                except (ImportError, OSError):
                    # fcntl unavailable (e.g., Windows) — best-effort single write
                    os.write(fd, line_bytes)
                # Log oversized events for review (likely indicates a
                # bloated payload that should be reduced)
                LOG.debug(
                    "observability: oversized event %s (%d bytes > %d PIPE_BUF)",
                    event_class, len(line_bytes), _PIPE_BUF,
                )
        finally:
            os.close(fd)
    except OSError as e:
        LOG.warning("observability JSONL write failed (%s): %s", path, e)

    # Live canvas mirror: enrich + append to lab/sor/events.jsonl in a
    # background thread so the canvas L0 SSE picks up this event with
    # content-aware tags. Best-effort, never raises. See core/canvas_emit
    # for the full design rationale.
    try:
        from . import canvas_emit  # local import to keep cold-start fast
        canvas_emit.emit_canvas_event(event_class, record)
    except Exception as e:  # noqa: BLE001
        LOG.debug("canvas mirror skipped (advisory): %s", e)


# ── OpenTelemetry GenAI semantic-convention emission ────────────────


def emit_otel_span(
    operation: str,
    *,
    system: str | None = None,
    request_model: str | None = None,
    response_model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_tokens: int | None = None,
    thinking_tokens: int | None = None,
    elapsed_ms: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit an OpenTelemetry GenAI-conventions span. No-op if OTel SDK
    is not installed.

    Per https://opentelemetry.io/docs/specs/semconv/gen-ai/ — uses the
    gen_ai.* attribute namespace:
      gen_ai.system           — provider name (anthropic, openai, etc.)
      gen_ai.operation.name   — chat / embeddings / etc.
      gen_ai.request.model    — model identifier
      gen_ai.response.model   — actual served model
      gen_ai.usage.input_tokens
      gen_ai.usage.output_tokens
      gen_ai.usage.cached_tokens   (custom; standardization in flight)
      gen_ai.usage.reasoning_tokens (custom; for thinking models)
    """
    tracer = _get_otel_tracer()
    if tracer is None:
        return
    try:
        with tracer.start_as_current_span(f"gen_ai.{operation}") as span:
            if system:
                span.set_attribute("gen_ai.system", system)
            span.set_attribute("gen_ai.operation.name", operation)
            if request_model:
                span.set_attribute("gen_ai.request.model", request_model)
            if response_model:
                span.set_attribute("gen_ai.response.model", response_model)
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            if output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if cached_tokens is not None:
                span.set_attribute("gen_ai.usage.cached_tokens", cached_tokens)
            if thinking_tokens is not None:
                span.set_attribute("gen_ai.usage.reasoning_tokens", thinking_tokens)
            if elapsed_ms is not None:
                span.set_attribute("gen_ai.client.duration", elapsed_ms / 1000.0)
            if extra:
                for k, v in extra.items():
                    span.set_attribute(k, v)
    except Exception as e:
        LOG.warning("OTel span emit failed: %s", e)


# ── Convenience: dual-emit a model_call event ───────────────────────


def emit_model_call(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    thinking_tokens: int = 0,
    elapsed_ms: int = 0,
    role: str | None = None,
    cycle: int | None = None,
) -> None:
    """Helper: emit both bert-JSONL and OTel-span for a single model_call
    event. Use from core/agent.py + core/subagent.py.
    """
    emit("model_call", {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "thinking_tokens": thinking_tokens,
        "elapsed_ms": elapsed_ms,
        "role": role,
        "cycle": cycle,
    })
    emit_otel_span(
        "chat",
        system=provider,
        request_model=model,
        response_model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        thinking_tokens=thinking_tokens,
        elapsed_ms=elapsed_ms,
        extra={"bert.role": role, "bert.cycle": cycle} if role or cycle else None,
    )


# ── Calibration query helpers ───────────────────────────────────────


# ── Rotation helpers ────────────────────────────────────────────────


def _maybe_rotate(path: Path) -> bool:
    """If `path` exceeds ROTATION_THRESHOLD_BYTES, archive it to
    state/observability/archive/<YYYY-MM-DD>/<name>_<n>.jsonl and
    let the next emit() create a fresh file. Returns True if rotation
    fired.

    The 'archive' subdirectory is excluded from emit() lookups (event
    files are top-level only). Readers that want full history walk
    the archive explicitly via `read_archived(event_class)`.
    """
    try:
        if not path.exists() or path.stat().st_size < ROTATION_THRESHOLD_BYTES:
            return False
    except OSError:
        return False
    archive_root = OBS_DIR / "archive"
    today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    archive_dir = archive_root / today
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        LOG.warning("observability rotate: cannot create archive dir: %s", e)
        return False
    # Pick a non-colliding name in the archive
    base = path.stem  # e.g., "model_call"
    n = 1
    while True:
        target = archive_dir / f"{base}_{n}.jsonl"
        if not target.exists():
            break
        n += 1
    try:
        path.rename(target)
        LOG.info("observability rotate: %s → %s (%.1f MB)",
                 path.name, target.relative_to(OBS_DIR),
                 target.stat().st_size / (1024 * 1024))
        return True
    except OSError as e:
        LOG.warning("observability rotate: rename failed (%s); leaving file in place", e)
        return False


def read_archived(event_class: str) -> list[dict]:
    """Read every archived JSONL for `event_class` across all date
    subdirs under state/observability/archive/. Used by retrospective
    audit / falsifier replay."""
    archive_root = OBS_DIR / "archive"
    if not archive_root.exists():
        return []
    out: list[dict] = []
    for archive_dir in sorted(archive_root.iterdir()):
        if not archive_dir.is_dir():
            continue
        for p in sorted(archive_dir.glob(f"{event_class}_*.jsonl")):
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue
    return out


def rotate_all(threshold_bytes: int | None = None) -> dict:
    """Force-rotate every event file at or above the threshold. Used
    by tools/nightly_backup.sh + consolidator periodic maintenance.
    Returns a dict {event_class: rotated_bool}."""
    global ROTATION_THRESHOLD_BYTES
    out: dict[str, bool] = {}
    if not OBS_DIR.exists():
        return out
    threshold = threshold_bytes if threshold_bytes is not None else ROTATION_THRESHOLD_BYTES
    old = ROTATION_THRESHOLD_BYTES
    ROTATION_THRESHOLD_BYTES = threshold
    try:
        for p in OBS_DIR.iterdir():
            if not p.is_file() or p.suffix != ".jsonl":
                continue
            out[p.stem] = _maybe_rotate(p)
    finally:
        ROTATION_THRESHOLD_BYTES = old
    return out


def calibration_count(event_class: str, predicate: dict[str, Any] | None = None) -> int:
    """Count events of a class matching simple key=value predicates.

    Used by falsifier checks. e.g.,
      calibration_count('threshing_dispatch')  # how many threshing dispatches?
      calibration_count('verdict', {'verdict': 'SCOPE_STOP'})  # how many SCOPE_STOPs?
    """
    path = OBS_DIR / f"{event_class}.jsonl"
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if predicate is None or all(rec.get(k) == v for k, v in predicate.items()):
                    count += 1
    except (OSError, json.JSONDecodeError) as e:
        LOG.warning("calibration_count read failed: %s", e)
    return count


# ── CLI ─────────────────────────────────────────────────────────────


def _cli() -> int:
    """`python -m core.observability calibration` — print calibration summary."""
    import sys
    if len(sys.argv) < 2 or sys.argv[1] != "calibration":
        print("usage: python -m core.observability calibration", file=sys.stderr)
        return 1
    print("=== bert observability calibration ===")
    print(f"observability dir: {OBS_DIR}")
    if not OBS_DIR.exists():
        print("(no observability data yet)")
        return 0
    for f in sorted(OBS_DIR.glob("*.jsonl")):
        count = sum(1 for _ in f.open())
        print(f"  {f.name}: {count} events")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
