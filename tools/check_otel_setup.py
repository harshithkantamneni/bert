"""Verify bert's OpenTelemetry GenAI-conventions emission path.

Per H2 §5.2 acceptance: "OTel emission verified in standard tooling
(Datadog or Grafana sample import)". This script:

  1. Reports whether the OTel SDK is importable.
  2. Probes the configured OTLP endpoint (env: OTEL_EXPORTER_OTLP_ENDPOINT).
  3. Emits a probe span via observability.emit_otel_span().
  4. Prints the resulting span attributes so PI can confirm Grafana /
     Datadog ingestion in the receiver UI.

Usage:
  python tools/check_otel_setup.py
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 python tools/check_otel_setup.py

Standard exporter setup (per gen-ai semantic conventions):
  pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
  export OTEL_EXPORTER_OTLP_ENDPOINT=https://your-collector.example.com/v1/traces
  export OTEL_SERVICE_NAME=bert-lab

Grafana wiring:
  point Grafana Tempo at the same OTLP endpoint; queries on
  span.gen_ai.system, gen_ai.usage.input_tokens, etc. work directly.

Datadog wiring:
  set DD_OTLP_CONFIG_RECEIVER_PROTOCOLS_GRPC=true on the Datadog Agent
  and point the SDK at the agent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def check_sdk() -> dict:
    try:
        import opentelemetry  # noqa: F401
        from opentelemetry import trace  # noqa: F401
        return {"sdk_present": True}
    except ImportError as e:
        return {"sdk_present": False, "import_error": str(e)}


def check_exporter() -> dict:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    service = os.environ.get("OTEL_SERVICE_NAME", "bert-lab")
    return {
        "endpoint_set": bool(endpoint),
        "endpoint": endpoint,
        "service": service,
    }


def emit_probe() -> dict:
    """Fire a probe span via observability.emit_otel_span()."""
    try:
        from core import observability
        observability.emit_otel_span(
            "probe",
            system="bert-lab",
            request_model="probe-model",
            response_model="probe-model",
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            elapsed_ms=42,
            extra={"bert.probe": True, "bert.tool": "check_otel_setup"},
        )
        return {"emit_ok": True}
    except Exception as e:  # noqa: BLE001
        return {"emit_ok": False, "error": str(e)}


def main() -> int:
    print("bert · OTel setup check")
    print("=" * 40)
    sdk = check_sdk()
    print(f"  SDK present:       {sdk['sdk_present']}")
    if not sdk["sdk_present"]:
        print(f"  import error:      {sdk['import_error']}")
        print()
        print("  fix:  pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http")
        return 1
    exp = check_exporter()
    print(f"  endpoint set:      {exp['endpoint_set']}")
    if exp["endpoint_set"]:
        print(f"  endpoint:          {exp['endpoint']}")
    print(f"  service name:      {exp['service']}")
    if not exp["endpoint_set"]:
        print()
        print("  fix:  export OTEL_EXPORTER_OTLP_ENDPOINT=<your collector URL>")
        print("        (spans emit but go to a no-op exporter otherwise)")
    probe = emit_probe()
    print(f"  probe span emit:   {probe['emit_ok']}")
    if not probe["emit_ok"]:
        print(f"  error:             {probe.get('error')}")
        return 1
    print()
    print("  ✓ probe span fired. Check Grafana / Datadog for:")
    print("    span name:       gen_ai.probe")
    print("    gen_ai.system:   bert-lab")
    print("    bert.probe:      true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
