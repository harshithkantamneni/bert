"""Smoke test for OTel SDK wiring in core/observability.py.

Per F.6 follow-up. Each test runs in a fresh subprocess because
opentelemetry.trace.set_tracer_provider() can only be called once
per process — running multiple scenarios in one Python interpreter
gives false negatives.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def _run_subprocess(env_extra: dict, code: str) -> tuple[int, str, str]:
    env = {**os.environ, **env_extra}
    env.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    for k, v in env_extra.items():
        env[k] = v
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True,
        # 45s (was 20s): a cold python start that imports core.observability
        # + the opentelemetry SDK pays a one-time page-cache disk read that
        # can exceed 20s on a disk-pressured macOS (96% full). Warm runs are
        # ~0.1s. The 20s ceiling made this test flaky as the FIRST otel
        # subprocess after a heavy eval. Functionality is unaffected.
        cwd=str(LAB_ROOT), timeout=45,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_no_endpoint_returns_noop_tracer() -> None:
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from core import observability\n"
        "from opentelemetry import trace\n"
        "tracer = observability._get_otel_tracer()\n"
        "assert tracer is not None\n"
        "provider = trace.get_tracer_provider()\n"
        "# Proxy / no-op provider lacks add_span_processor\n"
        "assert not hasattr(provider, 'add_span_processor'), type(provider).__name__\n"
        "print('OK')\n"
    )
    rc, out, err = _run_subprocess({}, code)
    assert rc == 0, f"stdout={out!r} stderr={err!r}"
    assert "OK" in out


def test_endpoint_installs_sdk_provider() -> None:
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from core import observability\n"
        "from opentelemetry import trace\n"
        "tracer = observability._get_otel_tracer()\n"
        "assert tracer is not None\n"
        "provider = trace.get_tracer_provider()\n"
        "assert hasattr(provider, 'add_span_processor'), type(provider).__name__\n"
        "print('OK')\n"
    )
    rc, out, err = _run_subprocess(
        {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}, code,
    )
    assert rc == 0, f"stdout={out!r} stderr={err!r}"
    assert "OK" in out


def test_emit_otel_span_doesnt_raise_when_endpoint_unset() -> None:
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from core import observability\n"
        "observability.emit_otel_span('test', system='bert-lab',\n"
        "    request_model='m', input_tokens=10, output_tokens=5)\n"
        "print('OK')\n"
    )
    rc, out, err = _run_subprocess({}, code)
    assert rc == 0, f"stdout={out!r} stderr={err!r}"
    assert "OK" in out


def test_emit_otel_span_with_endpoint_set() -> None:
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from core import observability\n"
        "observability.emit_otel_span('model_call', system='bert-lab',\n"
        "    request_model='meta/llama-3.3-70b-instruct',\n"
        "    input_tokens=1024, output_tokens=256, cached_tokens=900,\n"
        "    elapsed_ms=2340)\n"
        "print('OK')\n"
    )
    rc, out, err = _run_subprocess(
        {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}, code,
    )
    assert rc == 0, f"stdout={out!r} stderr={err!r}"
    assert "OK" in out


def main() -> int:
    tests = [
        test_no_endpoint_returns_noop_tracer,
        test_endpoint_installs_sdk_provider,
        test_emit_otel_span_doesnt_raise_when_endpoint_unset,
        test_emit_otel_span_with_endpoint_set,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
