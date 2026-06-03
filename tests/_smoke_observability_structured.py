"""Smoke test for H2 day 2 — observability + structured_output.

Per FINAL_implementation_plan_2026-05-07.md §5.2 H2 day 2.

Verifies:
  observability:
    1. emit() appends to JSONL with timestamp
    2. emit_model_call dual-emits (JSONL + OTel no-op when SDK absent)
    3. calibration_count counts matching events
    4. emit handles missing dirs (creates state/observability/)
  structured_output:
    5. build_response_format returns OAI-compat shape for NVIDIA / Cerebras
       / Groq / Mistral / Cloudflare / OpenRouter / hf_router
    6. Gemini path returns response_schema (strips $schema, $defs, allOf)
    7. Ollama path returns `format` parameter
    8. Anthropic / OpenAI return empty dict (out of scope)
    9. supports_structured_output reports correctly per provider
"""

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import observability, structured_output  # noqa: E402


# ── Observability tests ─────────────────────────────────────────────


def _isolated_obs_dir(monkey_attr_path: tuple) -> Path:
    """Replace OBS_DIR with a temp dir; return the temp Path."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_obs_"))
    obj = observability
    for attr in monkey_attr_path[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, monkey_attr_path[-1], tmp)
    return tmp


def test_emit_creates_dir_and_appends_jsonl() -> None:
    tmp = _isolated_obs_dir(("OBS_DIR",))
    try:
        observability.emit("test_event", {"foo": 1, "bar": "baz"})
        path = tmp / "test_event.jsonl"
        assert path.exists()
        line = path.read_text().strip()
        rec = json.loads(line)
        assert rec["event_class"] == "test_event"
        assert rec["foo"] == 1
        assert rec["bar"] == "baz"
        assert "ts" in rec
    finally:
        # Restore the original OBS_DIR so subsequent tests see real dir
        observability.OBS_DIR = LAB_ROOT / "state" / "observability"


def test_emit_appends_multiple() -> None:
    tmp = _isolated_obs_dir(("OBS_DIR",))
    try:
        for i in range(3):
            observability.emit("e", {"i": i})
        lines = (tmp / "e.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        recs = [json.loads(l) for l in lines]
        assert [r["i"] for r in recs] == [0, 1, 2]
    finally:
        observability.OBS_DIR = LAB_ROOT / "state" / "observability"


def test_calibration_count() -> None:
    tmp = _isolated_obs_dir(("OBS_DIR",))
    try:
        observability.emit("verdict", {"verdict": "APPROVE", "role": "evaluator"})
        observability.emit("verdict", {"verdict": "SCOPE_STOP", "role": "threshing_pass"})
        observability.emit("verdict", {"verdict": "SCOPE_STOP", "role": "clearness_phase1"})
        assert observability.calibration_count("verdict") == 3
        assert observability.calibration_count("verdict", {"verdict": "SCOPE_STOP"}) == 2
        assert observability.calibration_count("verdict", {"role": "evaluator"}) == 1
        assert observability.calibration_count("nonexistent_class") == 0
    finally:
        observability.OBS_DIR = LAB_ROOT / "state" / "observability"


def test_emit_model_call_dual_emit() -> None:
    """emit_model_call writes JSONL even when OTel is no-op."""
    tmp = _isolated_obs_dir(("OBS_DIR",))
    try:
        observability.emit_model_call(
            provider="cerebras", model="llama3.1-8b",
            input_tokens=1500, output_tokens=200, cached_tokens=0,
            thinking_tokens=80, elapsed_ms=820, role="researcher", cycle=8,
        )
        path = tmp / "model_call.jsonl"
        assert path.exists()
        rec = json.loads(path.read_text().strip())
        assert rec["provider"] == "cerebras"
        assert rec["model"] == "llama3.1-8b"
        assert rec["input_tokens"] == 1500
        assert rec["thinking_tokens"] == 80
    finally:
        observability.OBS_DIR = LAB_ROOT / "state" / "observability"


# ── Structured output tests ─────────────────────────────────────────


SAMPLE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "test/sample",
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["APPROVE", "REJECT"]},
    },
    "required": ["verdict"],
    "$defs": {"X": {"type": "string"}},
    "allOf": [{"if": {}, "then": {}}],
}


def test_oai_compat_provider_returns_response_format() -> None:
    for p in ("nvidia", "cerebras", "groq", "mistral", "openrouter",
              "cloudflare", "hf_router"):
        out = structured_output.build_response_format(p, SAMPLE_SCHEMA)
        assert "response_format" in out, f"provider={p}: missing response_format"
        rf = out["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"]["type"] == "object"


def test_gemini_returns_response_schema_stripped() -> None:
    out = structured_output.build_response_format("gemini", SAMPLE_SCHEMA)
    assert "response_schema" in out
    sch = out["response_schema"]
    # Stripped fields
    for stripped in ("$schema", "$id", "$defs", "allOf"):
        assert stripped not in sch, f"gemini schema should strip {stripped}"
    # Kept fields
    assert sch["type"] == "object"
    assert "properties" in sch


def test_ollama_returns_format_param() -> None:
    out = structured_output.build_response_format("ollama", SAMPLE_SCHEMA)
    assert "format" in out
    assert out["format"]["type"] == "object"


def test_anthropic_returns_empty_out_of_scope() -> None:
    """Anthropic + OpenAI paid are out of strict-free-tier scope."""
    out_a = structured_output.build_response_format("anthropic", SAMPLE_SCHEMA)
    out_o = structured_output.build_response_format("openai", SAMPLE_SCHEMA)
    assert out_a == {}
    assert out_o == {}


def test_supports_structured_output() -> None:
    for p in ("nvidia", "cerebras", "groq", "mistral", "gemini",
              "ollama", "openrouter", "cloudflare", "hf_router"):
        assert structured_output.supports_structured_output(p), p
    for p in ("anthropic", "openai", "unknown_provider"):
        assert not structured_output.supports_structured_output(p), p


def test_rotation_archives_oversized_jsonl() -> None:
    """Quality-first scaling: when an event_class JSONL exceeds the
    rotation threshold, _maybe_rotate moves it to archive/<date>/
    and lets emit() open a fresh file. read_archived() walks the
    archive for retrospective audit."""
    tmp = _isolated_obs_dir(("OBS_DIR",))
    try:
        big = tmp / "model_call.jsonl"
        big.write_text('{"x":"y"}\n' * 1_100_000)  # ~10.5 MB > 10 MB threshold
        assert big.stat().st_size > observability.ROTATION_THRESHOLD_BYTES
        result = observability.rotate_all()
        assert result.get("model_call") is True
        assert not big.exists(), "live file should have been archived"
        # Archive directory created with today's date
        import datetime
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        archived = tmp / "archive" / today
        files = list(archived.glob("model_call_*.jsonl"))
        assert len(files) == 1
        # read_archived reads back the archived events
        rec = observability.read_archived("model_call")
        assert len(rec) >= 1_000_000  # roughly the count we wrote
    finally:
        observability.OBS_DIR = LAB_ROOT / "state" / "observability"


def test_rotation_no_op_below_threshold() -> None:
    """Files below ROTATION_THRESHOLD_BYTES are left in place."""
    tmp = _isolated_obs_dir(("OBS_DIR",))
    try:
        small = tmp / "verdict.jsonl"
        small.write_text('{"x":"y"}\n')  # tiny
        result = observability.rotate_all()
        assert result.get("verdict") is False or "verdict" not in result
        assert small.exists()
    finally:
        observability.OBS_DIR = LAB_ROOT / "state" / "observability"


def main() -> int:
    tests = [
        test_emit_creates_dir_and_appends_jsonl,
        test_emit_appends_multiple,
        test_calibration_count,
        test_emit_model_call_dual_emit,
        test_rotation_archives_oversized_jsonl,
        test_rotation_no_op_below_threshold,
        test_oai_compat_provider_returns_response_format,
        test_gemini_returns_response_schema_stripped,
        test_ollama_returns_format_param,
        test_anthropic_returns_empty_out_of_scope,
        test_supports_structured_output,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
