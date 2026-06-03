"""Smoke: prompt-cache prefix-stability (Sprint 4 C3, launch criterion 12).

Provider prefix caching only hits if the stable prefix (system prompt +
constitutional preamble + context brief = message[0]) is byte-identical
across a cycle's iterations. The compaction shapers must therefore NEVER
perturb the system message. These tests guard that invariant on the
deterministic shapers (network-free). The live ≥30% hit-rate against a real
cache-capable provider (Gemini explicit cache / Ollama native KV) is a
live-tier check, not a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import compact  # noqa: E402
from core.types import AgentMessage  # noqa: E402

# the stable, cacheable prefix — large enough to be worth caching
SYS = "SYSTEM: constitutional preamble + cycle context brief.\n" * 80


def _over_budget_messages() -> list[AgentMessage]:
    msgs = [AgentMessage(role="system", content=SYS),
            AgentMessage(role="user", content="the cycle task")]
    for i in range(30):
        msgs.append(AgentMessage(role="assistant", content="reasoning text " * 300))
        msgs.append(AgentMessage(role="tool", content="large tool result " * 300,
                                 tool_call_id=f"t{i}", name="Bash"))
    return msgs


def test_budget_reduce_preserves_system_prefix():
    msgs = _over_budget_messages()
    before = compact.total_tokens(msgs)
    out, dropped = compact.budget_reduce(msgs, target_tokens=2000, keep_system=True)
    # the cacheable prefix is byte-identical after compaction
    assert out[0].role == "system" and out[0].content == SYS
    assert compact.total_tokens(out) < before and dropped > 0


def test_snip_preserves_system_prefix():
    msgs = _over_budget_messages()
    out, _snipped = compact.snip_stale_tool_results(msgs)
    assert out[0].role == "system" and out[0].content == SYS


def test_full_apply_shapers_preserves_prefix():
    # recheck 2026-05-28 — the agent loop calls the FULL apply_shapers pipeline
    # (incl. the microcompact LLM shaper), not just the deterministic ones. Stub
    # provider.call so microcompact runs network-free, and assert message[0]
    # (the cacheable system prefix) survives the whole pipeline byte-identical.
    from core import provider as prov
    from core.types import ProviderResponse
    orig_call = prov.call

    def _fake_call(provider_name, messages, **kw):
        return ProviderResponse(
            text="summary of the older turns", tool_calls=[], finish_reason="stop",
            usage_prompt_tokens=10, usage_completion_tokens=10,
            usage_thinking_tokens=0, usage_cached_tokens=0,
            model="stub", provider="stub", elapsed_ms=1)
    try:
        prov.call = _fake_call
        msgs = _over_budget_messages()
        out = compact.apply_shapers(msgs, target_tokens=1500,
                                    provider_name="stub", model="stub")
        assert out[0].role == "system" and out[0].content == SYS
    finally:
        prov.call = orig_call


def test_cache_hit_rate_from_real_ledger():
    # criterion 12's metric, computed by the REAL cost_ledger.cache_hit_rate()
    # over real ledger rows (not a local lambda — recheck 2026-05-28).
    import shutil
    import tempfile

    from core import cost_ledger as cl
    tmp = Path(tempfile.mkdtemp())
    orig = cl.LEDGER_PATH
    try:
        cl.LEDGER_PATH = tmp / "cost.jsonl"
        assert cl.cache_hit_rate() == 0.0          # no rows
        cl.record(provider="gemini", model="g", input_tokens=1000,
                  output_tokens=200, cached_tokens=0)        # iter 1 — cold
        cl.record(provider="gemini", model="g", input_tokens=1000,
                  output_tokens=200, cached_tokens=900)      # iter 2 — warm
        # 900 cached / 2000 prompt = 0.45 ≥ the 0.30 target band
        assert cl.cache_hit_rate() == 0.45
    finally:
        cl.LEDGER_PATH = orig
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    tests = [
        test_budget_reduce_preserves_system_prefix,
        test_snip_preserves_system_prefix,
        test_full_apply_shapers_preserves_prefix,
        test_cache_hit_rate_from_real_ledger,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
