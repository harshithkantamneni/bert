"""Measure Ollama's built-in prefix-cache effectiveness on bert's
dispatch patterns.

Before assuming we need true KVComm (multi-week custom Ollama
build), find out how much speedup Ollama's *automatic* prefix
cache already delivers.

Method (per call pair):
  1. Fire a long stable prefix + per-call delta A
  2. Fire the SAME stable prefix + delta B
  3. Compare prompt_eval_duration on call 1 vs call 2 — the second
     should be near-zero because the prefix tokens are cache-warm
  4. Compare total_duration to surface TTFT speedup

Ollama response fields we read:
  prompt_eval_count      — tokens in the prompt (constant when prefix
                            stable)
  prompt_eval_duration   — ns spent processing the prompt (drops to
                            near-zero on cache hit)
  load_duration          — ns spent loading the model (drops to ~0 on
                            warm calls because OLLAMA_KEEP_ALIVE=24h
                            keeps the model resident)
  eval_count             — tokens generated
  eval_duration          — ns spent generating
  total_duration         — sum of all phases (= TTFT + generation)

We report:
  speedup ratio = call_1_total_duration / call_2_total_duration
  prefix_cache_ratio = call_1_prompt_eval_duration / call_2_prompt_eval_duration
  tokens/sec on each call

Exit code:
  0 — measurement succeeded
  1 — Ollama unreachable
  2 — model not installed

Usage:
  python tools/measure_ollama_prefix_cache.py
  python tools/measure_ollama_prefix_cache.py --model qwen3:8b --iterations 3
  python tools/measure_ollama_prefix_cache.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent

# A stable prefix that's representative of bert's actual cycle context.
# Long enough (≥1024 tokens) that we expect Ollama to cache it.
STABLE_PREFIX = """\
You are bert, an autonomous research lab. You read, debate, decide,
and write across eight free-tier model providers with cross-family
adversarial review per P-VS-02. Your dispatches follow the Quaker
discernment pipeline: threshing → clearness phase 1 (queries) →
clearness phase 2 (verdicts + concerns) → cross-family judge on
high-stakes verdicts. Maintain stable-prefix discipline so prefix-
cache hits land. Use the 10-layer memory system: core (always-in-
context, ≤3K tokens), recall (semi-warm, retrieved per dispatch),
archival (cold-stored, searched on demand). Falsifier discipline
per P-003: pre-register a measurable failure mode before commit.
Permission gates per P-005: PI ratifies skill promotion. Provider
limits per quota: respect RPM ceilings (NVIDIA 40 / Groq 30 /
Cerebras 30 / Mistral 30 / Gemini 60 / OpenRouter 20 / HF Router 30).
Cache discipline: stable prefix → byte-identical → automatic cache
hit on Gemini 2.5+ implicit (≥1024 tokens, 90% off), Groq GPT-OSS
automatic (50% + cached tokens don't count vs RPM), Ollama
prefix-cache (warm model + identical prefix → KV reused). LLMLingua-
2 compression for cross-family judge legs (4-10× compression with
BERTScore F1 ≥ 0.92). Concern propagation via caveats_embedded
across dispatches; address within 5 cycles or age out. Seasoning
queue (P-VS-09): high-water mark 25 unrevived entries; revival_
outcome_quality ≥40%. All decisions are reproducible
per defining commitment #4: every event in lab/sor/events.jsonl is
Merkle-hashed at checkpoint; canvas time-machine mode replays any
past lab state. Standing context this dispatch follows below.
""" * 5  # ~3500 chars of stable prefix per repeat → ~1100 tokens × 5 = 5500


def _call_ollama(host: str, model: str, prefix: str, delta: str,
                  timeout: float = 120.0) -> dict:
    import httpx
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": delta},
        ],
        "stream": False,
        "options": {"num_predict": 32, "temperature": 0.0},
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{host}/api/chat", json=body)
        resp.raise_for_status()
        return resp.json()


def _ns_to_ms(ns: int | float) -> float:
    return (ns or 0) / 1_000_000


def _measure_pair(host: str, model: str, prefix: str,
                   delta_a: str, delta_b: str) -> dict:
    """Run a cold → warm pair and compute deltas."""
    cold = _call_ollama(host, model, prefix, delta_a)
    warm = _call_ollama(host, model, prefix, delta_b)
    return {
        "cold": {
            "prompt_eval_count": cold.get("prompt_eval_count", 0),
            "prompt_eval_ms": _ns_to_ms(cold.get("prompt_eval_duration", 0)),
            "load_ms": _ns_to_ms(cold.get("load_duration", 0)),
            "eval_count": cold.get("eval_count", 0),
            "eval_ms": _ns_to_ms(cold.get("eval_duration", 0)),
            "total_ms": _ns_to_ms(cold.get("total_duration", 0)),
        },
        "warm": {
            "prompt_eval_count": warm.get("prompt_eval_count", 0),
            "prompt_eval_ms": _ns_to_ms(warm.get("prompt_eval_duration", 0)),
            "load_ms": _ns_to_ms(warm.get("load_duration", 0)),
            "eval_count": warm.get("eval_count", 0),
            "eval_ms": _ns_to_ms(warm.get("eval_duration", 0)),
            "total_ms": _ns_to_ms(warm.get("total_duration", 0)),
        },
    }


def measure(host: str = "http://localhost:11434",
             model: str = "qwen3:8b",
             iterations: int = 3) -> dict:
    """Run `iterations` cold→warm pairs and aggregate."""
    # First check the model is loaded; pre-warm with a throwaway call
    # so load_duration on the "cold" call is fair (no model load).
    try:
        _call_ollama(host, model, "warmup", "ok", timeout=60.0)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}

    deltas = [
        ("Analyse cycle 1 outputs.", "Analyse cycle 2 outputs."),
        ("Summarise the threshing pass.", "Summarise the clearness pass."),
        ("Review the falsifier baseline.", "Review the seasoning queue."),
    ]
    pairs = []
    for i in range(iterations):
        delta_a, delta_b = deltas[i % len(deltas)]
        # Use a unique tail in delta_a so it's not cache-hit from prior
        # pair, but the prefix stays byte-identical.
        unique_a = f"{delta_a} [iter-{i}-a]"
        unique_b = f"{delta_b} [iter-{i}-b]"
        result = _measure_pair(host, model, STABLE_PREFIX, unique_a, unique_b)
        result["iteration"] = i
        pairs.append(result)
        time.sleep(0.5)  # brief breathing room between pairs

    # Aggregate
    cold_prompts = [p["cold"]["prompt_eval_ms"] for p in pairs]
    warm_prompts = [p["warm"]["prompt_eval_ms"] for p in pairs]
    cold_totals = [p["cold"]["total_ms"] for p in pairs]
    warm_totals = [p["warm"]["total_ms"] for p in pairs]

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    prefix_speedup = (avg(cold_prompts) / avg(warm_prompts)) if avg(warm_prompts) > 0 else float("inf")
    total_speedup = (avg(cold_totals) / avg(warm_totals)) if avg(warm_totals) > 0 else float("inf")

    return {
        "ok": True,
        "host": host,
        "model": model,
        "iterations": iterations,
        "prefix_chars": len(STABLE_PREFIX),
        "summary": {
            "avg_cold_prompt_ms": round(avg(cold_prompts), 1),
            "avg_warm_prompt_ms": round(avg(warm_prompts), 1),
            "prompt_eval_speedup_x": round(prefix_speedup, 2),
            "avg_cold_total_ms": round(avg(cold_totals), 1),
            "avg_warm_total_ms": round(avg(warm_totals), 1),
            "total_speedup_x": round(total_speedup, 2),
        },
        "pairs": pairs,
        "verdict": _verdict(prefix_speedup, total_speedup),
    }


def _verdict(prefix_speedup: float, total_speedup: float) -> dict:
    """Translate measurements into an action recommendation."""
    if prefix_speedup >= 5.0:
        return {
            "rating": "excellent",
            "message": (
                f"Ollama prefix cache delivers {prefix_speedup:.1f}× on "
                "prompt-eval. The KVComm 'deferred multi-week build' "
                "is unnecessary at bert's scale — the built-in cache "
                "satisfies the SAME_FAMILY_LOCAL route."
            ),
            "recommended_action": "drop kv_handoff_pending sentinel from "
                                  "core/kv_sharing.py; rename route to "
                                  "'ollama_native_cache'",
        }
    if prefix_speedup >= 2.0:
        return {
            "rating": "good",
            "message": (
                f"Ollama prefix cache delivers {prefix_speedup:.1f}× on "
                "prompt-eval. Most of the KVComm win is already realized; "
                "the marginal 2-3× more from true KVComm probably doesn't "
                "justify the multi-week build at bert's single-user scale."
            ),
            "recommended_action": "document Ollama-native as the SAME_FAMILY"
                                  "_LOCAL implementation; revisit KVComm "
                                  "when concurrent-agent load demands it",
        }
    return {
        "rating": "poor",
        "message": (
            f"Ollama prefix cache only delivers {prefix_speedup:.1f}× — "
            "below expected. Investigate: (1) is OLLAMA_KEEP_ALIVE=24h "
            "set? (2) is the prefix byte-identical across calls? (3) does "
            "the model support prefix caching (Ollama 0.3+)? (4) is the "
            "prefix length ≥1024 tokens?"
        ),
        "recommended_action": "fix the configuration first; if speedup "
                              "remains <2×, true KVComm becomes justified",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Ollama prefix cache")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--json", action="store_true",
                        help="emit JSON only (machine-readable)")
    args = parser.parse_args()

    result = measure(host=args.host, model=args.model, iterations=args.iterations)
    if not result["ok"]:
        print(f"error: {result.get('error', 'unknown')}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    s = result["summary"]
    print("bert · Ollama prefix-cache measurement")
    print("=" * 50)
    print(f"  model:                {result['model']}")
    print(f"  iterations:           {result['iterations']}")
    print(f"  stable prefix length: {result['prefix_chars']} chars")
    print()
    print(f"  cold prompt-eval:     {s['avg_cold_prompt_ms']:.0f} ms")
    print(f"  warm prompt-eval:     {s['avg_warm_prompt_ms']:.0f} ms")
    print(f"  prompt-eval speedup:  {s['prompt_eval_speedup_x']:.2f}×")
    print()
    print(f"  cold total:           {s['avg_cold_total_ms']:.0f} ms")
    print(f"  warm total:           {s['avg_warm_total_ms']:.0f} ms")
    print(f"  total speedup:        {s['total_speedup_x']:.2f}×")
    print()
    v = result["verdict"]
    print(f"  verdict: {v['rating'].upper()}")
    print(f"    {v['message']}")
    print()
    print("  recommended action:")
    print(f"    {v['recommended_action']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
