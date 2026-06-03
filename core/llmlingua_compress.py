"""LLMLingua-2 prompt compression for cross-family judge legs.

The cross-family judge dispatch (P-VS-02 + P-VS-07 phase 2 on META/SPEC
altitude) can't use KV-cache reuse because the producer and judge run on
different model families (KV state isn't transferable across model
architectures). LLMLingua-2 fills the gap: provider-agnostic prompt
compression that preserves semantic content while shrinking the standing-
context portion of the prompt.

Per R8 grounded numbers (`findings/researcher_lab_latent_comms_R8.md`):
  - 4-10× typical compression on dense-prose standing context
  - 20× peak compression on highly-redundant prose
  - ~1.5% accuracy loss on GSM8K
  - End-to-end speedup 1.7-5.7× on cross-family-judge dispatches
  - Production case: $42K/month → $2.1K/month bill (20× cost reduction)
  - Provider-agnostic: works as preprocessing before any /chat/completions
    call, including bert's free-tier providers

bert-specific use:
  - Threshing → clearness phase 1 → clearness phase 2 chain that ends
    with cross-family judge: compress the standing-context portion
    BEFORE sending to the judge; preserve phase-1 questions verbatim
    (those need to fire as queries with their original phrasing).
  - Same-family chains DON'T compress (KV-cache reuse handles them;
    this module is specifically the cross-family bridge).

This module exposes:
  - compress_for_cross_family(text, target_ratio=5.0, force_keep_segments=None)
    → (compressed_text: str, stats: dict)
  - get_compressor() → lazily-loaded PromptCompressor singleton

Lazy init: model loads on first call (~10-30s on M3 Pro CPU, ~5-10s on
MPS). HF_HUB_OFFLINE=1 default per same discipline as core/memory.py;
first-time download falls back to online once.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

# Same offline-by-default discipline as core/memory.py — bert is
# free-tier-runtime, model is cached after first download, library
# default of online-update-check is wrong for bert.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from core import log

if TYPE_CHECKING:
    from llmlingua import PromptCompressor

LOG = log.get_logger("bert.llmlingua")

# LLMLingua-2 model — task-agnostic, 3-6× faster than original LLMLingua-1.
# Multilingual roberta-large variant ~280MB; English-only variant ~140MB.
# We default to the multilingual one because bert's research can include
# non-English content (e.g., A6's Sheeran citations are English but bert
# could surface foreign-language sources). User can override via
# BERT_LLMLINGUA_MODEL env var.
DEFAULT_MODEL = os.environ.get(
    "BERT_LLMLINGUA_MODEL",
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
)

_compressor: PromptCompressor | None = None


def get_compressor() -> PromptCompressor:
    """Lazily load and cache the LLMLingua-2 PromptCompressor.

    First call: ~10-30s (model load on CPU, ~5-10s on MPS). Subsequent
    calls return the cached instance instantly. Thread-unsafe — bert's
    sub-agent dispatch is sequential per cycle, so this is fine.
    """
    global _compressor
    if _compressor is not None:
        return _compressor

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    # Pick device: LLMLingua's default is 'cuda' which crashes on M3 Pro
    # (no NVIDIA GPU; transformers' caching_allocator_warmup unconditionally
    # calls cudaMemGetInfo). Use BERT_LLMLINGUA_DEVICE env override or
    # auto-detect: MPS if available, else CPU.
    device = os.environ.get("BERT_LLMLINGUA_DEVICE")
    if not device:
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
    LOG.info(
        "loading LLMLingua-2 compressor model=%s device=%s (offline=%s; "
        "first call ~10-30s cached, ~30-60s if model needs download)",
        DEFAULT_MODEL, device, offline,
    )
    from llmlingua import PromptCompressor

    def _build():
        return PromptCompressor(
            model_name=DEFAULT_MODEL,
            use_llmlingua2=True,
            device_map=device,
        )

    try:
        _compressor = _build()
    except Exception as e:
        if offline:
            LOG.warning(
                "offline LLMLingua load failed (%s); retrying with "
                "HF_HUB_OFFLINE=0 for one-time model download", e
            )
            os.environ["HF_HUB_OFFLINE"] = "0"
            try:
                _compressor = _build()
            finally:
                os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            raise
    return _compressor


def compress_for_cross_family(
    text: str,
    target_ratio: float = 5.0,
    force_keep_segments: list[str] | None = None,
) -> tuple[str, dict]:
    """Compress `text` to roughly 1/target_ratio of its original size while
    preserving semantic content. Designed for the cross-family judge leg
    (P-VS-02 + P-VS-07 phase 2 META/SPEC) of bert's Quaker pipeline.

    Args:
      text: the standing-context portion to compress. NOT the full prompt
        — the cacheable prefix (constitutional preamble + role framing)
        and the per-call delta (variable inputs) live elsewhere; this
        function only compresses the bulky middle part.
      target_ratio: target compression factor. 5.0 means compressed text
        is ~20% of original token count. Real LLMLingua-2 typically
        achieves 4-10× depending on input redundancy. Values >10 push
        toward semantic loss.
      force_keep_segments: list of substrings that MUST appear verbatim
        in the compressed output. Use for phase-1 clearness queries that
        need to pass through to the judge with their original phrasing.

    Returns:
      (compressed_text, stats) where stats dict contains:
        origin_tokens: int — original token count per LLMLingua-2's tokenizer
        compressed_tokens: int — compressed token count
        ratio: float — origin / compressed (the actual achieved ratio)
        compress_ms: int — wall-clock time for compression
        compressor_model: str — which model produced the compression
    """
    compressor = get_compressor()

    # LLMLingua-2 expects rate (target_ratio inverse) or target_token.
    # rate=0.2 means "keep 20% of tokens" = 5x compression.
    rate = 1.0 / max(target_ratio, 1.0)

    start = time.monotonic()
    # llmlingua-2 .compress_prompt(...) signature has varied across
    # versions; the key parameters are `context` (the text to compress)
    # and `rate` (or `target_token`). force_tokens preserves verbatim.
    # use_token_level_filter=True (LLMLingua-2 default) is required for real
    # 4-10× compression. False disables the per-token classifier and falls
    # back to context/sentence-level filtering which only achieves ~1× on
    # dense prose. Per the LLMLingua-2 paper + tests/_smoke_llmlingua live
    # verification on M3 Pro/MPS.
    kwargs: dict = {"context": [text], "rate": rate, "use_token_level_filter": True}
    if force_keep_segments:
        kwargs["force_tokens"] = force_keep_segments

    result = compressor.compress_prompt(**kwargs)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    # LLMLingua-2 returns dict with keys: compressed_prompt, origin_tokens,
    # compressed_tokens, ratio, rate. ratio is reported as "origin/compressed"
    # in some versions and as the rate (compressed/origin) in others; we
    # recompute to be safe.
    compressed_text = result.get("compressed_prompt", "")
    origin_tokens = result.get("origin_tokens", 0)
    compressed_tokens = result.get("compressed_tokens", 0)
    actual_ratio = (origin_tokens / compressed_tokens) if compressed_tokens > 0 else 1.0

    stats = {
        "origin_tokens": origin_tokens,
        "compressed_tokens": compressed_tokens,
        "ratio": round(actual_ratio, 2),
        "compress_ms": elapsed_ms,
        "compressor_model": DEFAULT_MODEL,
    }
    LOG.info(
        "compressed %d → %d tokens (%.2f×, %dms)",
        origin_tokens, compressed_tokens, actual_ratio, elapsed_ms,
    )
    return compressed_text, stats
