"""Smoke test for core/llmlingua_compress.py — API contract verification.

Mocks the PromptCompressor so the test is fast and deterministic. The
LIVE verification (real ~280MB model load + real compression on 3K-token
text + BERTScore F1 ≥ 0.92) is a separate manual step — see
`tools/verify_llmlingua_live.py` (created alongside).

Run: `.venv/bin/python tests/_smoke_llmlingua_compress.py`
Exit 0 = pass; non-zero = fail.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import llmlingua_compress  # noqa: E402


def _make_mock_compressor(origin_tokens: int, compressed_tokens: int,
                          compressed_text: str = "compressed"):
    """Build a mock PromptCompressor that returns LLMLingua-2-shape result."""
    mock = MagicMock()
    mock.compress_prompt.return_value = {
        "compressed_prompt": compressed_text,
        "origin_tokens": origin_tokens,
        "compressed_tokens": compressed_tokens,
        "ratio": origin_tokens / compressed_tokens if compressed_tokens else 1.0,
        "rate": compressed_tokens / origin_tokens if origin_tokens else 1.0,
    }
    return mock


def test_compress_returns_text_and_stats() -> None:
    mock = _make_mock_compressor(origin_tokens=1000, compressed_tokens=200,
                                  compressed_text="short version")
    with patch.object(llmlingua_compress, "_compressor", mock):
        # Bypass get_compressor() lazy-load by setting the module-level singleton
        out_text, stats = llmlingua_compress.compress_for_cross_family(
            "long text " * 200, target_ratio=5.0,
        )
    assert out_text == "short version"
    assert stats["origin_tokens"] == 1000
    assert stats["compressed_tokens"] == 200
    assert stats["ratio"] == 5.0
    assert "compress_ms" in stats
    assert stats["compressor_model"] == llmlingua_compress.DEFAULT_MODEL


def test_force_keep_segments_passed_through() -> None:
    mock = _make_mock_compressor(origin_tokens=1000, compressed_tokens=300)
    with patch.object(llmlingua_compress, "_compressor", mock):
        llmlingua_compress.compress_for_cross_family(
            "context here", target_ratio=3.3,
            force_keep_segments=["query 1", "query 2"],
        )
    # Verify force_tokens was passed to compress_prompt
    call_kwargs = mock.compress_prompt.call_args.kwargs
    assert call_kwargs.get("force_tokens") == ["query 1", "query 2"]


def test_target_ratio_to_rate_inversion() -> None:
    """target_ratio=5 → rate=0.2 (keep 20% = 5x compression)."""
    mock = _make_mock_compressor(origin_tokens=1000, compressed_tokens=200)
    with patch.object(llmlingua_compress, "_compressor", mock):
        llmlingua_compress.compress_for_cross_family("x", target_ratio=5.0)
    call_kwargs = mock.compress_prompt.call_args.kwargs
    assert abs(call_kwargs["rate"] - 0.2) < 1e-6


def test_low_target_ratio_clamped_to_1() -> None:
    """target_ratio < 1.0 gets clamped (rate ≤ 1.0) — can't expand text."""
    mock = _make_mock_compressor(origin_tokens=1000, compressed_tokens=1000)
    with patch.object(llmlingua_compress, "_compressor", mock):
        llmlingua_compress.compress_for_cross_family("x", target_ratio=0.5)
    call_kwargs = mock.compress_prompt.call_args.kwargs
    assert call_kwargs["rate"] == 1.0  # clamped


def test_zero_compressed_tokens_safe() -> None:
    """Edge case: if compressor returns 0 compressed tokens (never should),
    avoid divide-by-zero."""
    mock = _make_mock_compressor(origin_tokens=1000, compressed_tokens=0)
    with patch.object(llmlingua_compress, "_compressor", mock):
        _, stats = llmlingua_compress.compress_for_cross_family("x")
    assert stats["ratio"] == 1.0  # safe fallback


def test_offline_default() -> None:
    """Module import sets HF_HUB_OFFLINE=1 if not already set."""
    import os
    # The module already imported at top of file; just verify the env var
    assert os.environ.get("HF_HUB_OFFLINE") == "1", (
        "Expected HF_HUB_OFFLINE=1 after llmlingua_compress import; got "
        f"{os.environ.get('HF_HUB_OFFLINE')!r}"
    )


def main() -> int:
    tests = [
        test_compress_returns_text_and_stats,
        test_force_keep_segments_passed_through,
        test_target_ratio_to_rate_inversion,
        test_low_target_ratio_clamped_to_1,
        test_zero_compressed_tokens_safe,
        test_offline_default,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
