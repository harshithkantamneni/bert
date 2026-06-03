"""Real cross-encoder reranker (L4 in AGI's 4-layer hybrid pipeline).

Phase B3 of the v3 plan. Replaces the cosine-rerank stub in
core/retrieval.py with bge-reranker-v2-m3 — a proper cross-encoder
that scores (query, passage) pairs with full attention rather than
the single-vector cosine approximation.

Usage from core/retrieval.py:

    from core import reranker
    fn = reranker.get_cross_encoder_rerank_fn()  # may return None
    if fn is not None:
        top = retrieval.hybrid_retrieve(query, rerank_fn=fn, ...)

Model: BAAI/bge-reranker-v2-m3 (~568 MB on disk; cached on first
download). ~10-12s warm latency for top-30 → top-K on M3 Pro CPU.
For lower latency, the subprocess pattern (AGI's RetrievalWorker)
keeps the model loaded across calls.

Failure mode: if the model cannot load (no network on first call,
disk pressure, sentence-transformers cold-load timeout), this module
returns None from get_cross_encoder_rerank_fn() and core/retrieval.py
falls back to default_cosine_reranker(). No crash, just lower quality.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger("bert.reranker")

# Default model. Override via BERT_RERANKER_MODEL env var.
DEFAULT_MODEL = os.environ.get(
    "BERT_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)
# A smaller faster fallback (auto-tried if default fails to load)
FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankerStatus:
    model: str | None
    loaded: bool
    last_load_attempt_ts: float
    last_error: str | None


_lock = threading.Lock()
_model = None                  # the CrossEncoder instance
_model_name: str | None = None  # which model is loaded
_load_failed = False           # don't re-attempt if we've already failed
_last_error: str | None = None


def _try_load(model_name: str) -> bool:
    """Attempt to load a CrossEncoder model. Returns True on success.
    Bounded by BERT_RERANKER_LOAD_TIMEOUT_S seconds (default 30); if
    loading exceeds the budget, marks as failed and the caller falls
    back. This avoids hangs in tests + cold environments where HF Hub
    network checks stall on update verification."""
    global _model, _model_name, _last_error
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        _last_error = f"sentence_transformers not installed: {e}"
        return False

    timeout_s = float(os.environ.get("BERT_RERANKER_LOAD_TIMEOUT_S", "30"))
    result: dict[str, Any] = {"model": None, "err": None}

    def _load_thread() -> None:
        try:
            result["model"] = CrossEncoder(model_name)
        except Exception as e:  # noqa: BLE001
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_load_thread, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        _last_error = (
            f"load timed out after {timeout_s:.0f}s — leave daemon thread "
            f"running and mark as failed"
        )
        LOG.warning("cross-encoder %s: %s", model_name, _last_error)
        return False
    if result["err"]:
        _last_error = result["err"]
        LOG.warning("cross-encoder %s load failed: %s", model_name, result["err"])
        return False
    _model = result["model"]
    _model_name = model_name
    LOG.info("cross-encoder loaded: %s", model_name)
    return True


def _ensure_loaded() -> bool:
    """Lazy-load the cross-encoder. Idempotent + thread-safe. Caches
    failure so we don't retry on every search call."""
    global _model, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False
    with _lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        # Try default model first
        if _try_load(DEFAULT_MODEL):
            return True
        # Fall back to smaller MiniLM if default fails
        if _try_load(FALLBACK_MODEL):
            return True
        # Both failed — give up + cache the failure
        _load_failed = True
        return False


def is_available() -> bool:
    """Check whether the reranker can be used right now (without
    actually loading if not yet loaded)."""
    if _model is not None:
        return True
    if _load_failed:
        return False
    return _ensure_loaded()


def status() -> RerankerStatus:
    return RerankerStatus(
        model=_model_name,
        loaded=(_model is not None),
        last_load_attempt_ts=time.time(),
        last_error=_last_error,
    )


def rerank(query: str, passages: list[str]) -> list[float]:
    """Score each (query, passage) pair with the cross-encoder.
    Returns a list of float scores parallel to `passages`.

    Returns empty list if the reranker is unavailable (caller falls
    back to the RRF order or cosine-rerank)."""
    if not query or not passages:
        return []
    if not _ensure_loaded():
        return []
    pairs = [[query, p or ""] for p in passages]
    try:
        scores = _model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
    except Exception as e:  # noqa: BLE001
        LOG.warning("cross-encoder predict failed: %s", e)
        return []


# ── Adapter for core/retrieval.py rerank_fn ──────────────────────────


def get_cross_encoder_rerank_fn():
    """Return a callable matching the rerank_fn signature expected by
    core.retrieval.hybrid_retrieve: (query, candidates) → reranked.

    The returned fn lazy-loads the cross-encoder on first actual use
    (when called with non-empty candidates). This avoids paying the
    568 MB model load cost on empty searches, tests, and dry-runs.

    Caller falls back to RRF order if the reranker eventually fails
    to load.
    """
    # Quick disable path for tests + environments that can't afford
    # the cold-start cost. Honor either env var.
    if os.environ.get("BERT_DISABLE_RERANKER", "").strip() in {"1", "true", "yes"}:
        return None
    # If we've already tried + failed earlier in this process, don't
    # even return a fn — let caller skip the rerank step entirely.
    if _load_failed:
        return None

    def _rerank_fn(query: str, candidates: list) -> list:
        """Score top candidates with the cross-encoder; sort by score.
        Lazy-loads the model on first non-empty candidate set."""
        if not candidates:
            return candidates
        if not _ensure_loaded():
            # Load attempt failed (timeout, missing deps, etc.) — keep
            # RRF order; subsequent calls will skip the load attempt.
            return candidates
        passages = [c.text or "" for c in candidates]
        scores = rerank(query, passages)
        if not scores or len(scores) != len(candidates):
            return candidates
        for c, s in zip(candidates, scores, strict=False):
            c.rerank_score = float(s)
        return candidates

    return _rerank_fn


# ── CLI smoke ────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """python -m core.reranker status
    python -m core.reranker score "<query>" "<passage1>" "<passage2>" ...
    """
    import sys
    if len(argv) < 2:
        print("usage: reranker status|score ...", file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "status":
        avail = is_available()
        st = status()
        print(f"available={avail} model={st.model} last_error={st.last_error}")
        return 0
    if cmd == "score":
        if len(argv) < 4:
            print('usage: reranker score "<query>" "<passage1>" ...',
                  file=sys.stderr)
            return 2
        q = argv[2]
        passages = argv[3:]
        scores = rerank(q, passages)
        if not scores:
            print("(reranker unavailable; no scores)")
            return 1
        ranked = sorted(zip(passages, scores, strict=False), key=lambda x: x[1], reverse=True)
        for p, s in ranked:
            print(f"  [{s:+.4f}] {p[:80]}")
        return 0
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv))
