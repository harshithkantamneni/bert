"""Pin retrieval models resident at MCP-server start (memory v3+ priority 2).

The embedder (bge-base-en-v1.5, used on every search) and the cross-encoder
reranker (bge-reranker-v2-m3, used on every search unless BERT_DISABLE_RERANKER)
are otherwise lazy-loaded on the FIRST query — paying a measured cold-start tail
(p99.9 = 8.6s, max = 46s under memory pressure; 99.7% of warm calls are <1s).
Pre-warming them at server start moves that cost off the first user query.

Default is a BACKGROUND daemon thread: the server starts immediately and the
models load while the host LLM is still deciding what to do, so the first real
search is (almost always) warm. The lazy loaders remain the correctness
fallback — they're thread-safe singletons, so a query that races the warm-up
just blocks on the same load the thread is already doing.

Best-effort throughout: a load failure (no cached model / offline) is logged,
never raised. Pre-warming must never break server startup.
"""

from __future__ import annotations

import os
import threading

from core import log

LOG = log.get_logger("bert.prewarm")


def _reranker_disabled() -> bool:
    return os.environ.get("BERT_DISABLE_RERANKER", "").strip() in {"1", "true", "yes"}


def prewarm_embedder() -> bool:
    """Eagerly load the sentence-transformer embedder. Returns True if resident
    afterward, False (logged) on failure. Never raises."""
    try:
        from core import memory
        memory._get_embedder()
        return memory._embedder is not None
    except Exception as e:  # noqa: BLE001
        LOG.warning("prewarm: embedder load failed (advisory): %s", e)
        return False


def prewarm_reranker() -> bool:
    """Eagerly load the cross-encoder reranker, unless reranking is disabled.
    Returns True if loaded, False if skipped/failed. Never raises."""
    if _reranker_disabled():
        LOG.info("prewarm: reranker disabled (BERT_DISABLE_RERANKER); skipping")
        return False
    try:
        from core import reranker
        return bool(reranker._ensure_loaded())
    except Exception as e:  # noqa: BLE001
        LOG.warning("prewarm: reranker load failed (advisory): %s", e)
        return False


def prewarm(*, embedder: bool = True, reranker: bool = True,
            background: bool = True) -> threading.Thread | None:
    """Pre-warm retrieval models so the first query doesn't pay cold-start.

    background=True (default) loads in a daemon thread and returns it;
    background=False loads synchronously and returns None. Both are best-effort.
    """
    def _run() -> None:
        if embedder:
            ok_e = prewarm_embedder()
            LOG.info("prewarm: embedder %s", "resident" if ok_e else "unavailable")
        if reranker:
            ok_r = prewarm_reranker()
            LOG.info("prewarm: reranker %s", "resident" if ok_r else "skipped/unavailable")

    if background:
        t = threading.Thread(target=_run, name="bert-prewarm", daemon=True)
        t.start()
        return t
    _run()
    return None
