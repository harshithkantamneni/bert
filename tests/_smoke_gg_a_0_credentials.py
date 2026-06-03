"""Smoke test for the provider-key fix in tools/bert_run.py.

`_check_provider_keys` must read persisted credentials via `core.config.load()`
(not only `os.environ`), enumerate every provider env var, and fall back to the
environment when config loading fails. (The onboarding UI/API surfaces that
originally motivated this fix are not part of this repo.)
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def test_bert_run_check_uses_config_load_not_only_env() -> None:
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    start = src.index("def _check_provider_keys")
    body = src[start:start + 1000]
    assert "from core import config" in body or "core.config" in body
    assert "cfg.credentials.get(k)" in body or "cfg.credentials[k]" in body


def test_bert_run_check_includes_all_provider_env_vars() -> None:
    """`_check_provider_keys` must enumerate every provider env var (a prior
    version missed CEREBRAS_API_KEY, HF_TOKEN, and GOOGLE_AI_API_KEY)."""
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    start = src.index("def _check_provider_keys")
    body = src[start:start + 1000]
    for env_var in ("GROQ_API_KEY", "NVIDIA_API_KEY", "MISTRAL_API_KEY",
                    "CEREBRAS_API_KEY", "GOOGLE_AI_API_KEY",
                    "OPENROUTER_API_KEY", "HF_TOKEN"):
        assert env_var in body, f"_check_provider_keys candidates missing {env_var}"


def test_bert_run_falls_back_to_env_on_config_failure() -> None:
    """When `core.config.load()` raises, the runner falls back to os.environ."""
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    start = src.index("def _check_provider_keys")
    body = src[start:start + 1000]
    assert "except Exception" in body
    assert "os.environ.get(k)" in body
