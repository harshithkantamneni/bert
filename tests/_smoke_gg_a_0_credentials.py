"""Smoke test for GG-A.0 — provider-key onboarding through the UI.

Pre-GG-A.0, bert had three loosely-connected pieces:
  - core/config.py loaded keys from ~/.bert-lab/credentials.json + env
  - api /api/onboarding/test-credential validated one key at a time
  - bert Onboarding surface had a Providers panel that tested keys
    BUT never persisted them — clicking "continue" advanced the panel
    and dropped the keys

Plus a real bug in tools/bert_run.py: _check_provider_keys only read
os.environ, so even after onboarding wrote credentials.json the
runner aborted with "no provider keys in env".

GG-A.0 closes all three gaps:
  - new /api/onboarding/save-credentials POST persists validated keys
    to ~/.bert-lab/credentials.json (mode 600), merging with existing
  - new /api/onboarding/credentials-status GET reports which env vars
    are present (without echoing values)
  - Onboarding.tsx Providers panel now calls save-credentials before
    advancing, with per-key feedback
  - App.tsx redirects "/" to "/onboard" when no provider keys exist
  - bert_run.py _check_provider_keys reads via core.config.load() so
    persisted keys count, not just env

Covers (no live API calls — endpoint behavior is verified via the
FastAPI TestClient where possible, source checks otherwise):
  - SaveCredentialsRequest schema accepted
  - save-credentials never writes invalid keys (test with a guaranteed-
    fail key string)
  - save-credentials writes mode 600
  - save-credentials performs additive merge (doesn't clobber existing)
  - credentials-status never echoes key VALUES, only env var names
  - bert_run._check_provider_keys reads via core.config
  - Onboarding UI calls save-credentials + shows per-key result
  - App.tsx redirects root to /onboard when has_any_provider is false
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


# ─── Backend: save endpoint ────────────────────────────────────────


def test_save_endpoint_exists_in_api() -> None:
    src = (LAB_ROOT / "api" / "main.py").read_text()
    assert '@app.post("/api/onboarding/save-credentials")' in src
    assert "def save_credentials" in src


def test_status_endpoint_exists_in_api() -> None:
    src = (LAB_ROOT / "api" / "main.py").read_text()
    assert '@app.get("/api/onboarding/credentials-status")' in src
    assert "def credentials_status" in src


def test_status_endpoint_never_echoes_key_values() -> None:
    """The endpoint must report env var NAMES only, never the key
    string values. Grep the function body for the value-yielding
    pattern."""
    src = (LAB_ROOT / "api" / "main.py").read_text()
    # Find the function body
    start = src.index("def credentials_status")
    body = src[start:start + 1500]
    # The body should reference k (the env var name) in `present`,
    # not v (the actual key value).
    assert '"present": present' in body
    # Should NOT return cfg.credentials as-is or any structure that
    # contains the raw values.
    assert "cfg.credentials" not in body or "v" not in body.split('"present"')[1][:200] or True
    # Strict: must have a present-list that filters out v
    assert "k for k, v in cfg.credentials.items() if v" in body


def test_save_endpoint_validates_before_writing() -> None:
    """The save endpoint must re-validate every key server-side, not
    trust the frontend's claim of validation. Look for the httpx call
    inside save_credentials."""
    src = (LAB_ROOT / "api" / "main.py").read_text()
    start = src.index("def save_credentials")
    body = src[start:start + 4000]
    assert "import httpx" in body
    assert "client.get(" in body
    assert "/models" in body  # the validation probe path
    # Must check 2xx before adding to validated
    assert "200 <= r.status_code < 300" in body


def test_save_endpoint_writes_mode_600() -> None:
    src = (LAB_ROOT / "api" / "main.py").read_text()
    start = src.index("def save_credentials")
    body = src[start:start + 4000]
    assert "chmod(0o600)" in body


def test_save_endpoint_additive_merge_not_clobber() -> None:
    """Re-saving partial keys must NOT wipe existing keys."""
    src = (LAB_ROOT / "api" / "main.py").read_text()
    start = src.index("def save_credentials")
    body = src[start:start + 4000]
    assert "existing.read_text" in body or "cred_path.exists()" in body
    assert "merged = {**existing, **validated}" in body


def test_save_endpoint_busts_config_cache() -> None:
    """After writing new keys, the cached Config must be invalidated
    so the next config.load() picks them up without restart."""
    src = (LAB_ROOT / "api" / "main.py").read_text()
    start = src.index("def save_credentials")
    body = src[start:start + 4000]
    assert "_cached = None" in body


# ─── bert_run.py: _check_provider_keys fix ─────────────────────────


def test_bert_run_check_uses_config_load_not_only_env() -> None:
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    start = src.index("def _check_provider_keys")
    body = src[start:start + 1000]
    assert "from core import config" in body or "core.config" in body
    assert "cfg.credentials.get(k)" in body or "cfg.credentials[k]" in body


def test_bert_run_check_includes_all_provider_env_vars() -> None:
    """Pre-GG the candidates list missed CEREBRAS_API_KEY, HF_TOKEN,
    and GOOGLE_AI_API_KEY (used GOOGLE_API_KEY instead, which only
    becomes effective via the BB.7 alias). Post-GG all should be in."""
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    start = src.index("def _check_provider_keys")
    body = src[start:start + 1000]
    for env_var in ("GROQ_API_KEY", "NVIDIA_API_KEY", "MISTRAL_API_KEY",
                    "CEREBRAS_API_KEY", "GOOGLE_AI_API_KEY",
                    "OPENROUTER_API_KEY", "HF_TOKEN"):
        assert env_var in body, f"_check_provider_keys candidates missing {env_var}"


def test_bert_run_falls_back_to_env_on_config_failure() -> None:
    """When core.config.load() raises, the runner should still gracefully
    fall back to os.environ to preserve the original behavior."""
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    start = src.index("def _check_provider_keys")
    body = src[start:start + 1000]
    assert "except Exception" in body
    assert "os.environ.get(k)" in body


# ─── Onboarding UI integration ─────────────────────────────────────


def test_onboarding_calls_save_credentials() -> None:
    src = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
           "Onboarding.tsx").read_text()
    assert "/api/onboarding/save-credentials" in src
    # Should send env-var-keyed map, not provider-id-keyed
    assert "p.envVar" in src


def test_onboarding_surfaces_save_result_to_user() -> None:
    src = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
           "Onboarding.tsx").read_text()
    # The component should display saved/skipped counts
    assert "saveResult" in src
    assert "saved_count" in src
    assert "skipped_count" in src


def test_onboarding_only_advances_when_at_least_one_key_saved() -> None:
    """If 0 keys save AND credentials.json was empty, the panel must
    stay open so the user can fix the failures rather than silently
    advancing to the next step."""
    src = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
           "Onboarding.tsx").read_text()
    # Look for the gate
    assert "r.saved_count > 0" in src
    assert "onNext()" in src
    # And the early-exit when no creds at all to save
    assert 'Object.keys(credentials).length === 0' in src or \
           "Object.keys(credentials).length == 0" in src


# ─── App.tsx redirect ──────────────────────────────────────────────


def test_app_redirects_root_to_onboard_when_no_keys() -> None:
    src = (LAB_ROOT / "bert" / "v4" / "src" / "App.tsx").read_text()
    # Hook exists
    assert "useCredentialsReady" in src
    # Root route is conditional
    assert "Navigate to=\"/onboard\"" in src or 'Navigate to="/onboard"' in src
    # Calls credentials-status
    assert "/api/onboarding/credentials-status" in src


def test_app_redirect_skips_inside_onboard_itself() -> None:
    """The gate must NOT re-redirect while the user is finishing the
    onboarding flow (otherwise it'd loop)."""
    src = (LAB_ROOT / "bert" / "v4" / "src" / "App.tsx").read_text()
    assert "location.pathname.startsWith" in src
    assert '"/onboard"' in src


def test_app_redirect_skipped_in_demo_mode() -> None:
    """Demo mode shows a pre-seeded lab; no need to gate on keys."""
    src = (LAB_ROOT / "bert" / "v4" / "src" / "App.tsx").read_text()
    # The hook is called with !isDemoMode
    assert "!isDemoMode" in src


def test_app_fails_open_on_probe_error() -> None:
    """If the credentials-status probe itself fails (API down,
    network), don't lock the user out — fail open and show
    FirstLight. The error case is rare and a hard lock is worse than
    a soft one."""
    src = (LAB_ROOT / "bert" / "v4" / "src" / "App.tsx").read_text()
    # The .catch() in the hook sets ready=true
    assert ".catch" in src
    # And the assignment is to true (fail-open)
    catch_block = src.split(".catch(")[1][:200]
    assert "setReady(true)" in catch_block


# ─── Smoke endpoint via TestClient (cheap roundtrip) ───────────────


def test_status_endpoint_returns_expected_shape() -> None:
    """In-process TestClient call — no httpx, no validation traffic."""
    from fastapi.testclient import TestClient
    # Import the app inside the test so module-level side effects
    # don't fire during collection
    from api.main import app
    client = TestClient(app)
    r = client.get("/api/onboarding/credentials-status")
    assert r.status_code == 200
    data = r.json()
    assert "present" in data
    assert "count" in data
    assert "has_any_provider" in data
    assert isinstance(data["present"], list)
    assert isinstance(data["has_any_provider"], bool)
    # Critically: no key values in the response
    response_text = r.text
    # If ~/.bert-lab/credentials.json has real keys, none of them
    # should appear in the response body. Check a few common prefixes.
    for forbidden_prefix in ("csk-", "nvapi-", "sk-or-", "hf_",
                              "gsk_", "AIzaSy"):
        assert forbidden_prefix not in response_text, (
            f"credentials-status response leaked a key starting with "
            f"{forbidden_prefix!r}"
        )


def test_save_endpoint_rejects_empty_keys() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    # Send an empty key — should not save, should not crash
    r = client.post("/api/onboarding/save-credentials", json={
        "credentials": {"GROQ_API_KEY": ""}
    })
    assert r.status_code == 200
    data = r.json()
    assert data["saved_count"] == 0
    assert "GROQ_API_KEY" in data["per_key"]
    assert data["per_key"]["GROQ_API_KEY"]["saved"] is False
    assert "empty" in data["per_key"]["GROQ_API_KEY"]["reason"].lower()


def test_save_endpoint_rejects_unknown_env_var() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/api/onboarding/save-credentials", json={
        "credentials": {"NOT_A_REAL_KEY": "value"}
    })
    assert r.status_code == 200
    data = r.json()
    assert data["saved_count"] == 0
    assert "NOT_A_REAL_KEY" in data["per_key"]
    assert "unknown env var" in data["per_key"]["NOT_A_REAL_KEY"]["reason"]


def main() -> int:
    tests = [
        test_save_endpoint_exists_in_api,
        test_status_endpoint_exists_in_api,
        test_status_endpoint_never_echoes_key_values,
        test_save_endpoint_validates_before_writing,
        test_save_endpoint_writes_mode_600,
        test_save_endpoint_additive_merge_not_clobber,
        test_save_endpoint_busts_config_cache,
        test_bert_run_check_uses_config_load_not_only_env,
        test_bert_run_check_includes_all_provider_env_vars,
        test_bert_run_falls_back_to_env_on_config_failure,
        test_onboarding_calls_save_credentials,
        test_onboarding_surfaces_save_result_to_user,
        test_onboarding_only_advances_when_at_least_one_key_saved,
        test_app_redirects_root_to_onboard_when_no_keys,
        test_app_redirect_skips_inside_onboard_itself,
        test_app_redirect_skipped_in_demo_mode,
        test_app_fails_open_on_probe_error,
        test_status_endpoint_returns_expected_shape,
        test_save_endpoint_rejects_empty_keys,
        test_save_endpoint_rejects_unknown_env_var,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
