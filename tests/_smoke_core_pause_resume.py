"""Smoke: core/pause_resume.py — HMAC resume tokens + paused state (49%).

Pure crypto + file state, fully network-free. Covers mint↔verify round
trip, every verify rejection path (empty / no-separator / bad-base64 /
tampered-sig / expired / malformed-json), the NeedsUserInput envelope,
save/list/clear persistence (incl. expired-file cleanup), build_envelope,
and the mint/verify/list/usage/unknown CLI commands.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import pause_resume as pr  # noqa: E402


def test_mint_verify_roundtrip():
    st = pr.PausedState(lab="demo", cycle=4, step_id="fork1",
                        saved_state={"draft": "v1"})
    token = pr.mint_resume_token(st)
    assert "." in token
    back = pr.verify_resume_token(token)
    assert back is not None
    assert back.lab == "demo" and back.cycle == 4 and back.step_id == "fork1"
    assert back.saved_state == {"draft": "v1"}


def test_verify_rejection_paths():
    assert pr.verify_resume_token("") is None
    assert pr.verify_resume_token("no-separator-here") is None
    assert pr.verify_resume_token("!!!notbase64.deadbeef") is None
    # tampered signature
    token = pr.mint_resume_token(pr.PausedState(lab="x", cycle=1, step_id="s"))
    b64, _sig = token.rsplit(".", 1)
    assert pr.verify_resume_token(f"{b64}.0000bad") is None
    # expired (set a positive past expiry so mint doesn't reset it)
    expired = pr.PausedState(lab="x", cycle=1, step_id="s",
                             created_at_ts=time.time() - 1000,
                             expires_at_ts=time.time() - 100)
    et = pr.mint_resume_token(expired)
    assert pr.verify_resume_token(et) is None


def test_needs_user_input_envelope():
    env = pr.NeedsUserInput(
        question="Pick one",
        options=(pr.Option(value="a", label="A", cost_usd_est=0.1, risk_level="low"),),
        rationale="because", resume_token="tok")
    d = env.to_envelope()
    assert d["status"] == "needs_user_input"
    assert d["options"][0]["value"] == "a" and d["resume_token"] == "tok"


def test_save_list_clear(tmp_path):
    st = pr.PausedState(lab="demo", cycle=2, step_id="stepA",
                        created_at_ts=time.time(),
                        expires_at_ts=time.time() + 3600)
    p = pr.save_paused_state(tmp_path, st)
    assert p.exists()
    pending = pr.list_pending(tmp_path)
    assert len(pending) == 1 and pending[0]["step_id"] == "stepA"
    assert pr.clear_paused(tmp_path, "stepA") is True
    assert pr.list_pending(tmp_path) == []
    assert pr.clear_paused(tmp_path, "nonexistent") is False


def test_list_pending_cleans_expired(tmp_path):
    d = pr._paused_dir(tmp_path)
    expired = {"lab": "x", "cycle": 1, "step_id": "old",
               "saved_state": {}, "created_at_ts": 0.0,
               "expires_at_ts": time.time() - 10}
    (d / "0_old.json").write_text(json.dumps(expired))
    # also a malformed file (must be skipped, not crash)
    (d / "9_bad.json").write_text("{not json")
    assert pr.list_pending(tmp_path) == []        # expired removed, bad skipped
    assert not (d / "0_old.json").exists()


def test_build_envelope_persists_and_signs(tmp_path):
    env = pr.build_envelope(
        lab="demo", cycle=5, step_id="fork9", question="Continue?",
        options=[pr.Option(value="yes", label="Yes")],
        rationale="fork", saved_state={"k": "v"}, lab_path=tmp_path)
    assert env.resume_token and pr.verify_resume_token(env.resume_token) is not None
    assert len(pr.list_pending(tmp_path)) == 1


def test_cli(tmp_path):
    assert pr._cli(["x"]) == 2                              # usage
    assert pr._cli(["x", "mint"]) == 2                      # mint usage
    assert pr._cli(["x", "mint", "demo", "3", "stepZ"]) == 0
    assert pr._cli(["x", "verify"]) == 2                    # verify usage
    assert pr._cli(["x", "verify", "garbage.token"]) == 1   # INVALID
    token = pr.mint_resume_token(pr.PausedState(lab="d", cycle=1, step_id="s"))
    assert pr._cli(["x", "verify", token]) == 0
    assert pr._cli(["x", "list"]) == 2                      # list usage
    assert pr._cli(["x", "list", str(tmp_path)]) == 0
    assert pr._cli(["x", "bogus"]) == 2                     # unknown


def main() -> int:
    tests = [
        test_mint_verify_roundtrip,
        test_verify_rejection_paths,
        test_needs_user_input_envelope,
        test_save_list_clear,
        test_list_pending_cleans_expired,
        test_build_envelope_persists_and_signs,
        test_cli,
    ]
    for t in tests:
        td = Path(tempfile.mkdtemp())
        try:
            kwargs = {"tmp_path": td} if "tmp_path" in inspect.signature(t).parameters else {}
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        finally:
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
