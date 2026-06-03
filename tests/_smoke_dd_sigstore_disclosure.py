"""Smoke test for DD.2 — honest Sigstore disclosure.

Verifies:
  - No surface in the canonical investor materials still says
    "one config flip" / "one flip" for Sigstore migration
  - core/signing.py docstring documents the engineering punch list
  - core/proof_packet.py:_build_sigstore_bundle docstring is honest
  - findings/architecture/14_glossary.md reflects DD.2 + DD.1 + DD.3
  - Setting BERT_LAB_SIGNING_MODE=sigstore emits a runtime warning
    via logging (NOT a hard error — operators may still want to tag
    bundles as sigstore-mode for staging environments)
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


CANONICAL_SURFACES = (
    "core/signing.py",
    "core/proof_packet.py",
    "findings/architecture/14_glossary.md",
    "findings/architecture/05_signing.md",
    "findings/investor/demo_recording/storyboard.md",
    "findings/investor/demo_recording/dry_run_2026-05-13.md",
    "findings/investor/demo_recording/demo_run.sh",
    "findings/investor/demo_recording/narration.md",
    "tools/record_explainer.py",
)


def test_no_one_config_flip_in_canonical_surfaces() -> None:
    """The misleading "one config flip" framing must be GONE from
    every canonical surface listed above. Historical mentions are
    allowed in change logs / dry-run docs as long as they're framed
    as "earlier docs said X; DD.2 retired that framing"."""
    for rel in CANONICAL_SURFACES:
        p = LAB_ROOT / rel
        assert p.exists(), f"{rel} missing — smoke needs to be updated"
        text = p.read_text()
        for forbidden in ("is one config flip", "is one flip",
                          "migration is one config flip",
                          "Sigstore migration is one config flip"):
            assert forbidden not in text, (
                f"{rel} still contains forbidden phrase {forbidden!r}; "
                "DD.2 retired the one-flip framing"
            )


def test_signing_module_documents_engineering_punch_list() -> None:
    """The honest-disclosure punch list must be in core/signing.py
    so a reader who lands there from a grep sees the real cost."""
    text = (LAB_ROOT / "core" / "signing.py").read_text()
    # Specific terms from the punch list — easier to grep than long sentences
    for needle in ("sigstore-python", "OIDC", "Fulcio cert",
                   "Rekor entry submission", "RFC3161",
                   "verify_bytes", "engineering"):
        assert needle in text, f"signing.py docstring missing {needle!r}"


def test_signing_module_documents_tag_only_today() -> None:
    """The docstring must say sigstore mode is currently a TAG, not
    a code path."""
    text = (LAB_ROOT / "core" / "signing.py").read_text()
    assert "TAG" in text  # explicit
    assert "Cryptographic operations remain local ed25519" in text


def test_proof_packet_bundle_docstring_is_honest() -> None:
    text = (LAB_ROOT / "core" / "proof_packet.py").read_text()
    assert "transparency-log entries are empty" in text
    assert "RFC3161 timestamps are empty" in text
    assert "Wire-format compatibility is intentional" in text
    assert "production Sigstore being implemented" in text


def test_glossary_documents_dd2() -> None:
    text = (LAB_ROOT / "findings" / "architecture" / "14_glossary.md").read_text()
    assert "DD.2 retired" in text
    assert "engineering work" in text
    # And while we're here — DD.1 + DD.3 glossary entries should also exist
    assert "llm-driven-v2" in text
    assert "DD.3" in text


def test_signing_mode_sigstore_emits_runtime_warning() -> None:
    """Setting BERT_LAB_SIGNING_MODE=sigstore must log a one-time
    warning explaining that the flag is a tag, not a switchover."""
    # Reset the warn-emitted guard
    import importlib
    from core import signing as sig
    importlib.reload(sig)

    # Capture WARNING from the bert.signing logger
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    sig.LOG.addHandler(handler)
    prev_level = sig.LOG.level
    sig.LOG.setLevel(logging.WARNING)
    prev_env = os.environ.get("BERT_LAB_SIGNING_MODE")
    try:
        os.environ["BERT_LAB_SIGNING_MODE"] = "sigstore"
        mode = sig._signing_mode()
        assert mode == "sigstore"
        out = buf.getvalue()
        assert "TAG only" in out or "tag only" in out.lower()
        assert "ed25519" in out
        # And the guard must prevent a second emission
        buf.truncate(0)
        buf.seek(0)
        sig._signing_mode()
        assert buf.getvalue() == ""
    finally:
        sig.LOG.removeHandler(handler)
        sig.LOG.setLevel(prev_level)
        if prev_env is None:
            os.environ.pop("BERT_LAB_SIGNING_MODE", None)
        else:
            os.environ["BERT_LAB_SIGNING_MODE"] = prev_env


def test_signing_mode_default_is_local_dev() -> None:
    import importlib
    from core import signing as sig
    importlib.reload(sig)
    prev_env = os.environ.get("BERT_LAB_SIGNING_MODE")
    os.environ.pop("BERT_LAB_SIGNING_MODE", None)
    try:
        assert sig._signing_mode() == "local-dev"
    finally:
        if prev_env is not None:
            os.environ["BERT_LAB_SIGNING_MODE"] = prev_env


def test_demo_narration_mentions_engineering_not_flag() -> None:
    """Every demo recording surface must frame production Sigstore as
    engineering / commercial roadmap, not a flag."""
    for rel in ("findings/investor/demo_recording/storyboard.md",
                "findings/investor/demo_recording/demo_run.sh"):
        text = (LAB_ROOT / rel).read_text()
        assert "real engineering" in text or "engineering" in text, (
            f"{rel} narration should frame Sigstore migration as "
            "engineering, not a flag")
        assert ("commercial roadmap" in text or
                "not a flag we flip" in text), (
            f"{rel} should explicitly say production Sigstore is NOT a "
            "demo-day flag")


def main() -> int:
    tests = [
        test_no_one_config_flip_in_canonical_surfaces,
        test_signing_module_documents_engineering_punch_list,
        test_signing_module_documents_tag_only_today,
        test_proof_packet_bundle_docstring_is_honest,
        test_glossary_documents_dd2,
        test_signing_mode_sigstore_emits_runtime_warning,
        test_signing_mode_default_is_local_dev,
        test_demo_narration_mentions_engineering_not_flag,
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
