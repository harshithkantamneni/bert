"""Smoke test for DD.3 — honest reproduce.sh + ladder disclosure.

The old framing promised "full re-execution at I.4". DD.3 retired
that as undeliverable: hosted-LLM workflows cannot deliver byte-exact
replay because providers' KV-cache, batching, and load-balancing
introduce non-determinism that bert cannot control. The honest
disclosure has two rungs: (1) cryptographic verification — live
today, catches byte-tampering, (2) structural re-evaluation —
post-investor milestone, catches semantic fraud. Byte-exact LLM
replay is NOT a rung because it is structurally impossible.

Covers:
  - `_build_reproduce_sh` body contains the two-layer honesty
  - Script body is valid /bin/sh syntactically (shell -n check)
  - findings/investor/qa.md Q5 reflects DD.3
  - findings/investor/anti_patterns.md ladder reflects DD.3
  - findings/architecture/14_glossary.md references DD.3
  - No surface still promises "I.4 reproduce.sh" / "full re-execution at I.4"
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def _require(*paths):
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip("requires lab runtime artifact(s): "
                    + ", ".join(str(m) for m in missing))


def test_build_reproduce_sh_function_exists() -> None:
    from core import proof_packet
    assert hasattr(proof_packet, "_build_reproduce_sh")


def test_reproduce_sh_body_has_two_honesty_layers() -> None:
    from core.proof_packet import _build_reproduce_sh
    body = _build_reproduce_sh(400)
    assert "Layer (a)" in body
    assert "Layer (b)" in body
    assert "CRYPTOGRAPHIC VERIFICATION" in body
    assert "EXACT-PROCESS RE-EXECUTION" in body
    assert "NOT shipped" in body


def test_reproduce_sh_body_names_structural_impossibility() -> None:
    from core.proof_packet import _build_reproduce_sh
    body = _build_reproduce_sh(400)
    # Normalize: strip shell comment '#' markers then collapse whitespace so
    # multi-line documentation comments still match phrases.
    lines = []
    for ln in body.splitlines():
        stripped = ln.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        lines.append(stripped)
    flat = " ".join(" ".join(lines).lower().split())
    assert "structurally impossible" in flat
    assert "not deterministic" in flat
    assert "kv-cache" in flat or "batching" in flat
    assert "devin-class dishonesty" in flat or \
           "would be dishonest" in flat


def test_reproduce_sh_body_offers_structural_substitute() -> None:
    from core.proof_packet import _build_reproduce_sh
    body = _build_reproduce_sh(400)
    assert "bert verify --structural" in body
    assert "semantic equivalence" in body or "cross-family judge" in body


def test_reproduce_sh_body_includes_runnable_hash_check() -> None:
    from core.proof_packet import _build_reproduce_sh
    body = _build_reproduce_sh(400)
    assert "shasum -a 256" in body
    assert "HASHES.txt" in body
    # item 25: the printed cosign command is the REAL working one
    # (`cosign verify-blob --key cosign.pub --signature HASHES.sig …`), not the
    # old non-working verify-blob-attestation/--new-bundle-format form.
    assert "cosign verify-blob" in body
    assert "--signature HASHES.sig" in body
    assert "set -e" in body  # not a placeholder echo-only script


def test_reproduce_sh_body_is_syntactically_valid_sh() -> None:
    """Pipe through `sh -n` to confirm the generated script parses
    without syntax errors. This catches HEREDOC escape bugs + missing
    quotes that would only surface when the verifier actually runs it."""
    from core.proof_packet import _build_reproduce_sh
    body = _build_reproduce_sh(400)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                       delete=False) as f:
        f.write(body)
        path = f.name
    try:
        result = subprocess.run(["sh", "-n", path],
                                  capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, (
            f"sh -n found syntax errors:\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_reproduce_sh_body_includes_cycle_id() -> None:
    """The interpolated cycle id must land in the body — verifiers
    use it to disambiguate packets in their archive."""
    from core.proof_packet import _build_reproduce_sh
    body_a = _build_reproduce_sh(400)
    body_b = _build_reproduce_sh(99)
    assert "cycle 400" in body_a
    assert "cycle 99" in body_b
    assert "cycle 400" not in body_b


def test_qa_q5_references_dd3() -> None:
    _require(LAB_ROOT / "findings" / "investor" / "qa.md")
    text = (LAB_ROOT / "findings" / "investor" / "qa.md").read_text()
    # Find Q5 section
    m = re.search(r"## 5\.\s+\"(.+?)\"(.*?)(?=^## )", text,
                    re.MULTILINE | re.DOTALL)
    assert m, "Q5 section missing"
    section = m.group(0)
    assert "DD.3" in section
    assert "structurally impossible" in section.lower()
    assert "structural re-evaluation" in section.lower()


def test_anti_patterns_ladder_references_dd3() -> None:
    _require(LAB_ROOT / "findings" / "investor" / "anti_patterns.md")
    text = (LAB_ROOT / "findings" / "investor" / "anti_patterns.md").read_text()
    m = re.search(r"## The reproducibility ladder.*?(?=^## )", text,
                    re.MULTILINE | re.DOTALL)
    assert m, "ladder section missing"
    section = m.group(0)
    assert "DD.3" in section
    assert "structural re-evaluation" in section.lower()
    assert "structurally" in section.lower()


def test_glossary_references_dd3() -> None:
    _require(LAB_ROOT / "findings" / "architecture" / "14_glossary.md")
    text = (LAB_ROOT / "findings" / "architecture" / "14_glossary.md").read_text()
    assert "DD.3" in text
    assert "structurally impossible" in text.lower()


def test_no_surface_still_promises_i4_reproduce_sh() -> None:
    """The retired I.4-promise framings must NOT appear as positive
    promises in the canonical investor surfaces. Historical mentions
    (e.g. "earlier docs called this 'X'; DD.3 retired that framing")
    are allowed — they're the receipt of the change. The check uses
    a context window: a phrase is forbidden if it appears WITHOUT a
    "retired" / "misleading" / "DD.3" / "earlier" cue within 120 chars."""
    surfaces = [
        "findings/investor/qa.md",
        "findings/investor/anti_patterns.md",
        "findings/architecture/14_glossary.md",
    ]
    _require(*(LAB_ROOT / rel for rel in surfaces))
    forbidden_phrases = [
        "Milestone I.4 (re-execution layer)",
        "reproduce.sh ships in I.4",
        "full re-execution at I.4",
        "full re-execution runs once I.4 ships",
        "I.4 milestone where re-execution ships",
    ]
    history_cues = ("retired", "misleading", "DD.3", "earlier framing",
                    "earlier docs", "old framing")
    for rel in surfaces:
        text = (LAB_ROOT / rel).read_text()
        for phrase in forbidden_phrases:
            idx = 0
            while True:
                i = text.find(phrase, idx)
                if i < 0:
                    break
                window = text[max(0, i - 120):i + len(phrase) + 120]
                assert any(cue in window for cue in history_cues), (
                    f"{rel} contains {phrase!r} at offset {i} WITHOUT a "
                    f"history cue nearby; DD.3 retired this as a positive "
                    f"promise. Window: ...{window!r}..."
                )
                idx = i + len(phrase)


def test_proof_packet_calls_new_builder() -> None:
    """build_packet must call _build_reproduce_sh (not inline the old
    placeholder body)."""
    text = (LAB_ROOT / "core" / "proof_packet.py").read_text()
    assert "_build_reproduce_sh(cycle_id)" in text
    # Old placeholder phrasing should be gone from the runtime path
    assert "real reproduce.sh ships in I.4\\n" not in text


def main() -> int:
    tests = [
        test_build_reproduce_sh_function_exists,
        test_reproduce_sh_body_has_two_honesty_layers,
        test_reproduce_sh_body_names_structural_impossibility,
        test_reproduce_sh_body_offers_structural_substitute,
        test_reproduce_sh_body_includes_runnable_hash_check,
        test_reproduce_sh_body_is_syntactically_valid_sh,
        test_reproduce_sh_body_includes_cycle_id,
        test_qa_q5_references_dd3,
        test_anti_patterns_ladder_references_dd3,
        test_glossary_references_dd3,
        test_no_surface_still_promises_i4_reproduce_sh,
        test_proof_packet_calls_new_builder,
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
