"""Smoke test for Q.3: anti-patterns defensive doc.

Validates structure + anchor reality. Same anti-Devin discipline as
Q.2: a document that says "we don't do X" must point at a real file
where the alternative shows up.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
DOC = LAB_ROOT / "findings" / "investor" / "anti_patterns.md"


def test_doc_exists() -> None:
    assert DOC.exists(), "findings/investor/anti_patterns.md missing"


def test_doc_names_six_anti_patterns() -> None:
    text = DOC.read_text()
    for i in range(1, 7):
        assert re.search(rf"^### {i}\.", text, re.MULTILINE), \
            f"anti-pattern #{i} missing or malformed (expected '### {i}. ...')"


def test_doc_each_section_has_pattern_and_alternative() -> None:
    text = DOC.read_text()
    pattern_count = len(re.findall(r"\*\*The pattern\.\*\*", text))
    alt_count = len(re.findall(r"\*\*What bert does instead\.\*\*", text))
    why_count = len(re.findall(r"\*\*Why it fails in 2026\.\*\*", text))
    verifiable_count = len(re.findall(r"\*\*Verifiable\.\*\*", text))
    assert pattern_count == 6, f"expected 6 'The pattern' blocks; got {pattern_count}"
    assert alt_count == 6, f"expected 6 'What bert does instead' blocks; got {alt_count}"
    assert why_count == 6, f"expected 6 'Why it fails in 2026' blocks; got {why_count}"
    assert verifiable_count == 6, f"expected 6 'Verifiable' blocks; got {verifiable_count}"


def test_doc_references_devin_and_berkeley_context() -> None:
    text = DOC.read_text()
    assert "Devin" in text, "doc must reference Devin demo collapse"
    assert "Berkeley" in text, "doc must reference UC Berkeley paper"
    assert "February 2026" in text, "doc must date Devin event"
    assert "April 2026" in text, "doc must date Berkeley event"


def test_doc_includes_reproducibility_ladder_disclosure() -> None:
    """Per DD.3 (2026-05-17): the ladder no longer promises "I.4 full
    re-execution" because byte-exact hosted-LLM replay is structurally
    impossible. The doc must instead frame the two rungs as detecting
    distinct fraud classes (byte-tampering vs semantic) and explicitly
    say byte-exact LLM replay is undeliverable.
    """
    text = DOC.read_text()
    assert "reproducibility ladder" in text.lower(), \
        "doc must contain the 'reproducibility ladder' section heading"
    assert "structural re-evaluation" in text.lower(), \
        "doc must offer structural re-evaluation as the rung-2 substitute"
    assert ("structurally undeliverable" in text.lower() or
            "structurally cannot" in text.lower()), \
        "doc must name the structural impossibility of byte-exact LLM replay"
    assert "DD.3" in text, "doc must reference DD.3 as the revision marker"


def test_doc_includes_anti_devin_demo_pattern() -> None:
    """Anti-pattern #1 must directly address Devin's demo collapse."""
    text = DOC.read_text()
    section_1 = re.search(r"### 1\..*?### 2\.", text, re.DOTALL)
    assert section_1, "section 1 not found"
    body = section_1.group(0)
    assert "live" in body.lower(), "anti-pattern #1 must emphasize live demo"
    assert "signed" in body.lower() or "packet" in body.lower(), \
        "anti-pattern #1 must reference signed packet as the leave-behind"


def test_doc_includes_reading_checklist() -> None:
    """The 6-step partner-can-verify checklist is the doc's payoff section."""
    text = DOC.read_text()
    assert "Reading checklist" in text or "reading checklist" in text, \
        "doc must include the partner reading checklist"
    assert "cosign verify-blob" in text, \
        "checklist must show the (real, working) cosign command"
    assert "failures.md" in text, \
        "checklist must point at the failures.md disclosure file"


def test_doc_anchors_exist_on_disk() -> None:
    expected_anchors = [
        "findings/proof_packets/cycle-0400.tar.gz",
        "findings/weekly_quality_report_2026-05-13.md",
        "findings/falsifier_corpus.md",
        "findings/falsifier_baseline_C400_all-time.md",
        "findings/investor/one_pager.md",
        "findings/investor/qa.md",
        "findings/investor/launch_essay.md",
        "findings/investor/pitch_deck.md",
        "findings/investor/demo_recording/README.md",
        "findings/investor/demo_recording/storyboard.md",
        "core/router.py",
    ]
    text = DOC.read_text()
    for anchor in expected_anchors:
        basename = anchor.rsplit("/", 1)[-1]
        mentioned = anchor in text or basename in text
        assert mentioned, f"anti_patterns.md should cite {anchor} (or basename) but doesn't"
        assert (LAB_ROOT / anchor).exists(), \
            f"anti_patterns.md cites {anchor} but file doesn't exist — anchor fabrication"


def test_doc_does_not_recite_benchmark_scores() -> None:
    """The doc argues against benchmark-citation; it must not itself cite
    bert's score on SWE-bench / WebArena / GAIA / etc. (Mentioning the
    *benchmark names* in the context of why we don't run them is fine; an
    actual bert-vs-benchmark percentage is not.)"""
    text = DOC.read_text()
    # No "bert scored X% on Y" patterns
    forbidden = re.findall(r"(?:bert|bert)[^.\n]*\b\d{1,3}(?:\.\d+)?%\s*(?:on)?\s*(?:SWE-bench|WebArena|GAIA|OSWorld|Terminal-Bench)", text)
    assert not forbidden, \
        f"doc must not cite a bert benchmark score; found: {forbidden}"


# ── S.2 depth audit: per-anti-pattern verification ──────────────────
# Each of the 6 sections must have the right specific defense AND
# anchor at a real file. Surface check of section count isn't enough.

def _anti_section(text: str, n: int) -> str:
    """Slice section N (between '### N.' and '### N+1.' or '## ' if last)."""
    pattern = rf"^### {n}\.(.*?)(?=^### {n+1}\.|^## (?!#)|\Z)"
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not m:
        raise AssertionError(f"could not slice anti-pattern #{n}")
    return m.group(0)


def test_anti1_edited_demos_advocates_live_first() -> None:
    """Anti-pattern #1 is about edited recorded demos. Defense must be
    live-first + signed packet leave-behind."""
    s = _anti_section(DOC.read_text(), 1)
    assert "live" in s.lower(), "#1 defense must emphasize live demos"
    assert "signed" in s.lower() or "packet" in s.lower(), \
        "#1 must reference signed packet as leave-behind"
    assert "tar" in s.lower() or "cosign" in s.lower() or "HASHES" in s, \
        "#1 must anchor at a verifiable artifact (tar/cosign/HASHES)"


def test_anti2_benchmarks_promotes_weekly_self_grade() -> None:
    """Anti-pattern #2 is about citing benchmark scores. Defense must
    be the weekly self-grade with honest C-grade disclosure."""
    s = _anti_section(DOC.read_text(), 2)
    assert "SWE-bench" in s or "benchmark" in s.lower(), \
        "#2 must name the benchmarks problem"
    assert "weekly" in s.lower(), "#2 defense must reference weekly self-grade"
    assert "axes" in s.lower() or "eight" in s.lower(), \
        "#2 must reference the 8-axis grade structure"
    assert "C" in s, "#2 must reference the honest C-grade disclosure"
    assert "weekly_quality_report" in s, "#2 must anchor at the real report file"


def test_anti3_llm_judge_promotes_cross_family_review() -> None:
    """Anti-pattern #3 is LLM-as-judge same-family. Defense: cross-family
    routing + adversarial-eval-by-design. MUST disclose heuristic-v1."""
    s = _anti_section(DOC.read_text(), 3)
    assert "Qwen" in s or "cross-family" in s.lower(), \
        "#3 must cite cross-family routing"
    assert "adversarial" in s.lower(), \
        "#3 defense must reference adversarial-eval"
    assert "heuristic-v1" in s, \
        "#3 MUST disclose heuristic-v1 status (anti-Devin)"
    assert "I.4" in s, "#3 must name I.4 milestone for LLM-driven v2"
    assert "adversarial.json" in s, "#3 must anchor at adversarial.json"


def test_anti4_vague_reliability_promotes_numeric_grades() -> None:
    """Anti-pattern #4 is vague reliability claims. Defense must include
    numeric grades + falsifier baseline + Gartner cancellation reference."""
    s = _anti_section(DOC.read_text(), 4)
    assert "Gartner" in s, \
        "#4 must reference Gartner cancellation prediction as ammunition"
    assert "14" in s, "#4 must cite the 14 falsifier targets"
    assert "PASS" in s or "INSUFFICIENT" in s, \
        "#4 must include the actual PASS/INSUFFICIENT numbers"
    assert "falsifier_corpus" in s or "falsifier_baseline" in s, \
        "#4 must anchor at the falsifier baseline file"


def test_anti5_talker_framing_promotes_workflow_ownership() -> None:
    """Anti-pattern #5 is 'AI-powered X' talker framing. Defense must be
    workflow ownership (threshing/clearness/seasoning)."""
    s = _anti_section(DOC.read_text(), 5)
    assert "workflow" in s.lower(), "#5 defense must frame in workflow terms"
    assert "threshing" in s.lower() or "clearness" in s.lower(), \
        "#5 must name the actual pipeline stages"
    assert "talker" in s.lower(), "#5 must use the locked 'talker' framing"


def test_anti6_recorded_only_promotes_live_with_fallback() -> None:
    """Anti-pattern #6 is recorded-only demos. Defense: live-first with
    Meta/Devin precedent context."""
    s = _anti_section(DOC.read_text(), 6)
    assert "recorded" in s.lower(), "#6 must name the recorded-only problem"
    assert "live" in s.lower(), "#6 defense must emphasize live demos"
    assert "Meta" in s or "Devin" in s, \
        "#6 must cite Meta or Devin precedent for the trust collapse"
    # Local-first install on partner machine is the locked counter
    assert "curl" in s.lower() or "partner" in s.lower() or "local-first" in s.lower(), \
        "#6 must reference the local-first install or partner-machine pattern"


def test_reading_checklist_has_six_steps() -> None:
    """The 6-step reading checklist is the partner-can-verify payoff. Each
    step must be numbered AND include the actual command the partner runs."""
    text = DOC.read_text()
    # The checklist is in the "Reading checklist" section
    m = re.search(r"^## Reading checklist.*?(?=^## )", text,
                  re.MULTILINE | re.DOTALL)
    assert m, "Reading checklist section not found"
    checklist = m.group(0)
    # Six numbered steps
    steps = re.findall(r"^\d+\.\s+`", checklist, re.MULTILINE)
    assert len(steps) >= 6, \
        f"reading checklist must have at least 6 numbered steps with backtick commands; got {len(steps)}"
    # Specific commands the partner runs
    expected_commands = ["tar -tzf", "cosign verify", "cat", "jq"]
    for cmd in expected_commands:
        assert cmd in checklist, \
            f"reading checklist must include the '{cmd}' command"


def test_each_anti_pattern_has_real_anchor_file() -> None:
    """Stronger than anchors_exist_on_disk — every Verifiable block must
    name at least one file path that resolves. Reads the Verifiable block
    of each of the 6 sections."""
    text = DOC.read_text()
    for i in range(1, 7):
        s = _anti_section(text, i)
        # Find the Verifiable block within the section
        v_match = re.search(r"\*\*Verifiable\.\*\*(.+?)(?=\n\n|\Z)", s, re.DOTALL)
        assert v_match, f"section #{i} missing Verifiable block content"
        verifiable_text = v_match.group(1)
        # Find at least one file-path-shaped token (allow - and . in path
        # bodies so we don't truncate dated filenames like
        # 'weekly_quality_report_2026-05-13.md' into '13.md').
        paths = re.findall(r"[\w./-]+\.(?:py|md|json|sigstore|sh|tar\.gz|yaml|ts|tsx|html)\b",
                           verifiable_text)
        # Filter out matches that don't include a leading letter — those
        # are typically date-fragment artifacts.
        paths = [p for p in paths if re.match(r"^[A-Za-z_]", p)]
        assert paths, \
            f"section #{i} Verifiable block must name at least one file; got: {verifiable_text[:200]}"
        # At least one of them must exist on disk. Check the path
        # as-cited first, then a few likely parent directories.
        any_exists = False
        prefixes = (
            "",
            "findings/proof_packets/",
            "cycle-0400/",
            "findings/investor/",
            "findings/investor/demo_recording/",
            "findings/",
            "core/",
            "tools/",
            "tests/",
            "api/",
            "landing/",
        )
        for p in paths:
            for prefix in prefixes:
                if (LAB_ROOT / f"{prefix}{p}").exists():
                    any_exists = True
                    break
            if any_exists:
                break
        assert any_exists, \
            f"section #{i} Verifiable cites paths {paths} but none exist on disk"


def test_reproducibility_ladder_names_both_layers() -> None:
    """Per DD.3 (2026-05-17): the two rungs are (1) cryptographic
    verification — live today, catches byte-tampering — and (2) structural
    re-evaluation — post-investor milestone, catches semantic fraud.
    Byte-exact LLM replay is explicitly NOT a rung because it is
    structurally impossible on hosted-LLM stacks. The ladder must say so.
    """
    text = DOC.read_text()
    m = re.search(r"## The reproducibility ladder.*?(?=^## )", text,
                  re.MULTILINE | re.DOTALL)
    assert m, "Reproducibility ladder section missing"
    ladder = m.group(0)
    # Rung 1 (cryptographic, live today)
    assert "SLSA" in ladder, "ladder must name SLSA in cryptographic rung"
    assert "Sigstore" in ladder, "ladder must name Sigstore"
    assert "cosign" in ladder, "ladder must reference cosign"
    # Rung 2 (structural re-evaluation, post-investor)
    assert "structural re-evaluation" in ladder.lower(), \
        "ladder must name the structural-re-evaluation rung"
    assert "bert verify --structural" in ladder, \
        "ladder must reference the post-investor verifier subcommand"
    # The retired "I.4 reproduce.sh" framing must NOT be in the ladder
    assert "structurally cannot" in ladder.lower() or \
           "structurally undeliverable" in ladder.lower(), \
        "ladder must name the structural impossibility of byte-exact LLM replay"
    assert "DD.3" in ladder, "ladder must reference DD.3 revision marker"


def main() -> int:
    tests = [
        test_doc_exists,
        test_doc_names_six_anti_patterns,
        test_doc_each_section_has_pattern_and_alternative,
        test_doc_references_devin_and_berkeley_context,
        test_doc_includes_reproducibility_ladder_disclosure,
        test_doc_includes_anti_devin_demo_pattern,
        test_doc_includes_reading_checklist,
        test_doc_anchors_exist_on_disk,
        test_doc_does_not_recite_benchmark_scores,
        # ── S.2 per-anti-pattern depth ──
        test_anti1_edited_demos_advocates_live_first,
        test_anti2_benchmarks_promotes_weekly_self_grade,
        test_anti3_llm_judge_promotes_cross_family_review,
        test_anti4_vague_reliability_promotes_numeric_grades,
        test_anti5_talker_framing_promotes_workflow_ownership,
        test_anti6_recorded_only_promotes_live_with_fallback,
        test_reading_checklist_has_six_steps,
        test_each_anti_pattern_has_real_anchor_file,
        test_reproducibility_ladder_names_both_layers,
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
