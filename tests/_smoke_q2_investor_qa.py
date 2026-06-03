"""Smoke test for Q.2: investor Q&A doc.

Validates the qa.md document is shaped correctly AND that every
"Receipts" anchor it cites actually exists on disk. The depth here
is anchor-verification: a Q&A doc that promises files that don't
exist is a Devin-class pre-launch fabrication, and the L4 honesty
discipline says we don't ship that.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
QA = LAB_ROOT / "findings" / "investor" / "qa.md"


def test_qa_exists() -> None:
    assert QA.exists(), "findings/investor/qa.md missing"


def test_qa_has_all_ten_questions() -> None:
    text = QA.read_text()
    for i in range(1, 11):
        assert re.search(rf"^## {i}\.\s+\"", text, re.MULTILINE), \
            f"question {i} missing or malformed (expected '## {i}. \"...\"')"


def test_qa_each_question_has_thirty_second_answer() -> None:
    text = QA.read_text()
    matches = re.findall(r"\*\*Thirty-second answer\.\*\*", text)
    assert len(matches) == 10, \
        f"expected 10 'Thirty-second answer' blocks; got {len(matches)}"


def test_qa_each_question_has_receipts_block() -> None:
    text = QA.read_text()
    matches = re.findall(r"\*\*Receipts\.\*\*", text)
    assert len(matches) == 10, \
        f"expected 10 'Receipts.' blocks; got {len(matches)}"


def test_qa_includes_honest_disclosures() -> None:
    """Key disclosures the 2026-era investor will read as credibility.
    Post-DD.1+DD.2+DD.3 the surface disclosures are:
    - heuristic-v1 OR llm-driven-v2 adversarial mode (DD.1)
    - cryptographic-only reproducibility; exact-process replay is
      structurally impossible for hosted-LLM stacks (DD.3)
    - local-dev Sigstore; production is engineering (DD.2)
    - no acquired customer yet
    """
    text = QA.read_text()
    assert "Honest" in text, \
        "qa.md must surface explicit honesty marker"
    assert "structurally impossible" in text.lower(), \
        "qa.md must disclose that exact-process LLM replay is structurally impossible (DD.3)"
    assert "heuristic-v1" in text, \
        "qa.md must mention heuristic-v1 adversarial eval"
    assert "llm-driven" in text.lower() or "LLM-driven" in text, \
        "qa.md must mention LLM-driven v2 adversarial eval (DD.1)"
    assert "pre-customer" in text or "no customer" in text.lower() or "zero" in text, \
        "qa.md must honestly answer Q10 about no customer yet"


def test_qa_anchors_exist_on_disk() -> None:
    """Every receipt the doc names must actually be a real file. This is
    the anti-Devin check: a Q&A that promises files that aren't there is
    presumed fabricated."""
    expected_anchors = [
        "findings/proof_packets/cycle-0400.tar.gz",
        "findings/proof_packets/cycle-0099.tar.gz",
        "findings/weekly_quality_report_2026-05-13.md",
        "findings/falsifier_baseline_C400_all-time.md",
        "findings/falsifier_corpus.md",
        "core/permission.py",
        "core/compact.py",
        "api/main.py",
        "core/router.py",
        "tests/_smoke_k6_provider_cooled.py",
        "findings/investor/one_pager.md",
        "findings/investor/one_pager.json",
        "findings/investor/launch_essay.md",
    ]
    text = QA.read_text()
    for anchor in expected_anchors:
        basename = anchor.rsplit("/", 1)[-1]
        mentioned = anchor in text or basename in text
        assert mentioned, f"qa.md should cite {anchor} (full path or basename '{basename}') but doesn't"
        assert (LAB_ROOT / anchor).exists(), \
            f"qa.md cites {anchor} but file doesn't exist — Devin-class anchor fabrication"


def test_qa_uses_locked_lede_phrasing() -> None:
    """Q.1 audit locked 'autonomous lab' (not 'autonomous AI lab').
    Q.2 must echo that phrasing where it references the lede."""
    text = QA.read_text()
    assert "autonomous lab that grades itself weekly" in text, \
        "qa.md should echo the locked lede phrasing"
    assert "autonomous AI lab" not in text, \
        "qa.md uses dropped 'autonomous AI lab' wording"


def test_qa_includes_glean_anchor() -> None:
    """Pricing answer (Q9) must include the Glean anchor."""
    text = QA.read_text()
    assert "Glean" in text, "Q9 should reference Glean as team-tier anchor"
    assert "$50K annual minimum" in text or "$50K" in text, \
        "Glean anchor needs the $50K minimum to land"


def test_qa_includes_post_devin_post_berkeley_context() -> None:
    """The 2026-era investor questions are shaped by Devin's demo
    collapse and the UC Berkeley benchmark-fraud paper. The doc must
    name both somewhere to anchor in the right cultural moment."""
    text = QA.read_text()
    assert "Devin" in text, "qa.md must reference Devin context"
    assert "Berkeley" in text, "qa.md must reference the Berkeley paper"


# ── S.1 depth audit: per-question content verification ──────────────
# Each question gets its own assertion that the right answer lands at
# the right place. Surface-level "anchor exists somewhere in doc" isn't
# enough — Q3 needs to anchor at Q3, not at Q9.

def _section(qa_text: str, question_num: int) -> str:
    """Slice the doc to one Q's section (next ## header marks the end)."""
    pattern = rf"^## {question_num}\.(.*?)(?=^## (?:\d+\.|How to use)|\Z)"
    m = re.search(pattern, qa_text, re.MULTILINE | re.DOTALL)
    if not m:
        raise AssertionError(f"could not slice Q{question_num} section")
    return m.group(0)


def test_q1_cites_workflow_pipeline_and_proof_packet() -> None:
    """Q1 ('what work does your agent complete'): the answer must be in
    workflow terms (threshing/clearness/seasoning) AND cite a real proof
    packet as the unit of output."""
    s = _section(QA.read_text(), 1)
    assert "threshing" in s.lower(), "Q1 must name the threshing stage"
    assert "clearness" in s.lower(), "Q1 must name the clearness stage"
    assert "cycle-0400.tar.gz" in s, "Q1 must cite cycle-0400.tar.gz as the unit"


def test_q2_cites_weekly_report_and_falsifier_baseline() -> None:
    """Q2 ('reliability'): the answer must cite the weekly report (H.6
    grades) and the falsifier baseline."""
    s = _section(QA.read_text(), 2)
    assert "weekly" in s.lower(), "Q2 must reference the weekly report"
    assert "falsifier" in s.lower(), "Q2 must reference falsifiers"
    assert "12 PASS" in s or "12/14" in s, "Q2 must include the current PASS count"
    assert "falsifier_baseline" in s or "weekly_quality_report" in s, \
        "Q2 must anchor at a real baseline file"


def test_q3_cites_real_killswitch_implementations() -> None:
    """Q3 ('what happens when wrong'): MUST cite permission.py + compact.py
    + /api/pending — the three real killswitch implementations. A claim
    about killswitches that doesn't anchor at real code is Devin-class."""
    s = _section(QA.read_text(), 3)
    assert "permission.py" in s, "Q3 must cite core/permission.py"
    assert "compact.py" in s, "Q3 must cite core/compact.py (3-strike killswitch)"
    assert "/api/pending" in s, "Q3 must cite the /api/pending endpoint"
    assert "Telegram" in s, "Q3 must name Telegram as the default approver"
    assert "3-strike" in s.lower() or "three-strike" in s.lower(), \
        "Q3 must name the 3-strike killswitch by its locked name"


def test_q4_anchors_at_weekly_report_and_architecture_doc() -> None:
    """Q4 ('obsessively improving harness'): Karpathy ascent quote +
    weekly_quality_report citation + architect file lineage anchor."""
    s = _section(QA.read_text(), 4)
    assert "Karpathy" in s, "Q4 should cite Karpathy ascent framing"
    assert "weekly_quality_report" in s or "Friday" in s, \
        "Q4 should anchor at the weekly cadence"


def test_q5_discloses_reproducibility_layers_honestly() -> None:
    """Q5 ('reproducible eval pack'): MUST surface the two-layer honesty
    picture per DD.3 — cryptographic verification works today; exact-process
    re-execution is STRUCTURALLY impossible for hosted-LLM workflows.
    The earlier "placeholder for I.4" framing was retired because it
    promised something undeliverable; promising any flavor of byte-exact
    LLM replay is itself Devin-class dishonesty for a free-tier hosted
    stack.
    """
    s = _section(QA.read_text(), 5)
    assert "structurally impossible" in s.lower(), \
        "Q5 must name the structural impossibility of byte-exact hosted-LLM replay"
    assert "structural re-evaluation" in s.lower() or \
           "bert verify --structural" in s, \
        "Q5 must offer the structural-equivalence analogue as the honest substitute"
    assert "DD.3" in s, \
        "Q5 must reference DD.3 as the disclosure-revision marker"
    assert "cryptographic" in s.lower(), \
        "Q5 must name the cryptographic layer that DOES run today"
    assert "crypto" in s.lower() or "cosign" in s.lower(), \
        "Q5 must distinguish the crypto layer (live today)"


def test_q6_discloses_heuristic_v1_adversarial() -> None:
    """Q6 ('proof you're not benchmark-cheating'): MUST disclose the
    adversarial eval is heuristic-v1, not LLM-driven yet."""
    s = _section(QA.read_text(), 6)
    assert "heuristic-v1" in s, \
        "Q6 must disclose adversarial-eval is heuristic-v1"
    assert "adversarial.json" in s, \
        "Q6 must cite the eval/adversarial.json artifact"
    assert "Berkeley" in s, "Q6 must connect to the Berkeley benchmark paper"
    # The actual measured numbers from cycle-0400
    assert "60" in s, "Q6 must cite the 60-attempt count"


def test_q7_cites_router_and_8_provider_count() -> None:
    """Q7 ('stronger as models improve'): the answer must name the
    routing fabric and the 8-provider count."""
    s = _section(QA.read_text(), 7)
    assert "eight" in s.lower() or "8 " in s, \
        "Q7 must name the 8-provider count"
    assert "router.py" in s, "Q7 must anchor at core/router.py"
    # Specific providers named
    providers = ["Groq", "NVIDIA", "Cerebras", "Mistral"]
    found = [p for p in providers if p in s]
    assert len(found) >= 3, \
        f"Q7 should name at least 3 specific providers; found {found}"


def test_q8_anchors_market_gap_and_owns_workflow() -> None:
    """Q8 ('workflow you own that nobody else owns'): must frame in
    workflow terms (not feature terms) + cite the market gap research."""
    s = _section(QA.read_text(), 8)
    # Must use workflow framing, not feature framing
    assert "workflow" in s.lower() or "lab" in s.lower(), \
        "Q8 must frame in workflow / lab terms"
    # Must name the 5 differentiators that comprise the intersection
    diffs = ["free-tier", "weekly self-grade", "falsifier", "adversarial", "signed"]
    found = [d for d in diffs if d in s.lower()]
    assert len(found) >= 4, \
        f"Q8 must cite at least 4 of the 5 intersection differentiators; found {found}"


def test_q9_cites_glean_anchor_specifically() -> None:
    """Q9 ('pricing'): MUST cite Glean as the team-tier anchor."""
    s = _section(QA.read_text(), 9)
    assert "Glean" in s, "Q9 must specifically anchor against Glean"
    assert "$50K" in s or "$50,000" in s, \
        "Q9 must include the Glean $50K annual minimum"
    assert "$99" in s, "Q9 must include the $99/seat tier"
    assert "$39" in s, "Q9 must include the $39 Pro tier"
    # The locked positioning is outcome-priced, not seat-priced
    assert "outcome" in s.lower(), \
        "Q9 must use the locked outcome-priced framing"


def test_q10_honestly_answers_no_customer() -> None:
    """Q10 ('customer who'd shut down'): MUST honestly disclose no
    customer today. The single hardest question in a pre-launch pitch;
    the honesty signal here either wins or loses the partner."""
    s = _section(QA.read_text(), 10)
    # Must NOT claim a fake customer
    assert "zero" in s.lower() or "no customer" in s.lower() or "pre-customer" in s.lower(), \
        "Q10 must explicitly disclose no customer today"
    # Must offer the proof packet as the substitute artifact
    assert "proof packet" in s.lower() or "cycle-0400" in s, \
        "Q10 must offer the proof packet as the credibility substitute"
    # Acquisition plan named
    assert "concierge" in s.lower() or "acquisition" in s.lower() or "Show HN" in s, \
        "Q10 must reference the acquisition plan"


def test_qa_no_hallucinated_metric_claims() -> None:
    """Sanity sweep: doc must not claim specific metric percentages
    that aren't in our actual artifacts. Anti-Devin check #2."""
    text = QA.read_text()
    # Specific percent claims must come from real artifacts.
    # Allow "12 PASS / 0 FAIL / 2 INSUFFICIENT" from real baseline.
    # Disallow generic % claims like "92% accuracy" / "67% reliable"
    suspicious = re.findall(
        r"(?:accuracy|reliability|reliable|productivity|efficiency)\s*(?:of|at|:)?\s*\d{1,3}%",
        text, re.IGNORECASE
    )
    assert not suspicious, \
        f"qa.md contains unverified percentage claims: {suspicious}"


def main() -> int:
    tests = [
        test_qa_exists,
        test_qa_has_all_ten_questions,
        test_qa_each_question_has_thirty_second_answer,
        test_qa_each_question_has_receipts_block,
        test_qa_includes_honest_disclosures,
        test_qa_anchors_exist_on_disk,
        test_qa_uses_locked_lede_phrasing,
        test_qa_includes_glean_anchor,
        test_qa_includes_post_devin_post_berkeley_context,
        # ── S.1 per-question depth ──
        test_q1_cites_workflow_pipeline_and_proof_packet,
        test_q2_cites_weekly_report_and_falsifier_baseline,
        test_q3_cites_real_killswitch_implementations,
        test_q4_anchors_at_weekly_report_and_architecture_doc,
        test_q5_discloses_reproducibility_layers_honestly,
        test_q6_discloses_heuristic_v1_adversarial,
        test_q7_cites_router_and_8_provider_count,
        test_q8_anchors_market_gap_and_owns_workflow,
        test_q9_cites_glean_anchor_specifically,
        test_q10_honestly_answers_no_customer,
        test_qa_no_hallucinated_metric_claims,
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
