"""Smoke test for R.1: regulated-industries landing variant.

Verifies that landing/regulated/index.html exists, follows the locked
positioning fixes (palette + Lora typography + appropriate honesty
disclosures), and cross-links bidirectionally with the main landing.

Honest-disclosure depth: regulated buyers will read the anti-claims
section twice, so the test asserts the specific disclosures a
compliance buyer will look for (not certified, not legal advice, not
a substitute for counsel).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
MAIN_LANDING = LAB_ROOT / "landing" / "index.html"
REG_LANDING = LAB_ROOT / "landing" / "regulated" / "index.html"


def test_regulated_landing_exists() -> None:
    assert REG_LANDING.exists(), "landing/regulated/index.html missing"


def test_regulated_has_compliance_hero() -> None:
    text = REG_LANDING.read_text()
    assert "When the auditor asks for receipts" in text, \
        "hero must lead with audit-trail framing"
    assert "regulated solos" in text.lower(), \
        "page must self-identify as regulated-solos variant"


def test_regulated_names_three_verticals() -> None:
    """Legal / healthcare / finance — the three verticals from the
    locked positioning fixes memory."""
    text = REG_LANDING.read_text().lower()
    assert "legal solos" in text, "legal solo vertical missing"
    assert "healthcare solos" in text, "healthcare solo vertical missing"
    assert "finance compliance" in text, "finance compliance vertical missing"


def test_regulated_names_compliance_stack() -> None:
    """The four compliance primitives: OWASP, AGNTCY DID, Sigstore + Rekor,
    separately signed failures.md."""
    text = REG_LANDING.read_text()
    assert "OWASP" in text, "OWASP Agentic Top-10 must be cited"
    assert "AGNTCY" in text, "AGNTCY DID must be cited"
    assert "Sigstore" in text and "Rekor" in text, \
        "Sigstore + Rekor pair must be cited"
    assert "failures.md" in text, "separately signed failures.md must be cited"


def test_regulated_pricing_in_10k_to_25k_range() -> None:
    """Per the locked memory, regulated setup is anchored at $10K-$25K
    against the consulting market."""
    text = REG_LANDING.read_text()
    assert "$10K" in text and "$25K" in text, \
        f"pricing must anchor in $10K-$25K consulting range"


def test_regulated_anti_claims_protect_against_regulatory_misuse() -> None:
    """Critical for regulated buyers — anti-claims must explicitly
    disclaim legal/medical/financial advice + lack of certification."""
    text = REG_LANDING.read_text()
    # Not professional advice
    assert "Not legal" in text or "not legal" in text, \
        "must disclaim legal advice"
    assert "medical" in text.lower(), "must disclaim medical advice"
    assert "financial" in text.lower(), "must disclaim financial advice"
    # Not a substitute for compliance counsel
    assert "compliance counsel" in text.lower() or \
           "compliance officer" in text.lower(), \
           "must reference compliance counsel as the authority"
    # Not certified
    assert "Not certified" in text or "not certified" in text, \
        "must disclaim certification status"
    # Name the specific certifications they're NOT
    assert "HIPAA" in text, "must name HIPAA in non-cert list"
    assert "SOC 2" in text, "must name SOC 2 in non-cert list"
    # Not a guarantee
    assert "Not a guarantee" in text or "not a guarantee" in text, \
        "must disclaim compliance guarantee"


def test_regulated_uses_same_palette_tokens() -> None:
    """Visual consistency: regulated variant must use the same bert
    palette as main landing + bert UI."""
    text = REG_LANDING.read_text()
    for var in ("--night:", "--bone:", "--candle:", "--rust:", "--moss:"):
        assert var in text, f"missing palette token {var}"


def test_regulated_uses_lora_typography() -> None:
    """Per feedback_visualization_as_art: every UI must match bert
    aesthetic. Lora serif + JetBrains Mono is the typography pair."""
    text = REG_LANDING.read_text()
    assert "'Lora'" in text, "Lora serif must be loaded"
    assert "'JetBrains Mono'" in text, "JetBrains Mono must be loaded"


def test_regulated_has_cosign_command() -> None:
    """The 'verifiable proof' moment — same as main landing, the
    cosign command must be visible so partner CTOs / auditors can
    verify independently."""
    text = REG_LANDING.read_text()
    # item 25: the REAL working command is `cosign verify-blob --key --signature`
    assert "cosign verify-blob" in text, \
        "must show vanilla cosign command for independent verification"
    assert "--signature HASHES.sig" in text


def test_main_landing_cross_links_to_regulated() -> None:
    """The main landing must surface a link to the regulated variant
    so regulated buyers can find it."""
    main = MAIN_LANDING.read_text()
    assert "regulated" in main.lower(), \
        "main landing must mention regulated variant"
    assert 'href="regulated/' in main or 'href="regulated"' in main or \
           "regulated/" in main, \
           "main landing must link to /regulated/"


def test_regulated_back_links_to_main() -> None:
    """Reciprocal — regulated must link back to the generic variant
    so a buyer who lands on /regulated/ first can navigate to the
    main pitch."""
    text = REG_LANDING.read_text()
    # Either an explicit "← back" link, or a footer link to /
    assert 'href="../"' in text or 'href="/"' in text or \
           "autonomous-lab variant" in text.lower(), \
           "regulated landing must link back to generic variant"


def test_regulated_html_valid_basics() -> None:
    """Sanity: tags balance."""
    text = REG_LANDING.read_text()
    assert text.count("<main>") == text.count("</main>")
    assert text.count("<section") == text.count("</section>")
    assert "<html" in text and "</html>" in text


# ── S.3 depth audit: per-vertical regulatory anchor verification ─────
# Each vertical must reference its specific 2026 regulatory context —
# legal → state bar AI rules, healthcare → OCR/HIPAA, finance → AML /
# regulatory examination. Without these, the vertical sections read as
# generic marketing copy rather than informed positioning.

def _vertical_section(text: str, label: str) -> str:
    """Slice the <li> block that contains a given vert-label."""
    # The pattern matches a <li> wrapping a vert-label that contains `label`
    pattern = (
        rf'<li>\s*<span class="vert-label">[^<]*{re.escape(label)}[^<]*</span>'
        rf'\s*<span class="vert-body">(.*?)</span>\s*</li>'
    )
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not m:
        raise AssertionError(f"could not slice vertical section for '{label}'")
    return m.group(1)


def test_legal_vertical_anchors_state_bar_rules() -> None:
    """Legal solos: must reference state bar AI-disclosure rules. The
    2026 buyer in this segment is preparing for an AI-disclosure
    obligation they didn't have last year."""
    body = _vertical_section(REG_LANDING.read_text(), "legal solos")
    assert "state-bar" in body.lower() or "state bar" in body.lower(), \
        "legal vertical must reference state bar AI rules"
    # Name at least one specific state (CA/FL/NY are the 2026 movers)
    body_lower = body.lower()
    assert "california" in body_lower or "florida" in body_lower or "new york" in body_lower, \
        "legal vertical must name at least one specific 2026-active state"
    # Specific legal use cases
    use_cases = ["discovery", "contract", "brief"]
    found = [u for u in use_cases if u in body.lower()]
    assert len(found) >= 2, \
        f"legal vertical must name at least 2 use cases; found {found}"


def test_healthcare_vertical_anchors_hipaa_or_ocr() -> None:
    """Healthcare solos: must reference HIPAA / OCR (Office for Civil
    Rights) review or equivalent compliance frame. PHI handling
    discipline must be explicit."""
    body = _vertical_section(REG_LANDING.read_text(), "healthcare solos")
    # OCR is the federal HIPAA enforcement body — direct anchor for the audience
    assert "OCR" in body or "HIPAA" in body, \
        "healthcare vertical must reference HIPAA or OCR"
    # PHI handling must be addressed explicitly
    assert "PHI" in body, "healthcare vertical must address PHI handling"
    # AGNTCY identity → EHR audit logs is the bert-specific value
    assert "EHR" in body, "healthcare vertical must reference EHR audit logs"
    # Local-first means PHI doesn't leave the laptop
    assert "local" in body.lower() or "laptop" in body.lower(), \
        "healthcare vertical must surface local-first / no-PHI-leaves discipline"


def test_finance_vertical_anchors_aml_or_regulatory_exam() -> None:
    """Finance compliance: must reference AML / regulatory examination /
    KYC — the actual workflow language a compliance officer uses."""
    body = _vertical_section(REG_LANDING.read_text(), "finance compliance")
    # At least one of the canonical finance compliance acronyms
    finance_anchors = ["AML", "KYC", "regulator", "examination"]
    found = [a for a in finance_anchors if a in body or a.lower() in body.lower()]
    assert len(found) >= 2, \
        f"finance vertical must include at least 2 finance compliance anchors; found {found}"
    # Replayability is the bert-specific value here
    assert "replay" in body.lower() or "deterministic" in body.lower(), \
        "finance vertical must surface replayability / determinism for examination"


def test_pricing_comparison_anchors_consulting_market() -> None:
    """Pricing section must directly compare to the AI compliance
    consulting market ($10K-$25K for a checklist)."""
    text = REG_LANDING.read_text()
    m = re.search(r'<section id="pricing">(.*?)</section>', text, re.DOTALL)
    assert m, "pricing section not found"
    pricing = m.group(1)
    assert "consultant" in pricing.lower() or "consulting" in pricing.lower(), \
        "pricing must contrast with consulting-market alternative"
    assert "$10K" in pricing or "$10,000" in pricing, \
        "pricing must quote $10K low anchor"
    assert "$25K" in pricing or "$25,000" in pricing, \
        "pricing must quote $25K high anchor"


def test_anti_claims_use_rust_warning_palette() -> None:
    """The anti-claims block must use the rust palette token for the
    warning border — visual signal that this is the 'read twice' section.
    Anti-Devin: regulated buyers SHOULD have to read disclaimers twice."""
    text = REG_LANDING.read_text()
    # The anti-claims block is styled with rust border (line ~165)
    m = re.search(r'\.anti\s*\{[^}]*border:\s*[^;]*var\(--rust\)', text)
    assert m, "anti-claims block must use --rust palette for warning border"


def test_regulated_landing_serves_http_200() -> None:
    """End-to-end serve check: a static HTTP server hosting landing/
    must return 200 for /regulated/ — proves the file path + relative
    links work as deployed."""
    import socket
    import subprocess
    import time
    import urllib.request

    # Find an unused port
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port),
         "--directory", str(LAB_ROOT / "landing")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Give the server a moment to bind
        for _ in range(20):
            time.sleep(0.1)
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=1
                ) as r:
                    if r.status == 200:
                        break
            except Exception:
                continue
        else:
            raise AssertionError("http.server didn't come up within 2s")

        # Main landing serves
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/", timeout=2
        ) as r:
            assert r.status == 200, f"main landing returned {r.status}"
            main_body = r.read().decode()
            # The cross-link to regulated must be reachable
            assert "regulated/" in main_body, "main landing must link to regulated/"

        # Regulated variant serves
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/regulated/", timeout=2
        ) as r:
            assert r.status == 200, f"regulated landing returned {r.status}"
            reg_body = r.read().decode()
            assert "auditor asks for receipts" in reg_body, \
                "regulated body must include the locked hero"

        # Back-link to main from regulated must resolve
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/", timeout=2
        ) as r:
            assert r.status == 200, "back-link target / must resolve"
    finally:
        proc.terminate()
        proc.wait(timeout=3)


def test_regulated_does_not_overclaim_certification() -> None:
    """Anti-Devin check for the regulated audience — the page must NOT
    use words like 'certified', 'compliant', 'guaranteed' in a way that
    asserts a status bert hasn't earned. The honest 'NOT certified'
    disclosure must be paired with no positive-direction certification
    claims elsewhere."""
    text = REG_LANDING.read_text()
    # Patterns of the form "bert/bert IS-certified / IS-compliant /
    # GUARANTEES-compliance". Negated forms ("No software guarantees ...",
    # "Not certified by HIPAA") are honest disclosures and must NOT match.
    forbidden_patterns = [
        r"\b(?:bert|bert)\s+is\s+HIPAA[-\s]compliant\b",
        r"\b(?:bert|bert)\s+is\s+SOC\s*2\s+certified\b",
        r"\b(?:bert|bert)\s+guarantees?\s+compliance\b",
        r"\b(?:bert|bert)\s+is\s+certified\s+(?:medical|legal|financial|HIPAA|SOC|ISO|FedRAMP)\b",
    ]
    for pattern in forbidden_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        assert not m, \
            f"regulated landing contains overclaim: {pattern!r} matched {m.group(0)!r}"


def main() -> int:
    tests = [
        test_regulated_landing_exists,
        test_regulated_has_compliance_hero,
        test_regulated_names_three_verticals,
        test_regulated_names_compliance_stack,
        test_regulated_pricing_in_10k_to_25k_range,
        test_regulated_anti_claims_protect_against_regulatory_misuse,
        test_regulated_uses_same_palette_tokens,
        test_regulated_uses_lora_typography,
        test_regulated_has_cosign_command,
        test_main_landing_cross_links_to_regulated,
        test_regulated_back_links_to_main,
        test_regulated_html_valid_basics,
        # ── S.3 per-vertical + HTTP serve depth ──
        test_legal_vertical_anchors_state_bar_rules,
        test_healthcare_vertical_anchors_hipaa_or_ocr,
        test_finance_vertical_anchors_aml_or_regulatory_exam,
        test_pricing_comparison_anchors_consulting_market,
        test_anti_claims_use_rust_warning_palette,
        test_regulated_landing_serves_http_200,
        test_regulated_does_not_overclaim_certification,
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
