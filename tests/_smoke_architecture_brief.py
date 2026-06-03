"""Smoke test for findings/architecture/ — the in-depth technical brief.

Structural checks only; doesn't validate the prose. Catches the bug
class where someone moves / renames a doc and downstream cross-references
silently rot.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
ARCH = LAB_ROOT / "findings" / "architecture"

EXPECTED_DOCS = [
    "00_README.md",
    "01_overview.md",
    "02_pipeline.md",
    "03_memory.md",
    "04_routing.md",
    "05_signing.md",
    "06_discipline.md",
    "07_observability.md",
    "08_ui.md",
    "09_autonomy.md",
    "10_security.md",
    "11_operations.md",
    "12_procedures.md",
    "13_interfaces.md",
    "14_glossary.md",
]


def test_all_docs_present() -> None:
    for name in EXPECTED_DOCS:
        path = ARCH / name
        assert path.exists(), f"architecture/{name} missing"


def test_readme_lists_all_docs() -> None:
    text = (ARCH / "00_README.md").read_text()
    for name in EXPECTED_DOCS:
        if name == "00_README.md":
            continue
        # The README's document map column references files by name; that
        # match suffices for our consistency check.
        assert name.replace(".md", "") in text or name in text, \
            f"README doesn't reference {name}"


def test_cross_references_resolve() -> None:
    """Every backtick-quoted reference to NN_*.md in a doc must
    resolve to an existing file."""
    pattern = re.compile(r"`(\d{2}_[a-z_]+\.md)`")
    for doc in EXPECTED_DOCS:
        text = (ARCH / doc).read_text()
        for match in pattern.finditer(text):
            referenced = match.group(1)
            assert (ARCH / referenced).exists(), \
                f"{doc} references {referenced} which doesn't exist"


def test_each_doc_has_a_cross_references_section() -> None:
    """All non-README docs should end with a Cross-references section
    so readers can navigate. Glossary is the only exception
    (it IS the navigation aid)."""
    skip = {"00_README.md", "14_glossary.md"}
    for doc in EXPECTED_DOCS:
        if doc in skip:
            continue
        text = (ARCH / doc).read_text()
        assert "## Cross-references" in text or "Cross-references" in text[-2000:], \
            f"{doc} missing Cross-references section"


def test_each_doc_has_substantial_depth() -> None:
    """Every doc should be >= 500 words. Anything shorter is a stub."""
    for doc in EXPECTED_DOCS:
        text = (ARCH / doc).read_text()
        word_count = len(text.split())
        assert word_count >= 500, \
            f"{doc} is too short ({word_count} words; expected ≥500)"


def test_honest_disclosure_present() -> None:
    """The brief MUST include the honest disclosures the locked memories
    require. Per DD.1+DD.2+DD.3 (2026-05-17) the disclosure language
    shifted away from "I.4 placeholder" framing toward:
      - heuristic-v1 OR llm-driven-v2 adversarial (DD.1 toggle)
      - local-dev Sigstore — production is engineering, not a flag (DD.2)
      - cryptographic-only reproducibility — exact-process LLM replay is
        structurally impossible for hosted-LLM workflows (DD.3)
      - no acquired customer yet
    """
    full_text = "\n".join((ARCH / d).read_text() for d in EXPECTED_DOCS)
    for disclosure in (
        "heuristic-v1",
        "llm-driven-v2",
        "local-dev mode",
        "DD.3",
        "DD.2",
    ):
        assert disclosure in full_text, \
            f"brief missing honest disclosure: {disclosure}"


def test_brief_names_all_seven_ui_surfaces() -> None:
    """The UI doc must reference all 7 surfaces by name."""
    ui = (ARCH / "08_ui.md").read_text()
    for surface in ("FirstLight", "Meeting", "Tide",
                    "Manuscript", "Loom", "Atlas", "Diagnostics"):
        assert surface in ui, f"08_ui.md doesn't mention {surface}"


def test_brief_names_all_ten_memory_layers() -> None:
    """Memory doc structural check."""
    memory = (ARCH / "03_memory.md").read_text()
    for term in ("Events log", "Core memory tier", "Recall tier",
                 "Archival tier", "Knowledge graph", "Cycle artifacts",
                 "Brief assembler", "Semantic cache",
                 "Compaction shapers", "Provenance ledger"):
        assert term in memory, f"03_memory.md missing layer: {term}"


def test_brief_names_eight_providers() -> None:
    """Routing doc must enumerate the 8 free-tier providers."""
    routing = (ARCH / "04_routing.md").read_text()
    for provider in ("Groq", "NVIDIA", "Cerebras", "Mistral",
                     "Google AI Studio", "OpenRouter",
                     "Cloudflare", "Ollama"):
        assert provider in routing, f"04_routing.md missing {provider}"


def test_brief_documents_proof_packet_schema() -> None:
    """Signing doc must enumerate the proof packet's key files."""
    signing = (ARCH / "05_signing.md").read_text()
    for key_file in ("HASHES.txt", "cycle.json", "failures.md",
                     "slsa.intoto.jsonl", "slsa.sigstore", "adversarial.json"):
        assert key_file in signing, \
            f"05_signing.md doesn't reference {key_file}"


def test_glossary_covers_pipeline_terms() -> None:
    """Glossary structural check — must cover threshing/clearness/seasoning."""
    gloss = (ARCH / "14_glossary.md").read_text()
    for term in ("threshing", "clearness", "seasoning",
                 "cycle", "dispatch", "ResultPacket",
                 "falsifier", "Dominus", "Manuscript",
                 "FirstLight", "Atlas", "Loom"):
        assert f"**{term}**" in gloss, \
            f"glossary missing entry for {term}"


def main() -> int:
    tests = [
        test_all_docs_present,
        test_readme_lists_all_docs,
        test_cross_references_resolve,
        test_each_doc_has_a_cross_references_section,
        test_each_doc_has_substantial_depth,
        test_honest_disclosure_present,
        test_brief_names_all_seven_ui_surfaces,
        test_brief_names_all_ten_memory_layers,
        test_brief_names_eight_providers,
        test_brief_documents_proof_packet_schema,
        test_glossary_covers_pipeline_terms,
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
