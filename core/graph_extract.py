"""Heuristic finding → graph extraction.

When the Write tool emits a `finding` event (core.tools._maybe_emit_
finding_event), we ALSO walk the finding's markdown for structured
graph signals and upsert them into the active lab's graph.db. This
populates Atlas's strata ring per-cycle without waiting on a
dedicated LLM extraction subagent — the LLM-driven version is a
deeper feature we can layer on top of this contract later.

Extraction shape (v1 — pragmatic, evolve as we see real findings):

  Each finding becomes a Candidate node.
    id    = its event id ("find_<sha1[:12]>")
    label = first non-heading line (≤120 chars)
    props = {role, cycle, source_path}

  The cycle becomes a Mission node (idempotent per cycle).
    id    = "cycle_C{N}"
    label = "Cycle {N}"

  Markdown links [text](url) → Source nodes + REFERENCES edges
  from the Candidate to each Source.

  Bold + heading-bracketed proper-noun terms (e.g. "**Mamba**",
  "## Candidate: RWKV") → additional Candidate nodes + APPLIES_TO
  edges connecting the finding's Candidate to each named term.

  Every finding emits an EVIDENCED_BY edge from its Candidate to
  the cycle's Mission node — the structural backbone of the
  strata view.

Edges carry valid_from (now) so Graphiti-style time-windowed
queries work. Future LLM extraction can layer CONFLICTS_WITH /
SUPERSEDES / KILLED_BY for higher-quality rendering.

All operations are best-effort: failure to extract NEVER breaks
the write that triggered it.
"""

from __future__ import annotations

import logging
import re
import time

LOG = logging.getLogger("bert.graph_extract")


# Capture markdown links: [text](url) — extract the URL as a source id.
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Strong / heading-style proper nouns we treat as Candidates:
#   "**Transformer**", "## Candidate: Mamba", "## Mamba"
_BOLD_RE = re.compile(r"\*\*([A-Z][A-Za-z0-9_-]{1,40}(?:\s+[A-Z][A-Za-z0-9_-]{1,40}){0,3})\*\*")
_HEADING_CAND_RE = re.compile(
    r"^#{2,3}\s+(?:Candidate|Approach|Alternative|Technique|Architecture)\s*:?\s*"
    r"([A-Z][A-Za-z0-9_-]{1,40}(?:\s+[A-Z][A-Za-z0-9_-]{1,40}){0,3})",
    re.MULTILINE,
)


def extract_and_persist(
    *,
    finding_id: str,
    content: str,
    cycle: int | None,
    role: str | None,
    source_path: str,
) -> dict[str, int]:
    """Run heuristic extraction + upsert nodes/edges. Returns a small
    counter dict for debug. Swallows every exception."""
    counts = {"nodes": 0, "edges": 0}
    try:
        from core import graph_store
    except Exception as e:
        LOG.debug("graph_extract: graph_store import failed: %s", e)
        return counts

    try:
        first_line = _first_meaningful_line(content)
        # 1. Candidate node for THIS finding
        graph_store.add_node(
            node_id=finding_id, node_type="Candidate",
            label=first_line[:120],
            props={"role": role or "", "cycle": cycle,
                   "source_path": source_path},
        )
        counts["nodes"] += 1

        # 2. Mission node for the cycle (idempotent — upsert)
        if cycle is not None:
            cycle_node_id = f"cycle_C{cycle}"
            graph_store.add_node(
                node_id=cycle_node_id, node_type="Mission",
                label=f"Cycle {cycle}",
                props={"cycle": cycle},
            )
            counts["nodes"] += 1
            # 3. EVIDENCED_BY: Candidate → Mission (the strata backbone)
            graph_store.add_edge(
                src=finding_id, dst=cycle_node_id,
                edge_type="EVIDENCED_BY",
                props={"role": role or ""},
                valid_from=time.time(),
            )
            counts["edges"] += 1

        # 4. Source nodes from markdown links
        for m in _LINK_RE.finditer(content[:8000]):
            link_text, url = m.group(1).strip(), m.group(2).strip()
            if not url or url.startswith("#"):
                continue
            src_id = _source_id(url)
            try:
                graph_store.add_node(
                    node_id=src_id, node_type="Source",
                    label=(link_text or url)[:120],
                    props={"url": url},
                )
                counts["nodes"] += 1
                graph_store.add_edge(
                    src=finding_id, dst=src_id,
                    edge_type="REFERENCES",
                    props={"link_text": link_text[:120]},
                    valid_from=time.time(),
                )
                counts["edges"] += 1
            except Exception:
                continue

        # 5. Named Candidates from bold + heading patterns
        named_terms: set[str] = set()
        for m in _BOLD_RE.finditer(content[:8000]):
            named_terms.add(m.group(1).strip())
        for m in _HEADING_CAND_RE.finditer(content[:8000]):
            named_terms.add(m.group(1).strip())
        for term in named_terms:
            if len(term) < 2:
                continue
            term_id = "cand_" + re.sub(r"[^a-z0-9_]+", "-", term.lower())
            try:
                graph_store.add_node(
                    node_id=term_id, node_type="Candidate",
                    label=term[:80],
                    props={"surface": "named_in_finding"},
                )
                counts["nodes"] += 1
                # APPLIES_TO: finding's Candidate is informed by the
                # named technique — directional edge from finding to
                # the named candidate.
                graph_store.add_edge(
                    src=finding_id, dst=term_id,
                    edge_type="APPLIES_TO",
                    props={},
                    valid_from=time.time(),
                )
                counts["edges"] += 1
            except Exception:
                continue

        LOG.info("graph_extract: finding=%s → nodes=%d edges=%d",
                 finding_id, counts["nodes"], counts["edges"])
    except Exception as e:
        LOG.warning("graph_extract failed (advisory): %s", e)
    return counts


def _first_meaningful_line(content: str) -> str:
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s
    # Fallback: first heading without the leading #
    for line in content.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("# ").strip()
    return "(empty finding)"


def _source_id(url: str) -> str:
    """Stable id for a Source from a URL. Prefer arXiv/file/decision
    patterns the lab uses; fall back to a sha1 prefix."""
    import hashlib
    # arXiv: stable id from the paper number
    m = re.search(r"arxiv\.org/abs/([0-9.]+)", url)
    if m:
        return f"src_arxiv_{m.group(1).replace('.', '_')}"
    return "src_" + hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
