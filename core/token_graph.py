"""L0 canonical-token graph + Personalized PageRank for bert retrieval.

Phase B2 of the v3 plan. Implements AGI's L0 layer for bert:
extract regex-captured canonical tokens from the corpus, build a
co-occurrence graph (nodes = tokens, edges = same-chunk co-occurrence),
seed PPR from query tokens, return ranked related-token set as a
retrieval signal.

bert's canonical vocabulary (per the audit in plan v3 §1):

  C\\d+           cycle IDs                  e.g. C107, C108
  D-\\d+          decision IDs                e.g. D-005 (Phase A adopts)
  P-[A-Z0-9_-]+   carry-forward / process IDs e.g. P-A1-CYCLE-BUDGET
  arxiv:\\d{4}    arXiv paper IDs             e.g. arxiv:2312.00752
  roles           director, researcher, ...
  verdicts        APPROVE, REJECT, BUILD_PASS, ...
  cycle_shapes    research-deeper, mission-complete, ...
  finding paths   findings/bert_run_C{N}_(researcher|strategist).md

Graph DB: SQLite at `<lab>/state/token_graph.db`. Lightweight; rebuilds
incrementally on touched chunks.

Search returns ranked tokens → chunks containing those tokens →
SearchResult shape compatible with core.retrieval's signal sources.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

LOG = logging.getLogger("bert.token_graph")

LAB_ROOT = Path(__file__).resolve().parent.parent


# ── Canonical token regex registry ─────────────────────────────────


_PATTERNS = (
    ("cycle",        re.compile(r"\bC\d{1,4}\b")),
    ("decision",     re.compile(r"\bD-\d+\b")),
    ("carry_fwd",    re.compile(r"\bP-[A-Z0-9][A-Z0-9_-]+\b")),
    ("arxiv",        re.compile(r"\barxiv:\d{4}\.\d{4,5}\b", re.IGNORECASE)),
    ("arxiv_url",    re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")),
    ("github_repo",  re.compile(r"github\.com/([\w-]+/[\w.-]+)")),
    ("role",         re.compile(r"\b(director|researcher|strategist|"
                                 r"evaluator|implementer|consolidator)\b",
                                 re.IGNORECASE)),
    ("verdict",      re.compile(r"\b(APPROVE|REJECT|BUILD_PASS|BUILD_FAIL|"
                                 r"APPROVE_WITH_CAVEATS|CHANGES_REQUESTED|"
                                 r"SCOPE_STOP|BUILD_PARTIAL|OTHER)\b")),
    ("cycle_shape",  re.compile(r"\b(research-deeper|mission-complete|"
                                 r"strategy-refine|verification-tighten|"
                                 r"synthesis|idle)\b")),
    ("finding_path", re.compile(r"findings/bert_run_C\d+_"
                                 r"(?:researcher|strategist|director)\.md")),
)


@dataclass
class CanonicalToken:
    """A single canonical token mention in the corpus."""
    kind: str         # token family
    value: str        # canonical form (lowercased / normalized)
    chunk_id: int     # which chunk it appeared in
    raw: str          # the raw matched text


def extract_tokens(text: str) -> list[tuple[str, str]]:
    """Extract (kind, canonical_value) pairs from a text blob.
    Canonical form: lowercase; strip arxiv prefix; etc."""
    if not text:
        return []
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(0)
            value = _canonicalize(kind, raw, m)
            key = (kind, value)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _canonicalize(kind: str, raw: str, match) -> str:
    """Normalize a raw match to its canonical token form."""
    if kind == "arxiv_url":
        return f"arxiv:{match.group(1)}"
    if kind == "github_repo":
        return f"github:{match.group(1).lower()}"
    if kind in ("role", "verdict", "cycle_shape"):
        return raw.lower()
    return raw


# ── SQLite-backed graph store ─────────────────────────────────────


def _db_path(lab_path: Path) -> Path:
    state_dir = lab_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "token_graph.db"


def _init_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY,
            kind TEXT NOT NULL,
            value TEXT NOT NULL,
            mention_count INTEGER DEFAULT 0,
            UNIQUE(kind, value)
        );
        CREATE TABLE IF NOT EXISTS token_chunks (
            token_id INTEGER NOT NULL REFERENCES tokens(id),
            chunk_id INTEGER NOT NULL,
            PRIMARY KEY (token_id, chunk_id)
        );
        CREATE TABLE IF NOT EXISTS cooccur (
            token_a INTEGER NOT NULL REFERENCES tokens(id),
            token_b INTEGER NOT NULL REFERENCES tokens(id),
            weight INTEGER DEFAULT 1,
            PRIMARY KEY (token_a, token_b)
        );
        CREATE INDEX IF NOT EXISTS ix_token_chunks_chunk ON token_chunks(chunk_id);
        CREATE INDEX IF NOT EXISTS ix_tokens_kind ON tokens(kind);
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    con.commit()


def _ensure_token(con: sqlite3.Connection, kind: str, value: str) -> int:
    """Upsert a token, return its id."""
    con.execute(
        "INSERT INTO tokens(kind, value, mention_count) VALUES(?, ?, 1) "
        "ON CONFLICT(kind, value) DO UPDATE SET "
        "mention_count = mention_count + 1",
        (kind, value),
    )
    row = con.execute(
        "SELECT id FROM tokens WHERE kind=? AND value=?", (kind, value)
    ).fetchone()
    return int(row[0])


# ── Rebuild ──────────────────────────────────────────────────────


def rebuild(lab_path: Path) -> dict:
    """Walk all chunks in the lab's chunks.db, extract canonical tokens,
    populate token_graph.db. Idempotent — wipes existing token state."""
    chunks_db = lab_path / "memory.db"
    if not chunks_db.exists():
        chunks_db = LAB_ROOT / "memory.db"
    if not chunks_db.exists():
        return {"chunks_scanned": 0, "tokens": 0, "edges": 0, "ms": 0}

    t0 = time.monotonic()
    graph_db = _db_path(lab_path)
    chunks_scanned = 0
    edges_added = 0
    tokens_added = 0

    # Read chunks
    with sqlite3.connect(chunks_db) as chunk_con:
        rows = chunk_con.execute(
            "SELECT id, content FROM chunks ORDER BY id"
        ).fetchall()

    with sqlite3.connect(graph_db) as gcon:
        _init_schema(gcon)
        # Wipe existing token state (cheap; will populate fresh below)
        gcon.executescript(
            "DELETE FROM token_chunks; DELETE FROM cooccur; "
            "UPDATE tokens SET mention_count = 0;"
        )
        gcon.commit()

        for chunk_id, content in rows:
            chunks_scanned += 1
            tokens_in_chunk = extract_tokens(content or "")
            token_ids: list[int] = []
            for kind, value in tokens_in_chunk:
                tid = _ensure_token(gcon, kind, value)
                token_ids.append(tid)
                gcon.execute(
                    "INSERT OR IGNORE INTO token_chunks(token_id, chunk_id) "
                    "VALUES(?, ?)",
                    (tid, int(chunk_id)),
                )
            # Co-occurrence edges (undirected — store both directions
            # for simpler PPR query, weight aggregated)
            unique_ids = list(dict.fromkeys(token_ids))
            for i, a in enumerate(unique_ids):
                for b in unique_ids[i + 1:]:
                    gcon.execute("""
                        INSERT INTO cooccur(token_a, token_b, weight)
                        VALUES(?, ?, 1)
                        ON CONFLICT(token_a, token_b) DO UPDATE SET
                            weight = weight + 1
                        """, (a, b))
                    gcon.execute("""
                        INSERT INTO cooccur(token_a, token_b, weight)
                        VALUES(?, ?, 1)
                        ON CONFLICT(token_a, token_b) DO UPDATE SET
                            weight = weight + 1
                        """, (b, a))
                    edges_added += 2

        tokens_added = gcon.execute(
            "SELECT COUNT(*) FROM tokens"
        ).fetchone()[0]

        gcon.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('built_at', ?)",
            (str(time.time()),),
        )
        gcon.commit()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return {
        "chunks_scanned": chunks_scanned,
        "tokens": tokens_added,
        "edges": edges_added,
        "ms": elapsed_ms,
    }


# ── PPR + search ────────────────────────────────────────────────


def _load_graph(con: sqlite3.Connection) -> nx.DiGraph:
    g = nx.DiGraph()
    for tid, kind, value in con.execute(
        "SELECT id, kind, value FROM tokens"
    ):
        g.add_node(int(tid), kind=kind, value=value)
    for a, b, w in con.execute(
        "SELECT token_a, token_b, weight FROM cooccur"
    ):
        g.add_edge(int(a), int(b), weight=int(w))
    return g


def seed_tokens_from_query(query: str) -> list[tuple[str, str]]:
    """Find canonical tokens IN the query that match our vocabulary."""
    return extract_tokens(query)


@dataclass
class PPRHit:
    chunk_id: int
    score: float
    metadata: dict


def search(
    query: str,
    *,
    lab_path: Path,
    k: int = 20,
    alpha: float = 0.15,
) -> list[PPRHit]:
    """Personalized PageRank from query-extracted seeds → ranked chunks.

    If query has no canonical tokens, returns []. Caller's hybrid
    retriever still has vector + BM25 + cache signals to fall back on.

    Algorithm:
      1. Extract canonical tokens from query → seed set
      2. If no seeds: return empty (no PPR signal for this query)
      3. Run PPR on the cooccur graph; restart prob = alpha
      4. Top-k tokens by PPR score → join via token_chunks → chunks
      5. Aggregate chunk score by summing token PPR scores; rank top-k
    """
    if not query or not query.strip():
        return []
    seeds = seed_tokens_from_query(query)
    if not seeds:
        return []

    db = _db_path(lab_path)
    if not db.exists():
        return []

    with sqlite3.connect(db) as con:
        # Map seeds → token IDs
        seed_ids: list[int] = []
        for kind, value in seeds:
            row = con.execute(
                "SELECT id FROM tokens WHERE kind=? AND value=?",
                (kind, value),
            ).fetchone()
            if row:
                seed_ids.append(int(row[0]))
        if not seed_ids:
            return []

        g = _load_graph(con)
        if g.number_of_nodes() == 0:
            return []

        personalization = dict.fromkeys(g.nodes(), 0.0)
        for sid in seed_ids:
            if sid in personalization:
                personalization[sid] = 1.0 / len(seed_ids)

        try:
            ppr = nx.pagerank(
                g, personalization=personalization,
                alpha=1.0 - alpha,    # damping; networkx convention
                max_iter=100,
                tol=1e-6,
            )
        except (nx.NetworkXError, ZeroDivisionError) as e:
            LOG.warning("PPR failed: %s", e)
            return []

        # Top tokens by score (excluding seeds themselves)
        ranked_tokens = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
        top_token_ids = [tid for tid, score in ranked_tokens[:50] if score > 0]
        if not top_token_ids:
            return []

        # Aggregate chunk scores
        chunk_scores: Counter = Counter()
        for tid in top_token_ids:
            for row in con.execute(
                "SELECT chunk_id FROM token_chunks WHERE token_id=?", (tid,)
            ):
                chunk_scores[int(row[0])] += ppr.get(tid, 0.0)

    ranked = chunk_scores.most_common(k)
    return [
        PPRHit(chunk_id=cid, score=float(score),
               metadata={"source": "token_graph_ppr"})
        for cid, score in ranked
    ]


# ── CLI ────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """python -m core.token_graph rebuild <lab>
    python -m core.token_graph search <lab> "<query>" [k=10]
    """
    import sys
    if len(argv) < 3:
        print("usage: token_graph rebuild|search ...", file=sys.stderr)
        return 2
    cmd = argv[1]
    lab = Path(argv[2]).expanduser()
    if cmd == "rebuild":
        stats = rebuild(lab)
        print(f"rebuilt: chunks={stats['chunks_scanned']} "
              f"tokens={stats['tokens']} edges={stats['edges']} "
              f"ms={stats['ms']}")
        return 0
    if cmd == "search":
        if len(argv) < 4:
            print('usage: token_graph search <lab> "<query>" [k]',
                  file=sys.stderr)
            return 2
        q = argv[3]
        k = int(argv[4]) if len(argv) >= 5 else 10
        hits = search(q, lab_path=lab, k=k)
        print(f"{len(hits)} hits")
        for h in hits:
            print(f"  [{h.score:.4f}] chunk={h.chunk_id}")
        return 0
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv))
