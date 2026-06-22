"""B9 arm A6 — graph / PageRank retrieval baseline (the Aider repo-map approach).

The honest "strong alternative" to hybrid embedding retrieval: rank code by its
structure, not its embedding. We parse the corpus with `ast` into a file-level
reference graph (file A -> file B when A uses a symbol defined in B, or imports
B), then run QUERY-PERSONALIZED PageRank — the random walk is seeded on the
files that define identifiers the question mentions, exactly like Aider seeds its
repo map with chat-mentioned symbols. Chunks are then ranked by their file's
personalized-PageRank score, boosted by lexical overlap with the query.

No embeddings, no LLM. Pure structure + lexical. This is the baseline a senior
engineer asks for: "does graph centrality beat your vector index for code?"

Returns the same (chunk_id, content) shape as retrieval.hybrid_retrieve so it
drops straight into the B9 sweep as arm A6.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from pathlib import Path

import networkx as nx

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _split_identifiers(text: str) -> set[str]:
    """Query tokens: raw identifiers plus snake/Camel sub-tokens, lowercased,
    so 'SUPPORTED_DECODERS' matches 'supported' and 'decoders' too."""
    out: set[str] = set()
    for tok in _IDENT.findall(text):
        out.add(tok.lower())
        for part in tok.split("_"):
            if len(part) >= 3:
                out.add(part.lower())
        for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", tok):
            if len(part) >= 3:
                out.add(part.lower())
    return out


def _defs_refs(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    """(defined names, referenced names, imported module roots) for one file."""
    defs, refs, imports = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defs.add(t.id)
        elif isinstance(node, ast.Name):
            refs.add(node.id)
        elif isinstance(node, ast.Attribute):
            refs.add(node.attr)
        elif isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
            for a in node.names:
                refs.add(a.name)
    return defs, refs, imports


def build_graph(corpus_dir: str | Path):
    """Parse every .py file into a file-level reference graph + a symbol->files
    index. Edge F->G (weight = #references) means F depends on a symbol G defines.
    Returns (DiGraph over relative file paths, dict symbol-lower -> set[file])."""
    corpus_dir = Path(corpus_dir)
    files: dict[str, tuple[set, set, set]] = {}
    for f in sorted(corpus_dir.rglob("*.py")):
        rel = str(f.relative_to(corpus_dir))
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        files[rel] = _defs_refs(tree)

    sym2file: dict[str, set[str]] = {}
    pkgroot2file: dict[str, str] = {}
    for rel, (defs, _r, _i) in files.items():
        for d in defs:
            sym2file.setdefault(d.lower(), set()).add(rel)
        # module path root, e.g. httpx/_decoders.py -> "_decoders" and "httpx"
        parts = rel[:-3].split("/")
        for p in parts:
            pkgroot2file.setdefault(p, rel)

    g = nx.DiGraph()
    g.add_nodes_from(files.keys())
    for rel, (_d, refs, imports) in files.items():
        for r in refs:
            for tgt in sym2file.get(r.lower(), ()):  # who defines this symbol
                if tgt != rel:
                    g.add_edge(rel, tgt, weight=g.get_edge_data(rel, tgt, {}).get("weight", 0) + 1)
        for imp in imports:
            tgt = pkgroot2file.get(imp)
            if tgt and tgt != rel:
                g.add_edge(rel, tgt, weight=g.get_edge_data(rel, tgt, {}).get("weight", 0) + 1)
    return g, sym2file


def _load_chunks(db_path: str | Path) -> list[tuple[str, int, str]]:
    """All (path, chunk_idx, content) from the lab's chunks table."""
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT path, chunk_idx, content FROM chunks").fetchall()
    finally:
        con.close()
    return [(r[0], int(r[1]), r[2]) for r in rows]


def _chunk_to_corpus_file(chunk_path: str) -> str:
    """'findings/corpus/httpx/_decoders.py.md' -> 'httpx/_decoders.py'."""
    p = chunk_path
    for pre in ("findings/corpus/", "corpus/"):
        i = p.find(pre)
        if i != -1:
            p = p[i + len(pre):]
            break
    if p.endswith(".md"):
        p = p[:-3]
    return p


def graph_retrieve(question: str, *, db_path: str | Path, graph, sym2file,
                   corpus_dir: str | Path, top_n: int = 10,
                   alpha: float = 0.85) -> list[tuple[str, str]]:
    """Query-personalized PageRank over the file graph; rank chunks by their
    file's score boosted by lexical query overlap. Returns [(chunk_id, content)]."""
    qtokens = _split_identifiers(question)
    # seed files: those that DEFINE a symbol the query mentions
    seeds: dict[str, float] = {}
    for tok in qtokens:
        for f in sym2file.get(tok, ()):
            seeds[f] = seeds.get(f, 0.0) + 1.0
    nodes = list(graph.nodes())
    if not nodes:
        return []
    personalization = None
    if seeds:
        s = sum(seeds.values())
        personalization = {n: (seeds.get(n, 0.0) / s if s else 0.0) for n in nodes}
        # smooth so unseeded nodes are still reachable
        eps = 1e-6
        personalization = {n: personalization[n] + eps for n in nodes}
    try:
        pr = nx.pagerank(graph, alpha=alpha, personalization=personalization, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(graph, alpha=alpha, weight="weight", max_iter=200)

    chunks = _load_chunks(db_path)
    scored = []
    for path, idx, content in chunks:
        cf = _chunk_to_corpus_file(path)
        base = pr.get(cf, 0.0)
        ctoks = _split_identifiers(content)
        lex = len(qtokens & ctoks)
        # structure-first, but a chunk must have some lexical tie to rank high
        score = base * (1.0 + lex) + 1e-4 * lex
        scored.append((score, f"{cf}#chunk{idx}", content))
    scored.sort(key=lambda x: -x[0])
    return [(cid, content) for _s, cid, content in scored[:top_n]]


if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b9_corpus"
    db = sys.argv[2] if len(sys.argv) > 2 else "/tmp/b9_graph_lab/memory.db"
    g, s2f = build_graph(corpus)
    print(f"graph: {g.number_of_nodes()} files, {g.number_of_edges()} edges, "
          f"{len(s2f)} distinct symbols")
    top = sorted(nx.pagerank(g, weight="weight").items(), key=lambda x: -x[1])[:8]
    print("most central files (plain PageRank):")
    for f, r in top:
        print(f"  {r:.4f}  {f}")
    hits = graph_retrieve("which content encodings does SUPPORTED_DECODERS register",
                          db_path=db, graph=g, sym2file=s2f, corpus_dir=corpus, top_n=5)
    print("\nquery-personalized retrieve for SUPPORTED_DECODERS:")
    for cid, c in hits:
        print(f"  {cid}: {c[:70].strip()!r}")
