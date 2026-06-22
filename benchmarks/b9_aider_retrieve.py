"""B9 arm A6-REAL — graph/PageRank retrieval using the ACTUAL Aider RepoMap.

The earlier `b9_graph_retrieve.py` was a from-scratch reimplementation of the
Aider repo-map idea (file-level ast graph + query-personalized PageRank). A
senior reviewer's fair objection: benchmark against the real thing, not an
imitation. This module drives Aider's own `RepoMap.get_ranked_tags()`
(tree-sitter tags + MultiDiGraph + personalized PageRank, exactly what Aider
ships) and maps its ranked output onto the same chunk store the other B9 arms
use, so the only variable swapped is the ranking source.

Two faithful projections of Aider's ranking onto chunks (same recall metric):
  - aider_tags     : walk Aider's ranked definitions in order, take the chunk
                     containing each definition site. This is what Aider's repo
                     map literally surfaces.
  - aider_filerank : derive a per-file importance score from Aider's ranked
                     order, then select chunks within files by lexical overlap.
                     This mirrors b9_graph_retrieve EXACTLY (same chunk selector)
                     so it is a clean apples-to-apples swap of the PageRank source.

Run with the aider venv (has aider + networkx; chunk store is plain sqlite):
    /tmp/aider_venv/bin/python benchmarks/b9_aider_retrieve.py
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# Aider's real repo-map is imported lazily inside build_repomap() so this module
# can be imported under bert's .venv (which has no `aider`); only the actual
# graph/Aider arm needs the aider venv (/tmp/aider_venv).

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _split_identifiers(text: str) -> set[str]:
    """Lexical tokens for the chunk-overlap scorer (snake/Camel aware),
    identical to b9_graph_retrieve so the filerank arm is comparable."""
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


def build_repomap(corpus_dir: str | Path):
    """Instantiate the real Aider RepoMap over the corpus. Returns
    (RepoMap, list[abs .py paths], root)."""
    corpus_dir = Path(corpus_dir).resolve()
    from aider.io import InputOutput
    from aider.models import Model
    from aider.repomap import RepoMap
    io = InputOutput()
    main_model = Model("gpt-4o")  # offline; only used for token budget, not ranking
    rm = RepoMap(map_tokens=8192, root=str(corpus_dir), main_model=main_model, io=io, verbose=False)
    files = [str(p) for p in sorted(corpus_dir.rglob("*.py"))]
    return rm, files, str(corpus_dir)


def _mentioned_idents(question: str) -> set[str]:
    """Exactly Aider's get_ident_mentions: split on non-word chars."""
    return set(re.split(r"\W+", question))


def _mentioned_fnames(question: str, rel_files: list[str]) -> set[str]:
    """Aider's get_ident_filename_matches: file whose stem (>=5 chars) equals a
    mentioned ident (lowercased). Returns rel_fnames."""
    idents = {w.lower() for w in _mentioned_idents(question) if len(w) >= 5}
    out: set[str] = set()
    for rel in rel_files:
        stem = Path(rel).stem.lower()
        if len(stem) >= 5 and stem in idents:
            out.add(rel)
    return out


def aider_rank(rm, all_files: list[str], root: str, question: str):
    """Return Aider's ranked output as an ordered list of (rel_fname, line|None),
    best first. `line` is the definition line for tag entries, None for bare
    file entries Aider appends for files without ranked defs."""
    rel_files = [str(Path(f).relative_to(root)) for f in all_files]
    idents = _mentioned_idents(question)
    fnames = _mentioned_fnames(question, rel_files)
    ranked = rm.get_ranked_tags(
        chat_fnames=[], other_fnames=all_files,
        mentioned_fnames=fnames, mentioned_idents=idents,
    )
    out: list[tuple[str, int | None]] = []
    for rt in ranked:
        if len(rt) >= 4:  # Tag(rel_fname, fname, line, name, kind)
            out.append((rt[0], int(rt[2])))
        else:             # bare (rel_fname,)
            out.append((rt[0], None))
    return out


# ---- chunk store (same DB the other arms use) -------------------------------
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


def load_file_chunks(db_path: str | Path) -> dict[str, list[tuple[int, str]]]:
    """rel_fname -> [(chunk_idx, content), ...] sorted by chunk_idx."""
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT path, chunk_idx, content FROM chunks").fetchall()
    finally:
        con.close()
    out: dict[str, list[tuple[int, str]]] = {}
    for path, idx, content in rows:
        rel = _chunk_to_corpus_file(path)
        out.setdefault(rel, []).append((int(idx), content))
    for rel in out:
        out[rel].sort(key=lambda x: x[0])
    return out


def _line_text(corpus_dir: str | Path, rel_fname: str, line: int) -> str:
    """Source text of (0-based) `line` in the corpus file, stripped."""
    fp = Path(corpus_dir) / rel_fname
    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if 0 <= line < len(lines):
        return lines[line].strip()
    return ""


def _chunk_for_line(corpus_dir, rel_fname, line, file_chunks) -> tuple[str, str] | None:
    """The chunk (by chunk_idx order) whose content contains the source line at
    `line`. Robust to the 1500/100 char chunking. Falls back to first chunk."""
    chunks = file_chunks.get(rel_fname)
    if not chunks:
        return None
    txt = _line_text(corpus_dir, rel_fname, line) if line is not None else ""
    if txt and len(txt) >= 4:
        for idx, content in chunks:
            if txt in content:
                return (f"{rel_fname}#chunk{idx}", content)
    # bare-file entry or generic line: return file's first chunk
    idx, content = chunks[0]
    return (f"{rel_fname}#chunk{idx}", content)


def aider_tags_retrieve(question, *, rm, all_files, root, corpus_dir,
                        file_chunks, top_n: int = 10) -> list[tuple[str, str]]:
    """Walk Aider's ranked definitions; take the chunk containing each, in order,
    deduped. This is what Aider's repo map literally puts in front of the model."""
    ranked = aider_rank(rm, all_files, root, question)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rel_fname, line in ranked:
        hit = _chunk_for_line(corpus_dir, rel_fname, line, file_chunks)
        if hit and hit[0] not in seen:
            seen.add(hit[0])
            out.append(hit)
        if len(out) >= top_n:
            break
    return out


def aider_filerank_retrieve(question, *, rm, all_files, root, corpus_dir,
                            file_chunks, top_n: int = 10) -> list[tuple[str, str]]:
    """Aider PageRank -> per-file score (from rank order), then the SAME chunk
    selector as b9_graph_retrieve (file_score * (1 + lexical_overlap)). Isolates
    the ranking source as the only variable vs the from-scratch graph arm."""
    ranked = aider_rank(rm, all_files, root, question)
    # first appearance position -> score; earlier = higher
    file_score: dict[str, float] = {}
    for pos, (rel_fname, _line) in enumerate(ranked):
        if rel_fname not in file_score:
            file_score[rel_fname] = 1.0 / (pos + 1)
    qtokens = _split_identifiers(question)
    scored = []
    for rel_fname, chunks in file_chunks.items():
        base = file_score.get(rel_fname, 0.0)
        for idx, content in chunks:
            lex = len(qtokens & _split_identifiers(content))
            score = base * (1.0 + lex) + 1e-4 * lex
            scored.append((score, f"{rel_fname}#chunk{idx}", content))
    scored.sort(key=lambda x: -x[0])
    return [(cid, content) for _s, cid, content in scored[:top_n]]


if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b9_corpus"
    db = sys.argv[2] if len(sys.argv) > 2 else "/tmp/b9_graph_lab/memory.db"
    rm, files, root = build_repomap(corpus)
    fc = load_file_chunks(db)
    print(f"aider RepoMap: {len(files)} files, {len(fc)} files with chunks")
    q = "In httpx, which content encodings does the SUPPORTED_DECODERS registry support?"
    print(f"\nquery: {q}\n")
    print("-- aider_tags (literal repo-map order) --")
    for cid, c in aider_tags_retrieve(q, rm=rm, all_files=files, root=root,
                                      corpus_dir=corpus, file_chunks=fc, top_n=5):
        print(f"  {cid}: {c[:70].strip()!r}")
    print("-- aider_filerank (same selector as graph arm) --")
    for cid, c in aider_filerank_retrieve(q, rm=rm, all_files=files, root=root,
                                          corpus_dir=corpus, file_chunks=fc, top_n=5):
        print(f"  {cid}: {c[:70].strip()!r}")
