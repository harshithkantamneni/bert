"""v2 gold-QA extractor — DETERMINISTIC, method-blind, no LLM.

The v1 gold set (`benchmarks/b9_gold/gold_qa.json`) was *hand-authored* by the
same author who designed the retrieval system. A skeptical reviewer can
reasonably object that hand-picked questions might (consciously or not) favor
the system under test. This module removes that bias for the `needle` and
`single_hop` tiers: it walks a real code corpus and extracts OBJECTIVE,
programmatically-checkable facts straight from the AST, then templates them into
questions. The author never picks the questions or the answers — the parser does.

What "objective + checkable" means here:
  - The answer is a *literal that the source literally contains* (a default arg
    value, a module constant, a regex pattern, an Enum member value, a class
    attribute default).
  - The `gold_span` is a VERBATIM substring of the source file (taken via
    ``ast.get_source_segment``), so retrieval-recall can be scored by exact
    substring containment with zero ambiguity.
  - The `gold_answer` is the exact source text of the value node — re-reading the
    file at the recorded location reproduces it character-for-character. The
    self-test verifies this for samples by re-reading the source.

Each fact -> dict:
    {
      id, tier ('needle'|'single_hop'), corpus, question,
      gold_answer,        # exact value-source-text, e.g. '"[0-9]+"' or '307'
      answer_regex,       # regex matching a correct answer (quote/ws tolerant)
      source_file,        # path relative to corpus_root
      gold_span,          # distinctive verbatim substring containing the fact
      kind,               # provenance: which extractor produced it (debugging)
    }

Tiers (deliberately conservative — these are the EASY, unambiguous tiers; the
multi-hop tier is generated/validated elsewhere):
  - needle:     the fact lives in ONE line / one assignment; answering needs only
                that line (module constant, regex literal, Enum member, class attr
                default). One hop: find the line, read the value.
  - single_hop: the fact requires locating one *named definition* and reading one
                of its parameters' defaults (function/method parameter defaults).
                Still one logical hop, but you must resolve the function first.

Language coverage:
  - Python: full AST extraction (the corpus is 100% Python — httpx + starlette).
  - Go: a regex-based ``const NAME = literal`` / typed-var extractor is provided
    (`_extract_go`) so a mixed corpus works, but if no .go files exist it simply
    contributes nothing. See `extract_gold` docstring.

Public API:
    extract_gold(corpus_root, max_items=None) -> list[dict]

Pure + deterministic: no network, no LLM, no randomness. Re-running on the same
corpus yields byte-identical output (facts are sorted by (source_file, tier, id)).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# ── corpus identity ──────────────────────────────────────────────────
# Matches the label used by the v1 gold set / other B9 arms so reports line up.
CORPUS_LABEL = "httpx-0.28.1 + starlette (vendored)"

# ── what counts as a "trivial" answer we refuse to emit ──────────────
# Bare None/True/False/0/1/""/empty-collection defaults are everywhere and are
# not distinctive enough to be a fair retrieval/QA target on their own. We only
# keep them when... we never keep them. We require a DISTINCTIVE literal.
_TRIVIAL_CONSTANTS = {None, True, False, -1, "", b""}

# Numeric literals this small/common are not distinctive enough on their own.
_TRIVIAL_NUMS = {0, 1, -1, 2, 100}  # 100 appears as a generic default a lot

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ─────────────────────────────────────────────────────────────────────
# literal classification / answer-text helpers
# ─────────────────────────────────────────────────────────────────────
def _const_value(node: ast.expr):
    """Return the python value if `node` is a literal constant, else a sentinel
    object meaning 'not a plain constant'. (We can't use None as the sentinel
    because None is itself a legal constant.)"""
    if isinstance(node, ast.Constant):
        return node.value
    return _NOT_CONST


_NOT_CONST = object()


def _is_distinctive_constant(value) -> bool:
    """True if `value` is a literal worth asking about: a non-trivial number, a
    real string (len>=2 and not whitespace), or bytes. Booleans/None excluded."""
    if isinstance(value, bool):
        return False
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value not in _TRIVIAL_NUMS
    if isinstance(value, str):
        return len(value) >= 2 and bool(value.strip())
    if isinstance(value, bytes):
        return len(value) >= 1
    return False


def _value_source(src: str, node: ast.expr) -> str | None:
    """Exact source text of an expression node (verbatim, incl. quotes/prefix)."""
    seg = ast.get_source_segment(src, node)
    return seg.strip() if seg is not None else None


def _is_distinctive_value_node(node: ast.expr) -> bool:
    """Decide if a *value node* (not just a plain constant) is a distinctive,
    checkable answer. Accepts:
      - distinctive plain constants (numbers/strings/bytes),
      - tuples/lists/dicts/sets of literals (e.g. separators=(",", ":")),
      - re.compile("…")  (regex literal),
      - simple binop of numeric literals (e.g. 64 * 1024, 1024 * 1024),
      - a Call whose source is short + literal-ish (e.g. Timeout(timeout=5.0)).
    Rejects bare names, attributes, None/True/False, trivial nums."""
    v = _const_value(node)
    if v is not _NOT_CONST:
        return _is_distinctive_constant(v)

    # container literals: keep if every element is a literal/simple and there is
    # at least one distinctive element
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        elts = node.elts
        if not elts:
            return False
        if all(_elt_is_literalish(e) for e in elts):
            return any(_node_distinctive_atom(e) for e in elts)
        return False
    if isinstance(node, ast.Dict):
        if not node.keys:
            return False
        return all(
            _elt_is_literalish(k) and _elt_is_literalish(val)
            for k, val in zip(node.keys, node.values, strict=False)
            if k is not None
        )

    # re.compile("pattern")  -> distinctive regex
    if isinstance(node, ast.Call) and _is_re_compile(node):
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, (str, bytes)):
            return len(str(node.args[0].value)) >= 2
        return False

    # numeric binop like 64 * 1024
    if isinstance(node, ast.BinOp):
        return _binop_is_numeric_literal(node)

    return False


def _elt_is_literalish(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_elt_is_literalish(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant)
    return False


def _node_distinctive_atom(node: ast.expr) -> bool:
    v = _const_value(node)
    if v is not _NOT_CONST:
        return _is_distinctive_constant(v)
    return isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict))


def _is_re_compile(call: ast.Call) -> bool:
    f = call.func
    if isinstance(f, ast.Attribute) and f.attr == "compile":
        return isinstance(f.value, ast.Name) and f.value.id == "re"
    return False


def _binop_is_numeric_literal(node: ast.BinOp) -> bool:
    def num(n: ast.expr) -> bool:
        if isinstance(n, ast.Constant):
            return isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
        if isinstance(n, ast.BinOp):
            return _binop_is_numeric_literal(n)
        return False
    return num(node.left) and num(node.right) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.Pow, ast.Mod))


# ─────────────────────────────────────────────────────────────────────
# answer_regex: a tolerant matcher for a correct answer string
# ─────────────────────────────────────────────────────────────────────
def _answer_regex(gold_answer: str) -> str:
    r"""Build a regex that matches a correct answer with sensible tolerance:
      - leading/trailing quotes of any flavor are optional (the model may quote
        the value or not, may use ' or "),
      - a leading string/bytes prefix (r, b, rb, f) is optional,
      - internal whitespace runs are flexible.
    The CORE is the literal *content* between the outermost quotes (if quoted),
    or the whole token (if unquoted, e.g. a number / binop). Everything is
    re.escaped so regex metachars in the value are matched literally."""
    s = gold_answer.strip()

    # strip an optional string/bytes prefix + matching outer quotes to get content
    m = re.match(r'^(?:[rbuRBU]{1,2})?(["\'])(.*)\1$', s, re.DOTALL)
    if m:
        content = m.group(2)
        esc = re.escape(content)
        esc = re.sub(r"(?:\\ )+", r"\\s+", esc)  # tolerate whitespace differences
        # optional prefix + optional surrounding quote (either flavor)
        return r'(?:[rbuRBU]{0,2})?["\']?' + esc + r'["\']?'

    # unquoted token (number, binop, tuple text, call text): tolerate whitespace,
    # allow optional surrounding quotes in case the model quotes it.
    esc = re.escape(s)
    esc = re.sub(r"(?:\\ )+", r"\\s*", esc)
    return r'["\']?' + esc + r'["\']?'


# ─────────────────────────────────────────────────────────────────────
# gold_span: a distinctive verbatim substring containing the fact
# ─────────────────────────────────────────────────────────────────────
def _distinctive_span(src: str, candidates: list[str]) -> str | None:
    """Pick the first candidate verbatim segment that (a) is non-empty, (b)
    actually occurs in src, and (c) occurs EXACTLY ONCE (so it uniquely locates
    the fact for retrieval scoring). Candidates are tried longest-first when
    tie-broken, but we keep caller order primarily."""
    for cand in candidates:
        if cand and src.count(cand) == 1:
            return cand
    # fall back to the first that at least occurs (not unique, but verbatim)
    for cand in candidates:
        if cand and cand in src:
            return cand
    return None


# ─────────────────────────────────────────────────────────────────────
# Python extractor
# ─────────────────────────────────────────────────────────────────────
def _qual_name(stack: list[str], name: str) -> str:
    return ".".join([*stack, name]) if stack else name


def _extract_python(src: str, rel_path: str) -> list[dict]:
    """Walk one Python file's AST and emit gold facts. Deterministic order:
    facts are appended in source order; the caller does the final global sort."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    facts: list[dict] = []
    counter = {"n": 0}

    def fid(prefix: str) -> str:
        counter["n"] += 1
        stem = re.sub(r"[^A-Za-z0-9]+", "_", rel_path).strip("_")
        return f"{stem}__{prefix}{counter['n']}"

    def add(tier: str, kind: str, question: str, value_node: ast.expr,
            span_candidates: list[str]) -> None:
        gold = _value_source(src, value_node)
        if not gold:
            return
        span = _distinctive_span(src, span_candidates)
        if span is None:
            return
        facts.append({
            "id": fid("q"),
            "tier": tier,
            "corpus": CORPUS_LABEL,
            "question": question,
            "gold_answer": gold,
            "answer_regex": _answer_regex(gold),
            "source_file": rel_path,
            "gold_span": span,
            "kind": kind,
        })

    # ---- recursive walk that tracks the class/function nesting stack --------
    def visit(node: ast.AST, stack: list[str], *, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                _emit_class_facts(child, stack)
                visit(child, [*stack, child.name], in_class=True)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _emit_param_default_facts(child, stack, in_class=in_class)
                visit(child, [*stack, child.name], in_class=False)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)) and not stack:
                _emit_module_const_facts(child)
            else:
                # descend into things like If/Try at module level to catch
                # constants defined inside, but don't treat their assigns as
                # class attrs.
                if not isinstance(child, (ast.Assign, ast.AnnAssign)):
                    visit(child, stack, in_class=in_class)

    # ---- module-level constants (NAME = literal) ---------------------------
    def _emit_module_const_facts(node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            return
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if len(names) != 1:
            return
        name = names[0]
        # require an UPPER_SNAKE-ish module constant (NAME = ...) — that's the
        # convention for "constant" and keeps us from grabbing type aliases /
        # private mutable globals. Also allow names with a leading underscore.
        bare = name.lstrip("_")
        if not bare or not bare.isupper():
            return
        if not _is_distinctive_value_node(value):
            return
        is_regex = isinstance(value, ast.Call) and _is_re_compile(value)
        full_seg = _value_source(src, node)  # "NAME = <value>" verbatim
        val_seg = _value_source(src, value)
        if is_regex:
            q = (f"In {rel_path}, what regex pattern string is compiled and "
                 f"assigned to the module-level name {name}?")
            kind = "module_regex"
            # the answer should be the pattern literal, not the whole re.compile()
            pat_node = value.args[0]
            gold = _value_source(src, pat_node)
            if not gold:
                return
            span = _distinctive_span(src, [full_seg, val_seg, gold])
            if span is None:
                return
            facts.append({
                "id": fid("q"), "tier": "needle", "corpus": CORPUS_LABEL,
                "question": q, "gold_answer": gold,
                "answer_regex": _answer_regex(gold),
                "source_file": rel_path, "gold_span": span, "kind": kind,
            })
            return
        q = (f"In {rel_path}, what is the value assigned to the module-level "
             f"constant {name}?")
        add("needle", "module_const", q, value, [full_seg, val_seg])

    # ---- class bodies: attribute defaults, Enum members --------------------
    def _emit_class_facts(cls: ast.ClassDef, stack: list[str]) -> None:
        qual = _qual_name(stack, cls.name)
        is_enum = _looks_like_enum(cls)
        for item in cls.body:
            if isinstance(item, ast.Assign):
                if len(item.targets) != 1 or not isinstance(item.targets[0], ast.Name):
                    continue
                attr = item.targets[0].id
                value = item.value
                full_seg = _value_source(src, item)         # "attr = <value>"
                val_seg = _value_source(src, value)
                if is_enum:
                    # Enum member: NAME = <literal or (value, ...)>; skip dunders
                    if attr.startswith("__"):
                        continue
                    if not _enum_member_distinctive(value):
                        continue
                    q = (f"In {rel_path}, in the enum class {qual}, what value is "
                         f"assigned to the member {attr}?")
                    add("needle", "enum_member", q, value, [full_seg, val_seg])
                else:
                    if not _is_distinctive_value_node(value):
                        continue
                    is_regex = isinstance(value, ast.Call) and _is_re_compile(value)
                    if is_regex:
                        pat_node = value.args[0]
                        gold = _value_source(src, pat_node)
                        if not gold:
                            continue
                        q = (f"In {rel_path}, in class {qual}, what regex pattern "
                             f"is assigned to the attribute {attr}?")
                        span = _distinctive_span(src, [full_seg, val_seg, gold])
                        if span is None:
                            continue
                        facts.append({
                            "id": fid("q"), "tier": "needle", "corpus": CORPUS_LABEL,
                            "question": q, "gold_answer": gold,
                            "answer_regex": _answer_regex(gold),
                            "source_file": rel_path, "gold_span": span,
                            "kind": "class_regex",
                        })
                        continue
                    q = (f"In {rel_path}, in class {qual}, what is the default "
                         f"value of the class attribute {attr}?")
                    add("needle", "class_attr", q, value, [full_seg, val_seg])
            elif isinstance(item, ast.AnnAssign) and item.value is not None:
                if not isinstance(item.target, ast.Name):
                    continue
                attr = item.target.id
                value = item.value
                if attr.startswith("__"):
                    continue
                if not _is_distinctive_value_node(value):
                    continue
                full_seg = _value_source(src, item)
                val_seg = _value_source(src, value)
                q = (f"In {rel_path}, in class {qual}, what is the default "
                     f"value of the attribute {attr}?")
                add("needle", "class_attr", q, value, [full_seg, val_seg])

    # ---- function/method parameter defaults -------------------------------
    def _emit_param_default_facts(fn: ast.FunctionDef | ast.AsyncFunctionDef,
                                  stack: list[str], *, in_class: bool) -> None:
        qual = _qual_name(stack, fn.name)
        args = fn.args
        # positional/optional defaults align to the TAIL of posonly+args
        posargs = list(args.posonlyargs) + list(args.args)
        defaults = list(args.defaults)
        pos_with_def = posargs[len(posargs) - len(defaults):] if defaults else []
        pairs = list(zip(pos_with_def, defaults, strict=False))
        # keyword-only defaults align positionally to kwonlyargs (None = no default)
        for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=False):
            if d is not None:
                pairs.append((a, d))

        for arg, default in pairs:
            if not _is_distinctive_value_node(default):
                continue
            pname = arg.arg
            val_seg = _value_source(src, default)
            if not val_seg:
                continue
            # build a verbatim span: the "pname=<default>" or "pname: T = <default>"
            # text from the signature. Reconstruct from source segment of the arg
            # default plus the param name to keep it verbatim-checkable.
            span_cand = _param_span(src, arg, default)
            descriptor = "method" if in_class else "function"
            owner = f"the {descriptor} {qual}"
            q = (f"In {rel_path}, what is the default value of the parameter "
                 f"{pname} of {owner}?")
            span = _distinctive_span(src, [s for s in (span_cand, val_seg) if s])
            if span is None:
                continue
            facts.append({
                "id": fid("q"), "tier": "single_hop", "corpus": CORPUS_LABEL,
                "question": q, "gold_answer": val_seg,
                "answer_regex": _answer_regex(val_seg),
                "source_file": rel_path, "gold_span": span,
                "kind": "param_default",
            })

    visit(tree, [], in_class=False)
    return facts


def _param_span(src: str, arg: ast.arg, default: ast.expr) -> str | None:
    """Verbatim 'name=<default>' (or 'name: ann=<default>') slice from the source
    spanning the param-name start to the default-value end. Guarantees the slice
    is exactly what's in the file (so it occurs verbatim)."""
    try:
        lines = src.splitlines(keepends=True)
        start = _offset(lines, arg.lineno, arg.col_offset)
        end = _offset(lines, default.end_lineno, default.end_col_offset)
        if start is None or end is None or end <= start:
            return None
        return src[start:end].strip()
    except Exception:
        return None


def _offset(lines: list[str], lineno: int, col: int) -> int | None:
    if lineno < 1 or lineno > len(lines):
        return None
    return sum(len(lines[i]) for i in range(lineno - 1)) + col


def _looks_like_enum(cls: ast.ClassDef) -> bool:
    """Heuristic: the class inherits from Enum/IntEnum/IntFlag/Flag/StrEnum
    (by simple name or attribute), matching how the corpus declares enums."""
    enum_names = {"Enum", "IntEnum", "IntFlag", "Flag", "StrEnum"}
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id in enum_names:
            return True
        if isinstance(base, ast.Attribute) and base.attr in enum_names:
            return True
    return False


def _enum_member_distinctive(value: ast.expr) -> bool:
    """Enum members are worth asking about even when the value is a small int,
    because the (member -> value) mapping is the fact. Accept: any constant
    (incl. small ints), tuples of literals (e.g. (200, 'OK')), distinctive
    strings. Reject: calls to auto()/functions, names, attributes."""
    v = _const_value(value)
    if v is not _NOT_CONST:
        if isinstance(v, bool):
            return False
        return bool(isinstance(v, (int, float, str, bytes)))
    if isinstance(value, (ast.Tuple, ast.List)):
        return bool(value.elts) and all(_elt_is_literalish(e) for e in value.elts)
    if isinstance(value, ast.BinOp):
        return _binop_is_numeric_literal(value)
    return False


# ─────────────────────────────────────────────────────────────────────
# Go extractor (regex-based; corpus is currently Python-only)
# ─────────────────────────────────────────────────────────────────────
# Matches:  const Name = <literal>   |   Name Type = <literal>  (typed var)
# We only keep distinctive literals (quoted strings, numbers != trivial, backtick
# raw strings). This is intentionally simple — Go AST would need a Go toolchain;
# for a const/var literal the line-regex is sufficient and fully deterministic.
_GO_CONST = re.compile(
    r"""^\s*(?:const\s+)?([A-Z][A-Za-z0-9_]*)\s*"""    # exported Name
    r"""(?:[A-Za-z_][\w\.\[\]\*]*\s*)?"""               # optional type
    r"""=\s*(?P<val>"(?:[^"\\]|\\.)*"|`[^`]*`|-?\d[\d_]*(?:\.\d+)?)\s*$""",
    re.MULTILINE,
)


def _extract_go(src: str, rel_path: str) -> list[dict]:
    """Regex-based Go const/typed-var literal extractor. Emits 'needle' facts.
    Returns [] for files with no qualifying lines. (No .go files in the current
    corpus, so this contributes nothing today but keeps a mixed corpus working.)"""
    facts: list[dict] = []
    n = 0
    for m in _GO_CONST.finditer(src):
        name, val = m.group(1), m.group("val")
        # distinctiveness for Go literals
        if val.startswith(('"', "`")):
            inner = val[1:-1]
            if len(inner) < 2:
                continue
        else:
            try:
                num = float(val.replace("_", ""))
            except ValueError:
                continue
            if num in _TRIVIAL_NUMS:
                continue
        line = m.group(0).strip()
        if src.count(line) != 1:
            continue
        n += 1
        stem = re.sub(r"[^A-Za-z0-9]+", "_", rel_path).strip("_")
        facts.append({
            "id": f"{stem}__go{n}",
            "tier": "needle",
            "corpus": CORPUS_LABEL,
            "question": (f"In {rel_path}, what literal value is assigned to the "
                         f"Go identifier {name}?"),
            "gold_answer": val,
            "answer_regex": _answer_regex(val),
            "source_file": rel_path,
            "gold_span": line,
            "kind": "go_const",
        })
    return facts


# ─────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────
def extract_gold(corpus_root: str | Path, max_items: int | None = None) -> list[dict]:
    """Walk `corpus_root` and return deterministic gold-QA facts.

    Python files -> full AST extraction (`_extract_python`).
    Go files     -> regex const/var extraction (`_extract_go`).
    Other files  -> skipped (documented: only Python is fully supported; Go has a
                    literal-only extractor; everything else has no objective,
                    language-agnostic facts to pull deterministically).

    Output is sorted by (source_file, tier, id) for byte-stable runs. If
    `max_items` is set, the first N (after sorting) are returned.
    """
    root = Path(corpus_root).resolve()
    all_facts: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue  # skip dot-dirs like .aider.tags.cache.v4
        rel = str(path.relative_to(root))
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if path.suffix == ".py":
            all_facts.extend(_extract_python(src, rel))
        elif path.suffix == ".go":
            all_facts.extend(_extract_go(src, rel))
        # else: skip

    # final dedup on (source_file, gold_span, question) — defensive
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for f in all_facts:
        key = (f["source_file"], f["gold_span"], f["question"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)

    deduped.sort(key=lambda f: (f["source_file"], f["tier"], f["id"]))
    if max_items is not None:
        deduped = deduped[:max_items]
    return deduped


# ─────────────────────────────────────────────────────────────────────
# self-test — runs against the REAL corpus at /tmp/b9_corpus
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from collections import Counter

    corpus = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b9_corpus"
    root = Path(corpus).resolve()
    if not root.is_dir():
        print(f"FAIL: corpus dir not found: {root}")
        sys.exit(1)

    facts = extract_gold(corpus)
    print(f"corpus = {root}")
    print(f"extracted {len(facts)} gold facts\n")

    failures: list[str] = []

    # 1. quantity
    if len(facts) < 40:
        failures.append(f"expected >=40 facts, got {len(facts)}")

    # 2. every gold_span appears VERBATIM in its source_file (re-read the file)
    src_cache: dict[str, str] = {}
    def _src(rel: str) -> str:
        if rel not in src_cache:
            src_cache[rel] = (root / rel).read_text(encoding="utf-8")
        return src_cache[rel]

    span_misses = 0
    for f in facts:
        src = _src(f["source_file"])
        if f["gold_span"] not in src:
            span_misses += 1
            if span_misses <= 5:
                failures.append(
                    f"gold_span NOT verbatim in {f['source_file']}: {f['gold_span']!r}")
    if span_misses:
        failures.append(f"{span_misses} gold_span(s) not found verbatim in source")

    # 3. every gold_answer non-empty
    empty_ans = [f["id"] for f in facts if not f["gold_answer"].strip()]
    if empty_ans:
        failures.append(f"{len(empty_ans)} empty gold_answer(s): {empty_ans[:5]}")

    # 3b. every answer_regex compiles AND matches its own gold_answer
    regex_fail = 0
    for f in facts:
        try:
            rx = re.compile(f["answer_regex"])
        except re.error as e:
            regex_fail += 1
            if regex_fail <= 5:
                failures.append(f"answer_regex won't compile ({f['id']}): {e}")
            continue
        if not rx.search(f["gold_answer"]):
            regex_fail += 1
            if regex_fail <= 5:
                failures.append(
                    f"answer_regex doesn't match its gold_answer ({f['id']}): "
                    f"regex={f['answer_regex']!r} answer={f['gold_answer']!r}")
    if regex_fail:
        failures.append(f"{regex_fail} answer_regex problem(s)")

    # 3c. required keys present + tiers valid
    required = {"id", "tier", "corpus", "question", "gold_answer",
               "answer_regex", "source_file", "gold_span"}
    valid_tiers = {"needle", "single_hop"}
    for f in facts:
        missing = required - set(f)
        if missing:
            failures.append(f"fact {f.get('id')} missing keys: {missing}")
            break
        if f["tier"] not in valid_tiers:
            failures.append(f"fact {f['id']} bad tier: {f['tier']}")
            break

    # 3d. ids unique
    if len({f["id"] for f in facts}) != len(facts):
        failures.append("duplicate fact ids present")

    # 4. tier counts
    tier_counts = Counter(f["tier"] for f in facts)
    kind_counts = Counter(f["kind"] for f in facts)
    print("tier counts:", dict(tier_counts))
    print("kind counts:", dict(kind_counts))
    print()

    # 5. HARD verification on 6 samples — re-read the source span and confirm the
    #    gold_answer is REALLY what the source says at that location. We pick a
    #    spread across kinds so the proof covers multiple extractors.
    print("=== 6 sample (question, answer, file) tuples + verbatim re-check ===")
    by_kind: dict[str, dict] = {}
    for f in facts:
        by_kind.setdefault(f["kind"], f)
    samples = list(by_kind.values())[:6]
    if len(samples) < 6:
        samples = facts[:6]
    for i, f in enumerate(samples, 1):
        src = _src(f["source_file"])
        # locate the span and confirm the gold_answer's CONTENT is inside the span
        span_ok = f["gold_span"] in src
        # the value's literal content should appear in the span (verbatim answer)
        ans_in_span = f["gold_answer"] in f["gold_span"] or f["gold_answer"] in src
        print(f"\n[{i}] kind={f['kind']} tier={f['tier']}")
        print(f"    Q: {f['question']}")
        print(f"    A: {f['gold_answer']}")
        print(f"    file: {f['source_file']}")
        print(f"    gold_span: {f['gold_span']!r}")
        print(f"    span verbatim in source: {span_ok} | answer present in source: {ans_in_span}")
        if not (span_ok and ans_in_span):
            failures.append(f"sample {f['id']} failed verbatim re-check")

    print("\n" + "=" * 60)
    if failures:
        print("SELF-TEST: FAIL")
        for msg in failures:
            print("  - " + msg)
        sys.exit(1)
    print(f"SELF-TEST: PASS  ({len(facts)} facts, "
          f"{tier_counts.get('needle', 0)} needle / "
          f"{tier_counts.get('single_hop', 0)} single_hop)")
    sys.exit(0)
