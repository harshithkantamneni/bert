"""v2 grading — objective first, judge only where unavoidable, and the judge is
validated, not trusted.

v1 graded everything with a single LLM judge (unvalidated, biased, noisy). v2:

  - PROGRAMMATIC grading for the deterministic-gold tier (factual lookups):
    normalized exact / numeric / regex match against the gold. No LLM, no noise.
  - MULTI-JUDGE LLM grading for open-ended (multi-hop) answers: N non-Claude
    judges, majority vote, with inter-judge Cohen's kappa reported.
  - JUDGE VALIDATION: on the deterministic tier we have BOTH the programmatic
    verdict (ground truth) and the judge verdict, so we measure judge-vs-truth
    agreement and report it. A judge that disagrees with objective truth is not
    trusted on the open-ended tier either.

Programmatic grader is pure + unit-tested; judge functions use core.provider.
"""

from __future__ import annotations

import re

# ── programmatic grading ─────────────────────────────────────────────
_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding quotes/backticks/punct."""
    s = (s or "").strip().lower()
    s = _WS.sub(" ", s)
    return s.strip("`'\"() .,:;")


def _nums(s: str) -> list[float]:
    out = []
    for m in _NUM.findall(s or ""):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def _is_numeric(s: str) -> bool:
    """True if the gold is a single bare number (int/float/sci), so it must be
    graded by numeric equality, NOT substring (else gold '5' matches '50'/'15')."""
    return bool(re.fullmatch(r"\s*[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?\s*", s or ""))


def grade_programmatic(candidate: str, gold_answer: str,
                       answer_regex: str | None = None) -> int:
    """1 if the candidate answer contains the gold fact, else 0. In order:
    (1) explicit answer_regex; (2) for a purely-numeric gold, NUMERIC equality
    only (tolerant 5 vs 5.0, but 5 != 50); (3) otherwise a WORD-BOUNDED
    substring match (so 'br' does not match 'bright')."""
    cand = candidate or ""
    if not gold_answer and not answer_regex:
        return 0
    # 1. explicit regex (authoritative for regex/string facts)
    if answer_regex:
        try:
            if re.search(answer_regex, cand, re.IGNORECASE | re.DOTALL):
                return 1
        except re.error:
            pass
    # 2. purely-numeric gold -> numeric equality only (no substring false-positives)
    if _is_numeric(gold_answer):
        gv = float(gold_answer)
        for cn in _nums(cand):
            if abs(cn - gv) < 1e-9 or (gv != 0 and abs(cn - gv) / abs(gv) < 1e-6):
                return 1
        return 0
    # 3. non-numeric gold -> word-bounded normalized substring
    g = normalize(gold_answer)
    if not g:
        return 0
    if re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", normalize(cand)):
        return 1
    return 0


# ── multi-judge LLM grading (open-ended tier) ────────────────────────
# judges must be INDEPENDENT of the answer model (Claude) and actually reachable.
# mistral(429)/openrouter(402 no-credits)/cerebras(404 bad-model) were all dead and
# silently zeroed every judge-graded answer — these three are verified live.
JUDGES = [("nvidia", "meta/llama-3.3-70b-instruct"),
          ("nvidia", "meta/llama-3.1-70b-instruct"),
          ("groq", "llama-3.3-70b-versatile")]


def _judge_once(prov_name: str, model: str, question: str, gold: str, answer: str) -> int | None:
    import json
    from core import provider as prov
    sysp = ("You grade whether a candidate answer is factually correct given the "
            "gold answer. Strict on facts, lenient on wording. Return ONLY JSON: "
            '{"correct": true|false}.')
    userp = f"QUESTION: {question}\nGOLD: {gold}\nCANDIDATE: {answer}\n\nJSON only."
    try:
        r = prov.call(prov_name, [{"role": "system", "content": sysp},
                                  {"role": "user", "content": userp}],
                      model=model, max_tokens=40, temperature=0.0,
                      response_format={"type": "json_object"}, timeout=40.0)
        if r.finish_reason == "error" or (r.text or "").startswith("[bert]"):
            return None
        return 1 if bool(json.loads(r.text).get("correct")) else 0
    except Exception:  # noqa: BLE001
        return None


def grade_judges(question: str, gold: str, answer: str,
                 judges=JUDGES) -> dict:
    """N-judge majority vote. Returns {verdict, votes, n_valid}."""
    votes = [v for j in judges if (v := _judge_once(j[0], j[1], question, gold, answer)) is not None]
    if not votes:
        return {"verdict": 0, "votes": [], "n_valid": 0}
    verdict = 1 if sum(votes) * 2 > len(votes) else 0
    return {"verdict": verdict, "votes": votes, "n_valid": len(votes)}


if __name__ == "__main__":
    # programmatic grader unit tests (pure, no network)
    cases = [
        # (candidate, gold, regex, expected)
        ("The default is 5.0 seconds", "5.0", None, 1),
        ("It defaults to 5", "5.0", None, 1),                 # numeric tolerance
        ("The value is `[0-9]+`", "[0-9]+", None, 1),         # normalized substring
        ("max_redirects is 20", "20", None, 1),
        ("It is gzip, deflate, br", "br", None, 1),
        ("I don't know", "5.0", None, 0),
        ("The answer is 50", "5.0", None, 0),                 # wrong number
        ("returns SEE_OTHER", "SEE_OTHER", r"SEE[_ ]?OTHER", 1),
        ("the status is see other", "SEE_OTHER", r"SEE[_ ]?OTHER", 1),
        # regression: numeric false-positives (gold N must not match a longer number)
        ("max_redirects is 200", "20", None, 0),
        ("the value is 15", "5", None, 0),
        ("it is 100 connections", "100", None, 1),
        # regression: word-boundary (short gold must not match inside a word)
        ("the screen is bright", "br", None, 0),
        ("encodings: gzip, deflate, br", "br", None, 1),
    ]
    ok = 0
    for cand, gold, rgx, exp in cases:
        got = grade_programmatic(cand, gold, rgx)
        flag = "ok" if got == exp else "FAIL"
        if got == exp:
            ok += 1
        else:
            print(f"  {flag}: grade({cand!r},{gold!r},{rgx!r})={got} exp={exp}")
    print(f"programmatic grader: {ok}/{len(cases)} cases pass")
    assert ok == len(cases), "programmatic grader has failures"
    print("v2_grader self-test: OK")
