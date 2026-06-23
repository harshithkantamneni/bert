"""m1 memory-MCP benchmark — corpus compiler.

Builds a prose "project memory" haystack: a chronological stream of dated
engineering-log sessions (markdown). A small set of evidence NEEDLES (generated
separately, lexically disjoint from their questions) are planted at varied
timeline depths among procedurally-generated FILLER sessions in the same
fictional project ("Helix"). Filler is templated so the haystack scales to
arbitrary token counts cheaply and deterministically.

The same haystack is compiled at several sizes (S fits a context window, M/L
exceed it) so the arms can be compared as a function of corpus size — the
crossover curve where full-context truncates but retrieval/memory holds.

Determinism: everything is seeded; evidence is read from a frozen file; gold
records each needle's session index for Recall@k. No regeneration drift.

  .venv/bin/python benchmarks/m1_corpus.py --evidence <gold.json> --size S|M|L
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "benchmarks/results/m1"
SESS = Path("/tmp/m1_haystacks")  # bulky session files live OUTSIDE iCloud-synced ~/Desktop
TEAMS = ["Search", "Platform", "Growth", "SRE", "Data"]
PEOPLE = ["Ana", "Ben", "Carla", "Dmitri", "Elena", "Farid", "Gita", "Hugo"]
COMPONENTS = ["ingestion pipeline", "embedding service", "vector index store",
              "query router", "reranker", "billing service", "web dashboard",
              "eval harness", "auth gateway", "telemetry collector"]
SIZES = {"S": 115_000, "M": 1_500_000, "L": 4_000_000}  # target tokens

# filler templates — plausible Helix engineering prose that is NOT a needle
_FILLER = [
    "Standup: {p} is wrapping up the {c} refactor; {p2} unblocked the {c2} flake. "
    "No deploys planned today. Carryover: tidy the {c} dashboards.",
    "Design review for {c}: discussed back-pressure and retry budgets. Action items "
    "assigned to {p}. Decision deferred to next week pending load numbers.",
    "{p} merged the {c} cleanup (#{num}); cut p99 noise but no functional change. "
    "Follow-up: backfill the {c2} metrics once the migration settles.",
    "Weekly {t} sync: roadmap unchanged. {c} owners to draft an RFC; {p2} to review. "
    "Hiring: one open req on {t}. Nothing blocking ship this week.",
    "On-call notes ({p}): two paging alerts on {c}, both auto-resolved. Bumped the "
    "{c2} alert threshold to reduce noise. No customer impact.",
    "Retro: shipping cadence steady. {c} latency within SLO. {p} flagged {c2} tech "
    "debt for grooming. Kudos to {p2} for the dashboard polish.",
    "Notes from {c} grooming: split the epic into ingestion + indexing tracks. "
    "{p} takes the first, {p2} the second. Estimates pending.",
    "{p} prototyped a {c} tweak in a branch; results inconclusive, parked for now. "
    "Will revisit after the {c2} work lands.",
]


def _est_tokens(s: str) -> int:
    return len(s) // 4


def _filler_session(rng: random.Random) -> str:
    # 3-5 distinct template lines per session → realistic ~250-tok multi-paragraph
    # note, so the haystack scales without exploding the file count.
    k = rng.randint(3, 5)
    lines = []
    for t in rng.sample(_FILLER, k):
        lines.append(t.format(p=rng.choice(PEOPLE), p2=rng.choice(PEOPLE), t=rng.choice(TEAMS),
                              c=rng.choice(COMPONENTS), c2=rng.choice(COMPONENTS),
                              num=rng.randint(100, 9999)))
    return "\n\n".join(lines)


def _date(i: int) -> str:
    # ~3 sessions/day starting 2024-01-02; deterministic from index
    day = i // 3
    y, m, d = 2024, 1, 2 + day
    while d > 28:
        d -= 28; m += 1
        if m > 12:
            m = 1; y += 1
    return f"{y}-{m:02d}-{d:02d}"


def _session_md(idx: int, author: str, comp: str, body: str) -> str:
    return f"# {_date(idx)} · {author} · {comp}\n\n{body}\n"


def compile_haystack(evidence: list[dict], target_tokens: int, seed: int = 7) -> tuple[list[str], dict]:
    """Return (ordered session bodies, gold). Each needle's evidence_prose (and
    second_evidence_prose, if any) is planted as its own session at a seeded,
    spread-out position; gold[id] records the session indices that hold it."""
    rng = random.Random(seed)
    # one or two evidence sessions per needle
    ev_sessions = []  # (needle_id, prose)
    for e in evidence:
        ev_sessions.append((e["id"], e["evidence_prose"], "first"))
        if e.get("second_evidence_prose"):
            ev_sessions.append((e["id"], e["second_evidence_prose"], "second"))

    # estimate filler count to hit target (minus evidence budget)
    ev_tok = sum(_est_tokens(p) for _, p, _ in ev_sessions)
    avg_filler = 200  # ~tokens per multi-paragraph filler session
    n_filler = max(len(ev_sessions) * 4, (target_tokens - ev_tok) // avg_filler)

    total = n_filler + len(ev_sessions)
    # choose distinct spread-out slots for evidence; 'second' lands AFTER 'first'
    slots = sorted(rng.sample(range(total), len(ev_sessions)))
    rng.shuffle(ev_sessions)
    # keep second after first for the same needle by sorting per-needle on slot
    placement = {}  # slot -> (needle_id, prose)
    # assign: give 'first' an earlier slot, 'second' a later slot when paired
    firsts = [x for x in ev_sessions if x[2] == "first"]
    seconds = [x for x in ev_sessions if x[2] == "second"]
    half = len(slots) // 2 or 1
    early, late = slots[:max(half, len(firsts))], slots[max(half, len(firsts)):]
    for x in firsts:
        placement[early.pop(0) if early else late.pop(0)] = x
    leftover = early + late
    for x in seconds:
        placement[leftover.pop(0) if leftover else slots[-1]] = x

    sessions, gold_loc = [], {}
    for idx in range(total):
        if idx in placement:
            nid, prose, _which = placement[idx]
            sessions.append(_session_md(idx, rng.choice(PEOPLE), rng.choice(COMPONENTS), prose))
            gold_loc.setdefault(nid, []).append(idx)
        else:
            sessions.append(_session_md(idx, rng.choice(PEOPLE), rng.choice(COMPONENTS), _filler_session(rng)))

    gold = []
    for e in evidence:
        gold.append({"id": e["id"], "category": e["category"], "question": e["question"],
                     "gold_answer": e["gold_answer"], "grade_mode": "judge",
                     "evidence_sessions": sorted(gold_loc.get(e["id"], []))})
    return sessions, {"questions": gold, "n_sessions": total,
                      "approx_tokens": sum(_est_tokens(s) for s in sessions)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=str(OUT / "evidence.json"))
    ap.add_argument("--size", choices=list(SIZES), default="S")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    evidence = json.loads(Path(args.evidence).read_text())
    sessions, meta = compile_haystack(evidence, SIZES[args.size], seed=args.seed)

    sdir = SESS / f"haystack_{args.size}" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    for old in sdir.glob("*.md"):
        old.unlink()
    for i, body in enumerate(sessions):
        (sdir / f"{i:06d}.md").write_text(body)
    (OUT / f"gold_{args.size}.json").write_text(json.dumps(meta, indent=2))
    print(f"[m1] size={args.size}: {meta['n_sessions']} sessions (~{meta['approx_tokens']:,} tok), "
          f"{len(meta['questions'])} questions -> {sdir}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
