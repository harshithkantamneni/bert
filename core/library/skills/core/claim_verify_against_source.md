---
name: claim_verify_against_source
version: "1.0"
description: |
  Given a single factual claim and one or more candidate source URLs,
  fetch each source and judge whether the claim is supported,
  contradicted, or unmentioned. Returns a verdict plus the
  supporting / contradicting snippet(s) so a writer or critic can
  cite the exact passage instead of trusting a vibe.
trigger_when: |
  Critic or writer is about to publish a factual claim that came
  from a single retrieval or a paraphrase; the cost of being wrong
  is higher than the cost of one extra WebFetch.
inputs:
  claim:       {type: string, required: true, description: "Atomic factual statement; if compound, split first"}
  sources:     {type: list,   required: true, description: "List of URLs (already canonicalized by web_search_and_dedup, ideally)"}
  context:     {type: string, default: "", description: "Optional surrounding context to help the LLM judge relevance"}
outputs:
  verdict:        {type: string, description: "supported | contradicted | unmentioned | inconclusive"}
  evidence:       {type: list,   description: "[{url, quote, side: supports|contradicts}]"}
  confidence:     {type: float,  description: "0.0–1.0 — caller's own threshold decides accept/reject"}
tools_required: [WebFetch]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: fetch_sources
    foreach_parallel: sources
    foreach_max_concurrent: 3
    tool: WebFetch
    args:
      url: "{{item}}"
      prompt: "Find any passage that mentions or relates to this claim: {{claim}}"
      timeout: 15
    capture: source_bodies
  - id: judge
    tool: judge_claim_against_bodies
    args:
      claim: "{{claim}}"
      bodies: "{{source_bodies}}"
      context: "{{context}}"
    capture: judgement
  - id: pluck_verdict
    tool: identity
    args:
      value: "{{judgement.verdict}}"
    capture: verdict
  - id: pluck_evidence
    tool: identity
    args:
      value: "{{judgement.evidence}}"
    capture: evidence
  - id: pluck_confidence
    tool: identity
    args:
      value: "{{judgement.confidence}}"
    capture: confidence
failure_modes:
  - condition: "All sources fail to fetch"
    handler: "emit_no_sources"
  - condition: "Judge returns malformed JSON"
    handler: retry
    max_retries: 1
---

# claim_verify_against_source

Promotes a soft "the paper says X" into a citable "{paper}#p7 says X verbatim".

**Quality bar**: don't accept `verdict: supported` with `confidence < 0.6`
without a direct-quote snippet in `evidence`. Hallucinated quotes are
worse than no verification.
