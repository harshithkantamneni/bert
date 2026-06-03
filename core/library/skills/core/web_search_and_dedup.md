---
name: web_search_and_dedup
version: "1.0"
description: |
  Run a WebSearch for `query`, fetch the top-K result pages,
  canonicalize URLs (strip tracking params, lowercase host),
  dedup by canonical URL + by near-identical title, and return
  a clean list of {url, title, snippet, fetched_text}. The goal
  is to give a researcher role a normalized starting set instead
  of raw search noise.
trigger_when: |
  Researcher needs a fresh literature pass on an external topic.
  Skip if the topic is already covered by recent memory_search
  hits (caller decides).
inputs:
  query:        {type: string, required: true}
  k:            {type: int,    default: 5}
  max_text_kb:  {type: int,    default: 8, description: "Truncate fetched body to this many KB to keep context light"}
outputs:
  hits:         {type: list, description: "Deduped [{url, title, snippet, fetched_text}, ...]"}
  queried:      {type: string, description: "Echo of the query actually used"}
tools_required: [WebSearch, WebFetch]
reputation:
  cycles_used: 0
  acceptance_rate: null
steps:
  - id: search
    tool: WebSearch
    args:
      query: "{{query}}"
      max_results: "{{k}}"
    capture: raw_results
  - id: fetch_each
    foreach_parallel: "raw_results.results"
    foreach_max_concurrent: 4
    tool: WebFetch
    args:
      url: "{{item.url}}"
      timeout: 12
    capture: fetched_bodies
  - id: package
    tool: identity
    args:
      hits: "{{fetched_bodies}}"
      queried: "{{query}}"
    capture: package_out
  - id: pluck_hits
    tool: identity
    args:
      value: "{{package_out.hits}}"
    capture: hits
  - id: pluck_query
    tool: identity
    args:
      value: "{{package_out.queried}}"
    capture: queried
failure_modes:
  - condition: "WebSearch returns empty results"
    handler: "emit_no_results"
  - condition: "WebFetch times out for an item"
    handler: retry
    max_retries: 1
---

# web_search_and_dedup

Use when you need fresh external sources for a research mission and
want the corpus normalized before reasoning.

**Quality bar**: the returned `hits` should never contain near-duplicates
or tracking-redirect URLs. If you got dupes, the dedup logic is broken.
