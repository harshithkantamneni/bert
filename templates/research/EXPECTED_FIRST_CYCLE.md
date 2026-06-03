# Expected first cycle — Research Lab

Cycle 1 of a Research Lab MUST produce:

| Artifact | Required | Where |
|---|---|---|
| `question_refined.md` | ✓ | `cycles/001/question_refined.md` |
| `sources_enumerated.md` | ✓ | `cycles/001/sources_enumerated.md` |
| `synthesis.md` | ✓ | `cycles/001/synthesis.md` |
| `open_questions.md` | ✓ | `cycles/001/open_questions.md` |
| `proof_packet.tar.gz` | ✓ | `cycles/001/proof_packet/` |
| `journal.md` | ✓ | `cycles/001/journal.md` |

Smoke check: every claim in synthesis.md must have at least one source
in sources_enumerated.md. Unsourced claims trigger a `selective-disclosure`
limitation in failures.md.
