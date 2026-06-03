# Expected first cycle — Product Lab

Cycle 1 of a Product Lab MUST produce these artifacts:

| Artifact | Required | Format | Where |
|---|---|---|---|
| `plan.md` | ✓ | Markdown, 3-step plan | `cycles/001/plan.md` |
| `code/` | ✓ | actual implementation | `cycles/001/code/*` |
| `tests/` | ✓ | runnable test files | `cycles/001/tests/*` |
| Green test output | ✓ | captured stdout | `cycles/001/tests/_output.txt` |
| `proof_packet.tar.gz` | ✓ | signed `.tar.gz` | `cycles/001/proof_packet/` |
| `journal.md` | ✓ | 1-paragraph reflection | `cycles/001/journal.md` |

The smoke test `_smoketest.sh` validates:
1. All required files exist
2. Tests pass (`pytest cycles/001/tests/`)
3. Proof packet verifies (`bert verify cycles/001/proof_packet/*.tar.gz`)

If cycle 1 doesn't produce all six, the lab boots into FirstLight with
a clear "cycle 1 incomplete — run `bert run --first-cycle` to complete"
message rather than appearing successful.
