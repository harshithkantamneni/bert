---
name: lab-tools
description: bert-lab core tools — memory tool (6-op), step_hash, structured journal events. Used by Director/Implementer/Evaluator roles to read and write durable state at /sandbox/.openclaw-data/lab/.
---

# Lab tools

You have three Python modules at `/sandbox/.openclaw-data/lab/tools/`:

- `memory.py` — six file ops (view, create, str_replace, insert, delete, rename)
- `step_hash.py` — compute deterministic hash for plan steps
- `event.py` — append structured events to journal.md

Always import from these. Don't reinvent.

```python
import sys
sys.path.insert(0, "/sandbox/.openclaw-data/lab/tools")
from memory import Memory
from event import append as journal_append

mem = Memory("/sandbox/.openclaw-data/lab")
mem.view("state/hot.md")
journal_append("/sandbox/.openclaw-data/lab/journal.md", "MY_EVENT", key="value")
```
