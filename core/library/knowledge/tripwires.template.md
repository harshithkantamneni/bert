# Tripwires

*Conditions the monitor lab watches for. When a tripwire fires, the
director should consider it a high-priority signal — possibly worth
emitting `cycle_shape: clear-blocker` or surfacing via needs_user_input.*

## Conventions

- One H2 per tripwire (e.g., `## TW-001 — New non-transformer arch at frontier scale`)
- Body: definition (what counts as triggering?) · severity · response · source
- Inline status: `**ARMED**` / `**FIRED at C-N**` / `**DISARMED**`

---

*(No tripwires yet. Monitor labs typically scaffold a few from the mission seed.)*
