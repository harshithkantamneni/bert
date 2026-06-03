import hashlib


def compute_step_hash(plan_id: str, step_id: str, description: str) -> str:
    payload = f"{plan_id}|{step_id}|{description}".encode()
    # SHA1 as deterministic step id; not for security (B324 muted via flag).
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:12]
