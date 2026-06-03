import datetime as dt
import re
import shlex


def format_event(event_type: str, **kwargs) -> str:
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [f"[{ts}]", event_type]
    for k, v in kwargs.items():
        v_str = str(v)
        if " " in v_str or "=" in v_str:
            parts.append(f'{k}="{v_str}"')
        else:
            parts.append(f"{k}={v_str}")
    return " ".join(parts)


_LINE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<event>\S+)\s*(?P<rest>.*)$")


def parse_event(line: str) -> dict:
    m = _LINE_RE.match(line.strip())
    if not m:
        raise ValueError(f"unparseable event line: {line!r}")
    out = {"timestamp": m.group("ts"), "event": m.group("event")}
    for token in shlex.split(m.group("rest")):
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def append(journal_path, event_type: str, **kwargs) -> None:
    line = format_event(event_type, **kwargs)
    with open(journal_path, "a") as f:
        f.write(line + "\n")
