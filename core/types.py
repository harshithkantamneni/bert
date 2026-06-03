"""Shared dataclasses + enums + type aliases for bert's harness.

Every other core/ module imports from here. Pure types, no logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

# ── Enums ─────────────────────────────────────────────────────────────


class ExitReason(StrEnum):
    """First line of state/session_exit.md. Drives runner dispatch."""
    GRACEFUL_CHECKPOINT = "GRACEFUL_CHECKPOINT"
    CONTEXT_FULL = "CONTEXT_FULL"
    RATE_LIMIT = "RATE_LIMIT"
    VICTORY = "VICTORY"
    CATASTROPHIC = "CATASTROPHIC"
    PIVOT = "PIVOT"


class PermissionMode(StrEnum):
    """Permission spectrum per P-005. Determines harness gate behavior."""
    PLAN = "plan"           # read-only; bert can examine, not modify
    DEFAULT = "default"     # ask the human PI before mutations
    AUTO = "auto"           # auto-approve safe ops; ask for irreversible
    DONT_ASK = "dontAsk"    # full autonomy (only via explicit PI directive)


class DispatchAltitude(StrEnum):
    """Altitude of work for sub-agent dispatch."""
    META = "META"               # org-shaping, role design
    SPEC = "SPEC"               # design + planning
    IMPL = "IMPL"               # build
    INFRA = "INFRA"             # lab tooling
    NIT_CLEANUP = "NIT-cleanup" # small fix


class Verdict(StrEnum):
    """ResultPacket verdict. Matches schemas/result_packet.json."""
    APPROVE = "APPROVE"
    APPROVE_WITH_CAVEATS = "APPROVE_WITH_CAVEATS"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    REJECT = "REJECT"
    BUILD_PASS = "BUILD_PASS"
    BUILD_FAIL = "BUILD_FAIL"
    BUILD_PARTIAL = "BUILD_PARTIAL"
    SCOPE_STOP = "SCOPE_STOP"
    OTHER = "OTHER"


class SessionType(StrEnum):
    """brief_assembler classification of cycle context."""
    ROUTINE_MONITOR = "routine-monitor"
    PHASE_TRANSITION = "phase-transition"
    USER_ACTION = "user-action"
    POST_FAILURE = "post-failure"
    COLD_START = "cold-start"  # cycle 1, first run, empty state


class TripwireType(StrEnum):
    """Conditions that interrupt the daily-glance UX with a Telegram ping."""
    PHASE_TRANSITION = "PHASE_TRANSITION"
    CONFIDENCE_DROP = "CONFIDENCE_DROP"
    SPEND_BUDGET_HIT = "SPEND_BUDGET_HIT"
    HOLDING_LOOP = "HOLDING_LOOP"
    EVALUATOR_FAIL_3X = "EVALUATOR_FAIL_3X"
    SIGNATURE_FORGERY = "SIGNATURE_FORGERY"
    RATE_LIMIT_ALL = "RATE_LIMIT_ALL"
    CATASTROPHIC = "CATASTROPHIC"
    DESTRUCTIVE_OP_GATED = "DESTRUCTIVE_OP_GATED"
    IDENTICAL_CALL_LOOP = "IDENTICAL_CALL_LOOP"
    MISSION_PIVOT = "MISSION_PIVOT"
    CANDIDATE_PROPOSED = "CANDIDATE_PROPOSED"


class HeuristicStatus(StrEnum):
    """Knowledge-lifecycle status per Layer 8."""
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    ACCEPTED = "ACCEPTED"
    STABILIZED = "STABILIZED"
    ARCHIVED = "ARCHIVED"
    KILLED = "KILLED"


class SandboxTier(StrEnum):
    """Trust tier for tool execution."""
    TRUSTED = "trusted"               # subprocess + timeout (lab tools only)
    DOCKER = "docker"                 # --network=none --memory=512m --cpus=1 --rm
    SANDBOX_EXEC = "sandbox-exec"     # macOS sandbox-exec profile (browser/web)


# ── Core message types ───────────────────────────────────────────────


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of executing a ToolCall.

    Per P-016: content is wrapped with `<<TOOL_OUTPUT untrusted>...</TOOL_OUTPUT>>`
    sentinel before injection back into model context.
    """
    tool_call_id: str
    content: str
    error: str | None = None  # set if execution failed
    truncated: bool = False   # True if Snip shaper truncated content
    elapsed_ms: int = 0


@dataclass
class AgentMessage:
    """A message in the agent loop's messages array.

    Roles match OpenAI-compatible APIs: system, user, assistant, tool.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant role only
    tool_call_id: str | None = None  # tool role only — links to ToolCall
    name: str | None = None           # tool role only — tool name


@dataclass
class ProviderResponse:
    """Normalized response from any provider client."""
    text: str | None
    tool_calls: list[ToolCall]
    finish_reason: Literal["stop", "tool_use", "length", "content_filter", "error"]
    usage_prompt_tokens: int
    usage_completion_tokens: int
    usage_thinking_tokens: int = 0  # Gemini 2.5 / DeepSeek R1 reasoning models
    # Cached prompt tokens (cache observability).
    # Only populated by providers that surface cache metadata in their
    # OpenAI-compatible response: Gemini 2.5+ implicit caching reports
    # `usage.prompt_tokens_details.cached_content_token_count`; Groq GPT-OSS
    # automatic caching reports `usage.prompt_tokens_details.cached_tokens`.
    # 0 for providers without cache (Cerebras / Mistral / NVIDIA NIM no-op,
    # Ollama free TTFT but no $-tracked metric, Anthropic/OpenAI out of scope).
    usage_cached_tokens: int = 0
    model: str = ""
    provider: str = ""
    served_by: str | None = None   # for HF Router :fastest auto-route
    elapsed_ms: int = 0


# ── Quota tracking ────────────────────────────────────────────────────


@dataclass
class ProviderQuotaState:
    """Per-provider rate-limit accounting maintained by core.quota."""
    provider: str
    rpm_limit: int
    rpm_remaining: int
    tpm_limit: int
    tpm_remaining: int
    daily_token_cap: int | None
    daily_tokens_used: int = 0
    last_429_at: datetime | None = None
    upstream_queue_429s: int = 0  # Cerebras-specific
    health_status: Literal["healthy", "rate_limited", "down", "unknown"] = "unknown"
    next_probe_at: datetime | None = None


# ── Dispatch + result schemas (mirror schemas/*.json) ────────────────


@dataclass
class DispatchSpec:
    """Director-to-Specialist dispatch packet. Validated against
    schemas/dispatch_spec.json before any sub-agent launch."""
    dispatch_altitude: DispatchAltitude
    role: str
    cycle: int
    task: str                            # >50 chars
    success_criterion: str               # >20 chars, outside-observer-checkable
    output_path: str                     # agents/{role}/output_C{cycle}.md or findings/...
    model: str                           # provider/model
    process_hygiene: str                 # >20 chars
    confidence_required: bool = True
    caveats_embedded: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    parallel_wave: int = 1               # 1-5
    depends_on: list[str] = field(default_factory=list)
    falsifier_text: str = ""             # >30 chars when downstream-relevant


@dataclass
class FindingsCount:
    high: int = 0
    med: int = 0
    low: int = 0
    nit: int = 0


@dataclass
class Telemetry:
    tokens_in: int
    tokens_out: int
    latency_secs: float
    model_used: str
    provider: str = ""
    retry_count: int = 0
    fallback_chain: list[str] = field(default_factory=list)


@dataclass
class Falsifier:
    id: str             # stable falsifier id, e.g. {role}-C{N}-{nn}
    text: str           # >30 chars


@dataclass
class StreakExtension:
    name: str
    count: int
    falsifier_text: str


@dataclass
class ResultPacket:
    """Specialist-to-Director return packet. Validated against
    schemas/result_packet.json on every sub-agent return."""
    role: str
    cycle: int
    verdict: Verdict
    findings_count: FindingsCount
    confidence_1to10: int                          # 1-10
    calibration_reasoning: str                     # >80 chars
    telemetry: Telemetry
    falsifiers_pre_registered: list[Falsifier] = field(default_factory=list)
    caveats_blocking_downstream: list[str] = field(default_factory=list)
    streak_extension: StreakExtension | None = None


# ── Memory / retrieval types ──────────────────────────────────────────


@dataclass
class Hit:
    """One result from memory_search."""
    path: str
    chunk_text: str
    distance: float            # cosine, lower = more similar
    source_type: Literal["vector", "graph", "kv"]
    role: str | None = None
    deliverable_type: str | None = None


@dataclass
class RetrievedContext:
    """Hybrid retrieval result — vector + graph + key-value merged."""
    query: str
    vector_hits: list[Hit] = field(default_factory=list)
    graph_paths: list[dict[str, Any]] = field(default_factory=list)
    kv_hits: list[Hit] = field(default_factory=list)
    merged_ranked: list[Hit] = field(default_factory=list)


# ── Heuristic + decision records ──────────────────────────────────────


@dataclass
class Heuristic:
    """An entry in memories/heuristics.md."""
    id: str                          # H-C{N}-{nn}
    text: str
    status: HeuristicStatus
    created_cycle: int
    source: str
    related: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)


@dataclass
class Decision:
    """An entry in memories/log.md, format `## D-N (timestamp) – text`."""
    id: str                          # D-N
    cycle: int
    timestamp_utc: datetime
    text: str
    reasoning: str                   # ≥80 chars
    confidence: float                # [0, 1]
    role: str
    evidence_paths: list[str] = field(default_factory=list)


@dataclass
class KilledIdea:
    """An entry in memories/killed.md."""
    id: str                          # KID-NN
    date_killed: datetime
    killed_by: str                   # role / cycle
    what_was_tried: str
    why_failed: str                  # evidence-backed
    cost: str                        # time / tokens / cycles
    lesson: str                      # generalizable takeaway
    cross_refs: list[str] = field(default_factory=list)


# ── Permission gate + tool registry ──────────────────────────────────


@dataclass
class ToolDescriptor:
    """A single registered tool. tool_registry assembles these for the model."""
    name: str
    description: str
    parameters_schema: dict[str, Any]   # JSON Schema
    handler: Any                        # Callable; not typed strictly to avoid circular import
    permission_mode: PermissionMode = PermissionMode.AUTO
    sandbox_tier: SandboxTier = SandboxTier.TRUSTED
    is_destructive: bool = False        # P-011 hard gate regardless of mode
    source: Literal["builtin", "skill", "mcp", "creator"] = "builtin"


@dataclass
class PermissionDecision:
    """Output of permission_gate evaluation for a single tool call."""
    allowed: bool
    reason: str
    requires_telegram_approval: bool = False
    is_destructive: bool = False


# ── Session-level state ───────────────────────────────────────────────


@dataclass
class CycleContext:
    """Everything the agent loop needs to start a cycle.

    Assembled from credentials + state files + brief_assembler output.
    """
    cycle: int
    role: Literal["director", "researcher", "implementer", "evaluator",
                  "reflector", "consolidator", "strategist", "general"]
    task: str | None
    session_type: SessionType
    permission_mode: PermissionMode
    system_prompt: str               # constitutional + role + state refs
    initial_messages: list[AgentMessage]
    tool_registry_snapshot: list[ToolDescriptor]
    quota_state: dict[str, ProviderQuotaState]
    spend_budget_remaining: dict[Literal["mission", "daily"], int]
    pi_notes_mtime: float           # for fast-poll detection of new nudges


@dataclass
class CycleOutcome:
    """What the agent loop returns when the cycle exits."""
    cycle: int
    exit_reason: ExitReason
    tokens_used: int
    duration_secs: float
    decisions_logged: list[Decision]
    sub_agents_dispatched: int
    evaluator_verdict: Verdict | None
    tripwires_fired: list[TripwireType]


__all__ = [
    # Enums
    "ExitReason", "PermissionMode", "DispatchAltitude", "Verdict", "SessionType",
    "TripwireType", "HeuristicStatus", "SandboxTier",
    # Messages + tool calls
    "ToolCall", "ToolResult", "AgentMessage", "ProviderResponse",
    # Quota
    "ProviderQuotaState",
    # Dispatch + result
    "DispatchSpec", "FindingsCount", "Telemetry", "Falsifier", "StreakExtension",
    "ResultPacket",
    # Retrieval
    "Hit", "RetrievedContext",
    # Memory records
    "Heuristic", "Decision", "KilledIdea",
    # Permission + registry
    "ToolDescriptor", "PermissionDecision",
    # Session
    "CycleContext", "CycleOutcome",
]
