"""QualityContract — the 8-dimension weighted quality bar (spec §1.2).

A mission/feature declares how much each quality dimension matters via an
integer weight (1-5). After grading, judges produce a per-dimension score
(0-5); `weighted_score` collapses those to a single normalized 0-1 number
and `passes` compares it against `pass_threshold`.

Mission-specific weighting is the point: a research mission weights
provenance ★★★★★; a build mission weights usability ★★★★★. The contract
is carried on the LabSchema and consulted at finalization.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# The 8 canonical dimensions, in declaration order.
DIMENSIONS: tuple[str, ...] = (
    "correctness",
    "completeness",
    "provenance",
    "defensibility",
    "usability",
    "honesty",
    "reproducibility",
    "efficiency",
)

_MAX_DIMENSION_SCORE = 5


@dataclass
class QualityContract:
    """Per-dimension weights (1-5) + a pass threshold on the normalized
    weighted score."""

    correctness: int
    completeness: int
    provenance: int
    defensibility: int
    usability: int
    honesty: int
    reproducibility: int
    efficiency: int
    pass_threshold: float = 0.7

    def __post_init__(self) -> None:
        for dim in DIMENSIONS:
            w = getattr(self, dim)
            if not isinstance(w, int) or not (1 <= w <= 5):
                raise ValueError(
                    f"QualityContract weight {dim!r} must be an int in 1..5, "
                    f"got {w!r}"
                )
        if not (0.0 <= self.pass_threshold <= 1.0):
            raise ValueError(
                f"pass_threshold must be in 0.0..1.0, got {self.pass_threshold!r}"
            )

    def total_weight(self) -> int:
        return sum(getattr(self, dim) for dim in DIMENSIONS)

    def weighted_score(self, dimensions: dict[str, int]) -> float:
        """Collapse per-dimension scores (0-5) to a normalized 0-1 value.

        weighted_sum(score_i * weight_i) / (total_weight * MAX_SCORE).
        A dimension absent from `dimensions` scores 0 (not an error) —
        partial grades degrade rather than crash.
        """
        weighted = 0
        for dim in DIMENSIONS:
            score = dimensions.get(dim, 0)
            weighted += score * getattr(self, dim)
        denom = self.total_weight() * _MAX_DIMENSION_SCORE
        if denom == 0:  # unreachable (weights ≥ 1) but guards div-by-zero
            return 0.0
        return weighted / denom

    def passes(self, dimensions: dict[str, int]) -> bool:
        """True when the normalized weighted score clears pass_threshold."""
        return self.weighted_score(dimensions) >= self.pass_threshold

    def to_dict(self) -> dict[str, float | int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> QualityContract:
        """Build from a dict that carries the 8 weights + optional
        pass_threshold. Extra keys are ignored so a feature's larger
        quality_contract block parses cleanly."""
        kwargs: dict = {dim: int(data[dim]) for dim in DIMENSIONS}
        if "pass_threshold" in data and data["pass_threshold"] is not None:
            kwargs["pass_threshold"] = float(data["pass_threshold"])
        return cls(**kwargs)
