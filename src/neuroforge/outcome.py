"""
Прогон воронки фильтров и классификация исхода запроса.

Три исхода из требования ТЗ №6 различаются здесь и только здесь — ниже по
пайплайну (scoring, explain) попадают уже только выжившие кандидаты.
"""
from collections import Counter
from dataclasses import dataclass, field

from neuroforge.filters import CANDIDATE_FILTERS, select_pool
from neuroforge.schemas import OutcomeType, PipelineTrace, Profile, Query, RejectionReason


@dataclass
class FunnelResult:
    outcome: OutcomeType
    pool: list[Profile]
    survivors: list[Profile]
    rejections: dict[str, RejectionReason] = field(default_factory=dict)
    """id профиля -> основная (первая сработавшая) причина отсева."""

    stage_counts: dict[str, int] = field(default_factory=dict)
    """Причина фильтра -> сколько кандидатов осталось после него."""

    @property
    def rejection_breakdown(self) -> dict[str, int]:
        counts = Counter(reason.value for reason in self.rejections.values())
        return dict(counts)

    def to_trace(self, top_scores: list[float] | None = None) -> PipelineTrace:
        return PipelineTrace(
            pool_size=len(self.pool),
            stage_counts=self.stage_counts,
            rejection_breakdown=self.rejection_breakdown,
            top_scores=top_scores or [],
        )


def run_funnel(profiles: list[Profile], query: Query) -> FunnelResult:
    """Прогоняет каталог через CANDIDATE_FILTERS по порядку, запоминая на
    каждом шаге, сколько осталось и кто из-за чего выбыл."""
    pool = select_pool(profiles, query)

    if not pool:
        return FunnelResult(outcome=OutcomeType.NO_CATEGORY, pool=[], survivors=[])

    remaining = list(pool)
    rejections: dict[str, RejectionReason] = {}
    stage_counts: dict[str, int] = {}

    for spec in CANDIDATE_FILTERS:
        passed: list[Profile] = []
        for profile in remaining:
            if spec.predicate(profile, query):
                passed.append(profile)
            else:
                rejections[profile.id] = spec.reason
        remaining = passed
        stage_counts[spec.reason.value] = len(remaining)

    outcome = OutcomeType.FOUND if remaining else OutcomeType.NO_MATCH

    return FunnelResult(
        outcome=outcome,
        pool=pool,
        survivors=remaining,
        rejections=rejections,
        stage_counts=stage_counts,
    )
