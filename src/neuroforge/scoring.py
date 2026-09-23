"""
Детерминированный скоринг выживших кандидатов и выбор top-k.

Два свойства, на которых держится всё остальное:

1. **Детерминизм.** Ни одна фича не зависит от случайности, времени или
   состояния сети. Сортировка — по убыванию округлённого балла с tie-break
   по id, поэтому равные баллы дают стабильный порядок, а не произвольный.
2. **Перенормировка.** Фича, неприменимая к запросу (нет брифа — нет
   семантики; не указана длительность — нечего сравнивать), возвращает None
   и исключается, а её вес распределяется между оставшимися. Иначе запрос без
   брифа получал бы баллы в другой шкале, и пороги/сравнения поехали бы.

Набор фич — реестр, как и у фильтров: добавление фичи не требует правки
пайплайна, нужна лишь одноимённая настройка веса в config.ScoringWeights.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from neuroforge.config import settings
from neuroforge.embeddings import EmbeddingProvider, cosine_similarity
from neuroforge.schemas import Profile, Query

# Разряд округления балла перед сортировкой. Защищает порядок от дрожания
# float в последних знаках (разные сборки numpy/BLAS могут дать 1e-17).
SCORE_PRECISION = 6


@dataclass(frozen=True)
class ScoringContext:
    """Всё, что может понадобиться фиче. Расширяется вместе с реестром."""

    profile: Profile
    query: Query
    semantic_similarity: float | None
    language_universe_size: int


def semantic_similarity(ctx: ScoringContext) -> float | None:
    """Близость описания к свободному брифу заказчика.

    Неприменима без брифа: структурные поля у всех выживших кандидатов
    совпадают по построению, сравнивать описания больше не с чем.
    """
    return ctx.semantic_similarity


def budget_headroom(ctx: ScoringContext) -> float | None:
    """Доля бюджета, остающаяся сверх нижней цены подрядчика.

    Вес намеренно мал: price_from_kzt — цена «от», и дешевле не значит лучше.
    Запас по бюджету ценен как факт для объяснения, а не как преимущество.
    """
    if ctx.query.budget_kzt <= 0:
        return None
    headroom = (ctx.query.budget_kzt - ctx.profile.price_from_kzt) / ctx.query.budget_kzt
    return float(np.clip(headroom, 0.0, 1.0))


def duration_margin(ctx: ScoringContext) -> float | None:
    """Запас по времени сверх запрошенного.

    max_hours = null означает «работа не привязана к присутствию» (флорист,
    декоратор) — максимальная гибкость, а не отсутствие данных.
    """
    if ctx.query.duration_h is None or ctx.query.duration_h <= 0:
        return None
    if ctx.profile.max_hours is None:
        return 1.0
    margin = (ctx.profile.max_hours - ctx.query.duration_h) / ctx.query.duration_h
    return float(np.clip(margin, 0.0, 1.0))


def language_breadth(ctx: ScoringContext) -> float | None:
    """Языковая универсальность: при смешанной аудитории подрядчик, который
    кроме запрошенного языка владеет другими, покрывает больше гостей.

    Нормируется на число языков в датасете, а не на константу, — иначе
    добавление языка в данные потребовало бы правки кода.
    """
    if ctx.query.language is None or ctx.language_universe_size == 0:
        return None
    return len(ctx.profile.languages) / ctx.language_universe_size


@dataclass(frozen=True)
class ScoringFeature:
    name: str
    """Совпадает с именем поля в config.ScoringWeights."""

    compute: Callable[[ScoringContext], float | None]


SCORING_FEATURES: tuple[ScoringFeature, ...] = (
    ScoringFeature("semantic_similarity", semantic_similarity),
    ScoringFeature("budget_headroom", budget_headroom),
    ScoringFeature("duration_margin", duration_margin),
    ScoringFeature("language_breadth", language_breadth),
)


@dataclass(frozen=True)
class ScoredCandidate:
    profile: Profile
    score: float
    feature_values: dict[str, float]
    """Вклад каждой применённой фичи — для трассировки и для объяснений."""


def compute_semantic_similarities(
    profiles: list[Profile],
    query: Query,
    embedder: EmbeddingProvider,
    description_index: dict[str, np.ndarray],
) -> dict[str, float]:
    """Близость брифа к описанию каждого кандидата. Без брифа — пусто."""
    if not query.brief or not query.brief.strip():
        return {}

    brief_vector = embedder.encode([query.brief])[0]
    return {
        profile.id: cosine_similarity(brief_vector, description_index[profile.id])
        for profile in profiles
        if profile.id in description_index
    }


def score_candidate(ctx: ScoringContext) -> ScoredCandidate:
    weights = settings.scoring
    active: dict[str, tuple[float, float]] = {}

    for feature in SCORING_FEATURES:
        value = feature.compute(ctx)
        if value is None:
            continue
        active[feature.name] = (getattr(weights, feature.name), value)

    total_weight = sum(weight for weight, _ in active.values())
    if total_weight == 0:
        score = 0.0
    else:
        score = sum(weight * value for weight, value in active.values()) / total_weight

    return ScoredCandidate(
        profile=ctx.profile,
        score=score,
        feature_values={name: value for name, (_, value) in active.items()},
    )


def rank_candidates(
    survivors: list[Profile],
    query: Query,
    semantic_similarities: dict[str, float],
    language_universe_size: int,
    top_k: int | None = None,
) -> list[ScoredCandidate]:
    """Сортирует кандидатов детерминированно и возвращает top-k.

    Tie-break по id гарантирует, что одинаковые баллы не дают плавающего
    порядка между запусками — требование ТЗ о воспроизводимости выдачи.
    """
    limit = settings.top_k if top_k is None else top_k

    scored = [
        score_candidate(
            ScoringContext(
                profile=profile,
                query=query,
                semantic_similarity=semantic_similarities.get(profile.id),
                language_universe_size=language_universe_size,
            )
        )
        for profile in survivors
    ]

    scored.sort(key=lambda c: (-round(c.score, SCORE_PRECISION), c.profile.id))
    return scored[:limit]
