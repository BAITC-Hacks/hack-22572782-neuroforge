"""
Жёсткие фильтры — чистые функции над одним профилем.

Фильтры разделены на два уровня, и это разделение несёт смысл для исхода:

* POOL-фильтры (город, категория) определяют, существует ли в принципе
  такой каталог. Если после них пусто — исход NO_CATEGORY («в этом городе
  такой категории нет»), а не «никто не подошёл».
* CANDIDATE-фильтры (занятость, бюджет, формат, длительность, язык)
  отсеивают внутри существующего каталога. Если после них пусто — исход
  NO_MATCH, и мы обязаны сказать, по каким именно причинам отсеялись.

Набор CANDIDATE-фильтров задан реестром: добавление нового фильтра не
требует правки outcome.py, pipeline.py или схемы трассировки.
"""
from collections.abc import Callable
from dataclasses import dataclass

from neuroforge.schemas import Profile, Query, RejectionReason


def matches_city(profile: Profile, query: Query) -> bool:
    return profile.city == query.city


def matches_category(profile: Profile, query: Query) -> bool:
    return query.category in profile.categories


def matches_format(profile: Profile, query: Query) -> bool:
    return query.event_type in profile.event_formats


def is_free_on_date(profile: Profile, query: Query) -> bool:
    return query.date not in profile.busy_dates


def fits_budget(profile: Profile, query: Query) -> bool:
    """price_from_kzt — цена «от», то есть нижняя граница. Если она уже выше
    бюджета, подрядчик не тянет заказ ни при каком раскладе."""
    return profile.price_from_kzt <= query.budget_kzt


def fits_duration(profile: Profile, query: Query) -> bool:
    if query.duration_h is None:
        return True
    if profile.max_hours is None:
        return True  # null = работа не привязана к присутствию (флорист, декоратор)
    return profile.max_hours >= query.duration_h


def fits_language(profile: Profile, query: Query) -> bool:
    if query.language is None:
        return True
    return query.language in profile.languages


@dataclass(frozen=True)
class FilterSpec:
    reason: RejectionReason
    predicate: Callable[[Profile, Query], bool]


# Порядок важен дважды: он задаёт воронку в трассировке и определяет, какая
# причина считается основной, если кандидат не проходит сразу по нескольким.
# Порядок соответствует формулировке ТЗ: «заняты на дату, не тянут бюджет,
# не берут этот формат».
CANDIDATE_FILTERS: tuple[FilterSpec, ...] = (
    FilterSpec(RejectionReason.BUSY_DATE, is_free_on_date),
    FilterSpec(RejectionReason.BUDGET, fits_budget),
    FilterSpec(RejectionReason.FORMAT, matches_format),
    FilterSpec(RejectionReason.DURATION, fits_duration),
    FilterSpec(RejectionReason.LANGUAGE, fits_language),
)


def select_pool(profiles: list[Profile], query: Query) -> list[Profile]:
    """Каталог по городу и категории — до применения остальных условий."""
    return [p for p in profiles if matches_city(p, query) and matches_category(p, query)]
