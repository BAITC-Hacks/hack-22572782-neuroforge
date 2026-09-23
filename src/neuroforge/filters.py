"""
Жёсткие фильтры — чистые функции над одним профилем.
Каждая возвращает bool; при False вызывающий код помечает причину отказа
(schemas.RejectionReason) для последующего честного объяснения в исходе NO_MATCH.
"""
from datetime import date

from neuroforge.schemas import Profile, Query


def matches_city(profile: Profile, query: Query) -> bool:
    return profile.city == query.city


def matches_category(profile: Profile, query: Query) -> bool:
    return query.category in profile.categories


def matches_format(profile: Profile, query: Query) -> bool:
    return query.event_type in profile.event_formats


def is_free_on_date(profile: Profile, query: Query) -> bool:
    return query.date not in profile.busy_dates


def fits_budget(profile: Profile, query: Query) -> bool:
    return profile.price_from_kzt <= query.budget_kzt


def fits_duration(profile: Profile, query: Query) -> bool:
    if query.duration_h is None:
        return True
    if profile.max_hours is None:
        return True  # не привязан к присутствию — ограничения по времени нет
    return profile.max_hours >= query.duration_h


def fits_language(profile: Profile, query: Query) -> bool:
    if query.language is None:
        return True
    return query.language in profile.languages
