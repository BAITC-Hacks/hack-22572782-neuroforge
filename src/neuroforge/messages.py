"""
Тексты для исходов, в которых карточек нет или их меньше трёх.

ТЗ требует, чтобы пустой результат был объяснён словами, а не пустым экраном,
и чтобы три исхода различались для пользователя явно. Поэтому сообщения
строятся из тех же чисел, что попадают в трассировку: пользователь и жюри
видят одно и то же, просто в разной форме.
"""
from neuroforge.explain import format_date, format_kzt, plural
from neuroforge.outcome import FunnelResult
from neuroforge.schemas import Query, RejectionReason

_REASON_PHRASES = {
    RejectionReason.BUSY_DATE: ("занят", "заняты", "заняты"),
    RejectionReason.BUDGET: (
        "не укладывается в бюджет",
        "не укладываются в бюджет",
        "не укладываются в бюджет",
    ),
    RejectionReason.FORMAT: (
        "не берёт этот формат",
        "не берут этот формат",
        "не берут этот формат",
    ),
    RejectionReason.DURATION: (
        "не закрывает нужную длительность",
        "не закрывают нужную длительность",
        "не закрывают нужную длительность",
    ),
    RejectionReason.LANGUAGE: (
        "не работает на нужном языке",
        "не работают на нужном языке",
        "не работают на нужном языке",
    ),
}


def _describe_rejections(funnel: FunnelResult) -> list[str]:
    """Причины отсева словами, в порядке реестра фильтров — тот же порядок,
    что в воронке, чтобы текст и трассировка читались одинаково."""
    breakdown = funnel.rejection_breakdown
    parts = []
    for reason, forms in _REASON_PHRASES.items():
        count = breakdown.get(reason.value, 0)
        if count:
            parts.append(f"{count} {plural(count, *forms)}")
    return parts


def no_category_message(query: Query) -> str:
    return (
        f"В городе {query.city} в каталоге нет категории «{query.category}». "
        f"Это не значит, что все заняты — таких подрядчиков здесь нет вовсе, "
        f"и подобрать не из чего."
    )


def no_match_message(query: Query, funnel: FunnelResult) -> str:
    pool_size = len(funnel.pool)
    reasons = _describe_rejections(funnel)
    head = (
        f"В городе {query.city} нашлось {pool_size} "
        f"{plural(pool_size, 'подрядчик', 'подрядчика', 'подрядчиков')} "
        f"категории «{query.category}», но на {format_date(query.date)} "
        f"не подошёл ни один"
    )
    if not reasons:
        return head + "."
    return f"{head}: {', '.join(reasons)}."


def partial_result_message(query: Query, funnel: FunnelResult, found: int) -> str:
    """Требование ТЗ: если подходящих меньше трёх, сказать сколько есть и
    почему меньше."""
    pool_size = len(funnel.pool)
    reasons = _describe_rejections(funnel)

    head = (
        f"Подходящих меньше трёх — {found} "
        f"{plural(found, 'вариант', 'варианта', 'вариантов')}. "
        f"Всего в категории «{query.category}» по городу {query.city} "
        f"{pool_size} {plural(pool_size, 'профиль', 'профиля', 'профилей')}"
    )
    if not reasons:
        return head + "."
    return f"{head}, из них {', '.join(reasons)}."


def availability_note(query: Query, funnel: FunnelResult) -> str | None:
    """Контекст занятости на уровне каталога, а не карточки.

    Этот факт одинаков для всех кандидатов одного запроса, поэтому в
    объяснениях он делал бы карточки взаимозаменяемыми. На уровне ответа он,
    наоборот, полезен: при смене даты меняется и состав выдачи, и эта строка,
    так что видно, что дело именно в занятости (требование DoD).
    """
    busy = funnel.rejection_breakdown.get(RejectionReason.BUSY_DATE.value, 0)
    if busy == 0:
        return None
    pool_size = len(funnel.pool)
    free = pool_size - busy
    return (
        f"На {format_date(query.date)} в категории «{query.category}» "
        f"{plural(free, 'свободен', 'свободны', 'свободны')} {free} из "
        f"{pool_size}: {busy} {plural(busy, 'занят', 'заняты', 'заняты')}."
    )


def budget_hint(query: Query, funnel: FunnelResult) -> str | None:
    """Если всех снёс бюджет — подсказать минимальную цену в каталоге.
    Это не расширение выдачи, а объяснение, насколько мимо был бюджет."""
    if funnel.rejection_breakdown.get(RejectionReason.BUDGET.value, 0) == 0:
        return None
    if not funnel.pool:
        return None
    cheapest = min(p.price_from_kzt for p in funnel.pool)
    if cheapest <= query.budget_kzt:
        return None
    return (
        f"Самое доступное предложение в этой категории начинается от "
        f"{format_kzt(cheapest)}."
    )
