from datetime import date

import pytest

from neuroforge.data_loader import load_profiles
from neuroforge.outcome import run_funnel
from neuroforge.schemas import OutcomeType, Query, RejectionReason


@pytest.fixture(scope="module")
def profiles():
    return load_profiles()


def test_no_category_when_city_lacks_category(profiles):
    """Естественный кейс из датасета: все 3 «Декоратора» — в Алматы,
    в Астане их нет вообще."""
    query = Query(
        city="Астана",
        date=date(2026, 11, 14),
        event_type="корпоратив",
        category="Декоратор",
        budget_kzt=1_000_000,
    )
    result = run_funnel(profiles, query)

    assert result.outcome == OutcomeType.NO_CATEGORY
    assert result.pool == []
    assert result.survivors == []


def test_found_in_dense_category(profiles):
    query = Query(
        city="Алматы",
        date=date(2026, 10, 13),
        event_type="корпоратив",
        category="Ведущий",
        budget_kzt=2_000_000,
    )
    result = run_funnel(profiles, query)

    assert result.outcome == OutcomeType.FOUND
    assert len(result.pool) > 0
    assert len(result.survivors) > 0


def test_no_match_when_budget_too_low(profiles):
    """Каталог есть, но бюджет отсекает всех — это NO_MATCH, не NO_CATEGORY."""
    query = Query(
        city="Алматы",
        date=date(2026, 10, 13),
        event_type="корпоратив",
        category="Ведущий",
        budget_kzt=1,
    )
    result = run_funnel(profiles, query)

    assert result.outcome == OutcomeType.NO_MATCH
    assert len(result.pool) > 0
    assert result.survivors == []
    assert result.rejection_breakdown[RejectionReason.BUDGET.value] > 0


def test_busy_date_excludes_profile(profiles):
    """Тот же запрос на две разные даты даёт разный состав выживших —
    требование DoD про занятость."""
    base = dict(
        city="Алматы",
        event_type="корпоратив",
        category="Ведущий",
        budget_kzt=2_000_000,
    )
    free_day = run_funnel(profiles, Query(date=date(2026, 10, 13), **base))
    busy_season = run_funnel(profiles, Query(date=date(2026, 12, 31), **base))

    free_ids = {p.id for p in free_day.survivors}
    busy_ids = {p.id for p in busy_season.survivors}

    assert free_ids != busy_ids
    assert busy_season.rejection_breakdown.get(RejectionReason.BUSY_DATE.value, 0) > 0


def test_funnel_counts_are_monotonic(profiles):
    """Воронка не может расти: каждый следующий фильтр только сужает выборку."""
    query = Query(
        city="Алматы",
        date=date(2026, 11, 14),
        event_type="свадьба",
        category="Фотограф",
        budget_kzt=800_000,
        language="казахский",
    )
    result = run_funnel(profiles, query)

    counts = list(result.stage_counts.values())
    assert counts == sorted(counts, reverse=True)
    assert counts[0] <= len(result.pool)


def test_rejection_breakdown_sums_to_pool_minus_survivors(profiles):
    """Каждый отсеянный учтён ровно один раз — числа в объяснении жюри
    должны сходиться."""
    query = Query(
        city="Алматы",
        date=date(2026, 12, 20),
        event_type="корпоратив",
        category="Фотограф",
        budget_kzt=500_000,
        duration_h=6,
    )
    result = run_funnel(profiles, query)

    rejected_total = sum(result.rejection_breakdown.values())
    assert rejected_total == len(result.pool) - len(result.survivors)


def test_null_max_hours_passes_duration_filter(profiles):
    """max_hours = null означает «не привязан к присутствию» (флорист,
    декоратор) — такой профиль не должен отсекаться по длительности."""
    query = Query(
        city="Алматы",
        date=date(2026, 10, 13),
        event_type="свадьба",
        category="Флорист",
        budget_kzt=1_000_000,
        duration_h=12,
    )
    result = run_funnel(profiles, query)

    unbounded = [p for p in result.survivors if p.max_hours is None]
    assert unbounded, "профиль с max_hours=null должен пройти фильтр длительности"
