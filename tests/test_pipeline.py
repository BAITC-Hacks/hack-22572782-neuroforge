"""Сквозные тесты пайплайна на реальном датасете.

Внешних вызовов здесь нет намеренно: эмбеддер подменён фейком, LLM выключен.
Тесты проверяют механику пайплайна, а не качество модели — иначе они стали бы
медленными, платными и недетерминированными. Качество семантики и
формулировок проверяется отдельно, вживую (scripts/run_demo_queries.py).
"""
from datetime import date

import pytest

from neuroforge.pipeline import Recommender
from neuroforge.schemas import OutcomeType, Query
from tests.test_scoring import FakeEmbedder


@pytest.fixture(scope="module")
def recommender(tmp_path_factory):
    # Отдельный кэш: иначе тесты перезаписали бы настоящий индекс 768-мерных
    # векторов фейковыми 16-мерными, и следующий реальный запуск потратил бы
    # две минуты на пересчёт.
    cache = tmp_path_factory.mktemp("emb") / "index.npz"
    return Recommender.bootstrap(
        embedder=FakeEmbedder(), cache_path=cache, use_llm=False
    )


def test_found_returns_at_most_three_cards(recommender):
    response = recommender.recommend(
        Query(
            city="Алматы",
            date=date(2026, 10, 13),
            event_type="корпоратив",
            category="Ведущий",
            budget_kzt=2_000_000,
        )
    )
    assert response.outcome is OutcomeType.FOUND
    assert 1 <= len(response.cards) <= 3


def test_same_query_gives_same_order(recommender):
    query = Query(
        city="Алматы",
        date=date(2026, 10, 13),
        event_type="корпоратив",
        category="Ведущий",
        budget_kzt=2_000_000,
        brief="нужен ведущий с опытом больших мероприятий",
    )
    first = [c.id for c in recommender.recommend(query).cards]
    second = [c.id for c in recommender.recommend(query).cards]
    assert first == second


def test_two_dates_give_different_results(recommender):
    """DoD: один и тот же запрос на две даты даёт разную выдачу, и видно,
    что дело в занятости."""
    base = dict(
        city="Алматы",
        event_type="корпоратив",
        category="Ведущий",
        budget_kzt=2_000_000,
    )
    autumn = recommender.recommend(Query(date=date(2026, 10, 13), **base))
    december = recommender.recommend(Query(date=date(2026, 12, 31), **base))

    assert [c.id for c in autumn.cards] != [c.id for c in december.cards]
    busy_mentioned = (autumn.message or "") + (december.message or "")
    assert "занят" in busy_mentioned


def test_no_category_is_explained_in_words(recommender):
    response = recommender.recommend(
        Query(
            city="Астана",
            date=date(2026, 11, 14),
            event_type="корпоратив",
            category="Декоратор",
            budget_kzt=1_000_000,
        )
    )
    assert response.outcome is OutcomeType.NO_CATEGORY
    assert response.cards == []
    assert response.message
    assert "Декоратор" in response.message


def test_no_match_message_names_the_reasons(recommender):
    response = recommender.recommend(
        Query(
            city="Алматы",
            date=date(2026, 12, 31),
            event_type="корпоратив",
            category="Банкетный зал",
            budget_kzt=200_000,
        )
    )
    assert response.outcome is OutcomeType.NO_MATCH
    assert response.cards == []
    assert "заняты" in response.message
    assert "бюджет" in response.message


def test_partial_result_explains_why_fewer_than_three(recommender):
    """Требование ТЗ №4."""
    response = recommender.recommend(
        Query(
            city="Алматы",
            date=date(2026, 11, 7),
            event_type="свадьба",
            category="Флорист",
            budget_kzt=600_000,
        )
    )
    assert response.outcome is OutcomeType.FOUND
    if response.found_count < 3:
        assert response.message
        assert "меньше трёх" in response.message


def test_card_shows_requested_category_not_first_of_profile(recommender):
    """13 профилей мультикатегорийны — в карточке должна стоять запрошенная
    категория, иначе пользователь увидит не то, что искал."""
    response = recommender.recommend(
        Query(
            city="Алматы",
            date=date(2026, 10, 13),
            event_type="свадьба",
            category="Банкетный зал",
            budget_kzt=5_000_000,
        )
    )
    for card in response.cards:
        assert card.category == "Банкетный зал"


def test_trace_numbers_reconcile(recommender):
    response = recommender.recommend(
        Query(
            city="Алматы",
            date=date(2026, 11, 14),
            event_type="корпоратив",
            category="Фотограф",
            budget_kzt=700_000,
        )
    )
    rejected = sum(response.trace.rejection_breakdown.values())
    survivors = response.pool_size - rejected
    assert survivors >= response.found_count


def test_every_card_has_non_empty_explanation(recommender):
    response = recommender.recommend(
        Query(
            city="Алматы",
            date=date(2026, 10, 13),
            event_type="корпоратив",
            category="Ведущий",
            budget_kzt=2_000_000,
        )
    )
    for card in response.cards:
        assert card.explanation.strip()
