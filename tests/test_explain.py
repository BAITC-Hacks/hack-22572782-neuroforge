from datetime import date

from neuroforge.explain import (
    ExplanationFacts,
    format_date,
    format_kzt,
    plural,
    prepositional,
    render_explanation,
    split_sentences,
)


def make_facts(**overrides) -> ExplanationFacts:
    defaults = dict(
        candidate_id="HK-001",
        name="Тестовый профиль",
        category="Ведущий",
        city="Алматы",
        price_from_kzt=500_000,
        budget_kzt=1_000_000,
        budget_headroom_pct=50,
        event_date=date(2026, 11, 14),
        duration_h=5,
        max_hours=8,
        languages=["русский", "казахский"],
        requested_language="казахский",
        matched_sentence="Ведём свадьбы и корпоративы на двух языках",
        busy_in_pool=3,
        pool_size=10,
        is_synthetic=False,
    )
    defaults.update(overrides)
    return ExplanationFacts(**defaults)


def test_prepositional_case_for_languages():
    assert prepositional("русский") == "русском"
    assert prepositional("казахский") == "казахском"
    assert prepositional("английский") == "английском"


def test_prepositional_leaves_unknown_forms_untouched():
    """Правило общее, но не должно калечить слова, к которым не применимо."""
    assert prepositional("хинди") == "хинди"


def test_explanation_quotes_matched_sentence_first():
    text = render_explanation(make_facts())
    assert "Ведём свадьбы и корпоративы на двух языках" in text


def test_explanation_has_no_generic_filler():
    """ТЗ прямо запрещает фразы вроде «отличный выбор для вашего мероприятия»."""
    text = render_explanation(make_facts()).lower()
    for banned in ("отличный выбор", "идеально подойдёт", "лучший вариант"):
        assert banned not in text


def test_explanations_differ_for_different_candidates():
    """DoD: если стереть имена, карточки одного запроса нельзя перепутать."""
    first = render_explanation(
        make_facts(
            matched_sentence="Работаем с тоями и национальными обрядами",
            languages=["казахский"],
            max_hours=12,
        )
    )
    second = render_explanation(
        make_facts(
            matched_sentence="Специализируемся на европейских корпоративах",
            languages=["русский", "казахский", "английский"],
            max_hours=6,
        )
    )
    assert first != second


def test_explanation_is_deterministic():
    facts = make_facts()
    assert render_explanation(facts) == render_explanation(facts)


def test_availability_clause_is_last_resort_only():
    """Занятость одинакова для всех карточек запроса, поэтому не должна
    вытеснять факты, специфичные для кандидата."""
    rich = render_explanation(make_facts())
    assert "уже заняты" not in rich

    sparse = render_explanation(
        make_facts(
            matched_sentence=None,
            requested_language=None,
            duration_h=None,
            budget_headroom_pct=0,
            price_from_kzt=2_000_000,
            budget_kzt=1_000_000,
        )
    )
    assert "уже заняты" in sparse


def test_unbounded_hours_phrased_as_flexibility():
    text = render_explanation(make_facts(max_hours=None))
    assert "не привязана к часам" in text


def test_plural_agreement():
    assert plural(1, "профиль", "профиля", "профилей") == "профиль"
    assert plural(2, "профиль", "профиля", "профилей") == "профиля"
    assert plural(5, "профиль", "профиля", "профилей") == "профилей"
    assert plural(11, "профиль", "профиля", "профилей") == "профилей"
    assert plural(21, "профиль", "профиля", "профилей") == "профиль"


def test_date_and_money_formatting():
    assert format_date(date(2026, 11, 14)) == "14 ноября"
    assert format_kzt(1_500_000) == "1 500 000 ₸"


def test_sentence_splitting():
    assert split_sentences("Первое. Второе! Третье?") == [
        "Первое.",
        "Второе!",
        "Третье?",
    ]
