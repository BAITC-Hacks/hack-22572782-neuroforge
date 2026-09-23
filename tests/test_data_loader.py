from neuroforge.data_loader import (
    known_categories,
    known_cities,
    known_event_formats,
    known_languages,
    load_profiles,
)


def test_loads_all_real_profiles():
    profiles = load_profiles()
    assert len(profiles) == 66


def test_synthetic_flag_matches_dataset_expectation():
    profiles = load_profiles()
    synthetic_count = sum(1 for p in profiles if p.synthetic)
    assert synthetic_count == 13


def test_dense_categories_match_tz():
    profiles = load_profiles()
    cats = known_categories(profiles)
    assert "Ведущий" in cats
    assert "Фотограф" in cats
    assert "Банкетный зал" in cats

    def count(cat: str) -> int:
        return sum(1 for p in profiles if cat in p.categories)

    assert count("Ведущий") == 15
    assert count("Фотограф") == 12
    assert count("Банкетный зал") == 8


def test_rare_categories_have_three_profiles():
    profiles = load_profiles()

    def count(cat: str) -> int:
        return sum(1 for p in profiles if cat in p.categories)

    rare = [
        "Флорист",
        "Декоратор",
        "Подарки и сувениры",
        "Ведущий церемонии",
        "Фото и видеобудки",
        "Отель",
        "Инструменталист",
    ]
    for cat in rare:
        assert count(cat) == 3, f"{cat} should have 3 profiles"


def test_astana_has_no_decorator_natural_empty_case():
    """Естественный (не синтетический) пример для демо-сценария 'без результата':
    в Астане категории «Декоратор» нет вообще — outcome NO_CATEGORY."""
    profiles = load_profiles()
    astana_decorators = [
        p for p in profiles if p.city == "Астана" and "Декоратор" in p.categories
    ]
    assert astana_decorators == []


def test_known_value_sets_are_derived_not_hardcoded():
    profiles = load_profiles()
    cities = known_cities(profiles)
    formats = known_event_formats(profiles)
    languages = known_languages(profiles)

    assert cities == {"Алматы", "Астана", "Зарубежье"}
    assert "свадьба" in formats
    assert "русский" in languages
