"""Тесты LLM-слоя. Сети здесь нет: проверяется поведение вокруг провайдера —
деградация, кэш, заземление промпта на фактах."""
from datetime import date
import json
from pathlib import Path

import pytest

from neuroforge.config import LLMProvider
from neuroforge.explain import build_llm_messages, facts_to_lines
from neuroforge.llm_client import ExplanationCache, LLMClient
from neuroforge.pipeline import Recommender
from neuroforge.schemas import Query
from tests.test_explain import make_facts
from tests.test_scoring import FakeEmbedder


class FakeLLM:
    """Возвращает заранее заданный текст и считает обращения."""

    def __init__(self, text: str | None = "объяснение от модели"):
        self.text = text
        self.calls = 0

    def complete(self, system: str, user: str) -> str | None:
        self.calls += 1
        return self.text


def test_prompt_contains_only_computed_facts():
    """Заземление: в промпт не должно попадать описание подрядчика целиком —
    иначе модель пересказывает рекламу профиля вместо объяснения совпадения."""
    facts = make_facts(matched_sentence="Ведём свадьбы на двух языках")
    _, user = build_llm_messages(facts)

    assert "Ведём свадьбы на двух языках" in user
    assert str(facts.price_from_kzt // 1000) in user.replace(" ", "")


def test_prompt_marks_quote_as_distinctive():
    """Регрессия: плоский список фактов заставлял модель цепляться за бюджет
    и язык, одинаковые у всех кандидатов, и объяснения становились
    взаимозаменяемыми."""
    with_quote = build_llm_messages(make_facts(matched_sentence="Уникальная фраза"))[1]
    without_quote = build_llm_messages(make_facts(matched_sentence=None))[1]

    assert "Уникальная фраза" in with_quote
    assert json.loads(without_quote)["profile_quote"] is None


def test_client_skips_providers_without_keys(monkeypatch):
    monkeypatch.delenv("MISSING_KEY_A", raising=False)
    monkeypatch.setenv("PRESENT_KEY_B", "secret")

    client = LLMClient(
        providers=[
            LLMProvider(name="a", model="m", api_key_env="MISSING_KEY_A"),
            LLMProvider(name="b", model="m", api_key_env="PRESENT_KEY_B"),
        ]
    )
    assert [p.name for p in client.available_providers()] == ["b"]


def test_client_returns_none_when_no_provider_configured(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    client = LLMClient(providers=[LLMProvider(name="x", model="m", api_key_env="NOPE")])

    assert client.complete("system", "user") is None


def test_cache_roundtrip(tmp_path):
    cache = ExplanationCache(tmp_path / "c.sqlite")
    key = ExplanationCache.make_key("system", "user")

    assert cache.get(key) is None
    cache.put(key, "текст")
    assert cache.get(key) == "текст"


def test_cache_key_depends_on_prompt():
    """Правка промпта обязана инвалидировать кэш, иначе демо покажет тексты
    от прошлой версии инструкции."""
    a = ExplanationCache.make_key("system v1", "user")
    b = ExplanationCache.make_key("system v2", "user")
    assert a != b


@pytest.fixture
def recommender(tmp_path_factory):
    cache = tmp_path_factory.mktemp("emb") / "index.npz"
    return Recommender.bootstrap(
        embedder=FakeEmbedder(), cache_path=cache, use_llm=False
    )


def _query() -> Query:
    return Query(
        city="Алматы",
        date=date(2026, 10, 13),
        event_type="корпоратив",
        category="Ведущий",
        budget_kzt=2_000_000,
    )


def test_llm_selects_verified_facts_when_available(recommender, tmp_path):
    recommender.llm = FakeLLM('{"fact_ids": ["format"]}')
    recommender.cache = ExplanationCache(tmp_path / "e.sqlite")

    response = recommender.recommend(_query())

    assert all('принимает формат «корпоратив»' in c.explanation for c in response.cards)
    assert all(c.evidence_quote.rstrip('.') in c.explanation for c in response.cards)


def test_falls_back_to_template_when_llm_returns_nothing(recommender, tmp_path):
    """Отказ провайдера не должен оставлять карточку без объяснения."""
    recommender.llm = FakeLLM(None)
    recommender.cache = ExplanationCache(tmp_path / "e.sqlite")

    response = recommender.recommend(_query())

    for card in response.cards:
        assert card.explanation.strip()
        assert card.explanation != "текст от модели"


def test_cache_prevents_repeat_calls(recommender, tmp_path):
    """Повторный одинаковый запрос обязан дать тот же текст и не тратить
    вызовы — иначе на защите два прогона покажут разные формулировки."""
    fake = FakeLLM('{"fact_ids": ["format"]}')
    recommender.llm = fake
    recommender.cache = ExplanationCache(tmp_path / "e.sqlite")

    first = recommender.recommend(_query())
    calls_after_first = fake.calls
    second = recommender.recommend(_query())

    assert fake.calls == calls_after_first
    assert [c.explanation for c in first.cards] == [c.explanation for c in second.cards]


def test_facts_lines_never_mention_name(recommender):
    """Имя подрядчика в карточке уже есть; в фактах оно только провоцирует
    модель его повторять."""
    lines = facts_to_lines(make_facts(name="Уникальное Имя"))
    assert all("Уникальное Имя" not in line for line in lines)
