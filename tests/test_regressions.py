"""Проверки отказов и критериев ТЗ, найденных при сквозном ревью."""
import json
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import create_app
from neuroforge.config import settings
from neuroforge.data_loader import load_profiles
from neuroforge.embeddings import build_description_index
from neuroforge.explain import build_llm_messages, parse_fact_selection
from neuroforge.pipeline import Recommender
from neuroforge.schemas import Query
from tests.test_explain import make_facts
from tests.test_llm_layer import FakeLLM
from tests.test_scoring import FakeEmbedder, make_query, make_profile


@pytest.fixture
def service(tmp_path):
    r = Recommender.bootstrap(embedder=FakeEmbedder(), cache_path=tmp_path / 'vectors.npz', use_llm=False)
    yield r
    r.close()


def test_same_price_without_brief_still_has_distinct_explanations(service):
    response = service.recommend(make_query(date=date(2026, 9, 27), event_type='конференция', budget_kzt=5_000_000))
    assert len(response.cards) == 3
    assert len({c.explanation for c in response.cards}) == 3
    profiles = {p.id: p for p in service.profiles}
    for card in response.cards:
        assert card.evidence_quote in profiles[card.id].description
        assert card.evidence_quote.rstrip('.') in card.explanation


@pytest.mark.parametrize('day', [date(2026, 9, 22), date(2027, 1, 1)])
def test_unknown_calendar_date_is_not_treated_as_free(service, day):
    with pytest.raises(ValueError, match='Нет данных о доступности'):
        service.recommend(make_query(date=day))


@pytest.mark.parametrize('fields', [{'budget_kzt': -1}, {'duration_h': 0}, {'duration_h': -2}, {'duration_h': float('inf')}, {'city': '  '}])
def test_invalid_query_is_rejected(fields):
    with pytest.raises(ValidationError):
        make_query(**fields)


def test_query_normalization_preserves_matches(service):
    clean = service.recommend(make_query())
    normalized = service.recommend(make_query(city='  аЛмАтЫ ', category='ведущий', event_type=' КОРПОРАТИВ '))
    assert normalized == clean


def test_prompt_keeps_actual_brief_and_event_format():
    _, user = build_llm_messages(make_facts(brief='тихий семейный вечер', event_type='юбилей'))
    assert json.loads(user)['request']['brief'] == 'тихий семейный вечер'
    assert json.loads(user)['request']['event_type'] == 'юбилей'


@pytest.mark.parametrize('text', ['Работает бесплатно', '{"fact_ids": ["free_booking"]}', '{"fact_ids": ["language"], "price": 0}', '{"fact_ids": [{"injected": true}]}', '{"fact_ids": ["language", "language"]}'])
def test_llm_cannot_add_facts(text):
    assert parse_fact_selection(text, make_facts()) is None


def test_unverified_llm_text_is_never_shown(service):
    baseline = service.recommend(make_query())
    service.llm = FakeLLM('Работает бесплатно, цена 0 тенге, рейтинг 5.0!')
    actual = service.recommend(make_query())
    assert [c.explanation for c in actual.cards] == [c.explanation for c in baseline.cards]


def test_broken_cache_does_not_break_recommendation(service):
    class BrokenCache:
        def get(self, key): raise sqlite3.OperationalError('database locked')
        def put(self, key, value): raise sqlite3.OperationalError('disk full')
    service.cache = BrokenCache()
    service.llm = FakeLLM('{"fact_ids": ["format"]}')
    assert service.recommend(make_query()).cards


def test_deadline_returns_fallback_without_waiting_for_worker(service, monkeypatch):
    released = threading.Event()
    entered = threading.Event()
    class HangingLLM:
        def complete(self, system, user):
            entered.set()
            released.wait(2)
            return None
    monkeypatch.setattr(settings, 'llm_total_timeout_s', 0.05)
    service.llm = HangingLLM()
    started = time.perf_counter()
    try:
        response = service.recommend(make_query())
        assert entered.is_set()
        assert response.cards
        assert time.perf_counter() - started < 0.5
    finally:
        released.set()


def test_missing_dataset_fails_at_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'dataset_path', str(tmp_path / 'missing.jsonl'))
    with pytest.raises(FileNotFoundError, match='Датасет не найден'):
        load_profiles()


def test_corrupted_embedding_cache_is_rebuilt(tmp_path):
    path = tmp_path / 'vectors.npz'
    path.write_bytes(b'interrupted write')
    index = build_description_index([make_profile('HK-1')], FakeEmbedder(), path)
    assert len(index['HK-1']) == FakeEmbedder.dim


def test_http_endpoints_and_demo_scenarios(service):
    scenarios = json.loads(Path('data/demo_queries.json').read_text())
    with TestClient(create_app(service)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/assets/app.js').status_code == 200
        assert client.get('/health').json()['profile_count'] == 66
        assert client.get('/catalog').json()['synthetic_count'] == 13
        assert client.get('/demo-scenarios').json() == scenarios
        outputs = {}
        for scenario in scenarios:
            response = client.post('/recommend', json=scenario['query'])
            assert response.status_code == 200, response.text
            body = response.json()
            assert body['outcome'] == scenario['expected_outcome']
            assert len(body['cards']) <= 3
            assert body['pool_size'] == sum(body['trace']['rejection_breakdown'].values()) + body['eligible_count']
            if body['cards']:
                assert len({c['explanation'] for c in body['cards']}) == len(body['cards'])
            else:
                assert body['message']
            repeat = client.post('/recommend', json=scenario['query']).json()
            assert repeat == body
            outputs[scenario['id']] = body
        assert [c['id'] for c in outputs['dense']['cards']] != [c['id'] for c in outputs['date-change']['cards']]
        invalid = {**scenarios[0]['query'], 'date': '2027-01-01'}
        assert client.post('/recommend', json=invalid).status_code == 422
        invalid['budget_kzt'] = -1
        assert client.post('/recommend', json=invalid).status_code == 422
