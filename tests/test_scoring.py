"""Тесты скоринга. Провайдер эмбеддингов подменён: тесты не должны зависеть
от скачивания модели и от сети."""
import hashlib
from datetime import date

import numpy as np
import pytest

from neuroforge.embeddings import build_description_index
from neuroforge.scoring import (
    compute_semantic_similarities,
    rank_candidates,
    score_candidate,
    ScoringContext,
)
from neuroforge.schemas import Profile, Query


class FakeEmbedder:
    """Детерминированные псевдовекторы из хэша текста: одинаковый текст даёт
    одинаковый вектор, разные тексты — разные. Этого достаточно, чтобы
    проверить механику, не трогая настоящую модель."""

    dim = 16

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = np.frombuffer(digest[: self.dim], dtype=np.uint8).astype(np.float64)
            vectors.append(raw / np.linalg.norm(raw))
        return np.vstack(vectors)


def make_profile(pid: str, **overrides) -> Profile:
    defaults = dict(
        id=pid,
        anon_name=f"Профиль {pid}",
        categories=["Ведущий"],
        city="Алматы",
        price_from_kzt=500_000,
        event_formats=["корпоратив"],
        languages=["русский"],
        max_hours=8,
        busy_dates=[],
        description=f"Описание профиля {pid}",
    )
    defaults.update(overrides)
    return Profile(**defaults)


def make_query(**overrides) -> Query:
    defaults = dict(
        city="Алматы",
        date=date(2026, 10, 13),
        event_type="корпоратив",
        category="Ведущий",
        budget_kzt=1_000_000,
    )
    defaults.update(overrides)
    return Query(**defaults)


def rank(profiles, query, embedder=None, language_universe_size=3, top_k=3):
    embedder = embedder or FakeEmbedder()
    index = build_description_index(profiles, embedder)
    sims = compute_semantic_similarities(profiles, query, embedder, index)
    return rank_candidates(profiles, query, sims, language_universe_size, top_k=top_k)


def test_ranking_is_deterministic_across_runs():
    profiles = [make_profile(f"HK-{i}") for i in range(10)]
    query = make_query(brief="нужен опытный ведущий")

    first = [c.profile.id for c in rank(profiles, query)]
    second = [c.profile.id for c in rank(profiles, query)]

    assert first == second


def test_identical_scores_break_ties_by_id():
    """Профили, неразличимые по всем фичам, должны упорядочиваться по id,
    а не как повезёт."""
    profiles = [make_profile(pid) for pid in ("HK-003", "HK-001", "HK-002")]
    query = make_query()  # без брифа — семантика неактивна

    ranked = rank(profiles, query)
    scores = {round(c.score, 6) for c in ranked}

    assert len(scores) == 1, "профили должны получить одинаковый балл"
    assert [c.profile.id for c in ranked] == ["HK-001", "HK-002", "HK-003"]


def test_inactive_features_are_excluded_not_zeroed():
    """Отсутствие брифа не должно штрафовать кандидата нулём за семантику —
    фича выбывает, а веса перенормируются."""
    profile = make_profile("HK-001")

    with_brief = score_candidate(
        ScoringContext(
            profile=profile,
            query=make_query(brief="что-то"),
            semantic_similarity=0.5,
            language_universe_size=3,
        )
    )
    without_brief = score_candidate(
        ScoringContext(
            profile=profile,
            query=make_query(),
            semantic_similarity=None,
            language_universe_size=3,
        )
    )

    assert "semantic_similarity" in with_brief.feature_values
    assert "semantic_similarity" not in without_brief.feature_values
    assert 0.0 <= without_brief.score <= 1.0


def test_score_stays_in_unit_range():
    profiles = [
        make_profile("HK-001", price_from_kzt=100_000, max_hours=None),
        make_profile("HK-002", price_from_kzt=990_000, max_hours=5),
    ]
    query = make_query(brief="бриф", duration_h=5, language="русский")

    for candidate in rank(profiles, query):
        assert 0.0 <= candidate.score <= 1.0


def test_duration_margin_rewards_unbounded_profiles():
    """max_hours=null — это максимальная гибкость, а не отсутствие данных."""
    bounded = make_profile("HK-001", max_hours=5)
    unbounded = make_profile("HK-002", max_hours=None)
    query = make_query(duration_h=5)

    ranked = rank([bounded, unbounded], query)

    assert ranked[0].profile.id == "HK-002"


def test_semantic_similarity_absent_without_brief():
    profiles = [make_profile("HK-001")]
    embedder = FakeEmbedder()
    index = build_description_index(profiles, embedder)

    sims = compute_semantic_similarities(profiles, make_query(), embedder, index)

    assert sims == {}


def test_top_k_limits_results():
    profiles = [make_profile(f"HK-{i:03d}") for i in range(10)]

    ranked = rank(profiles, make_query(), top_k=3)

    assert len(ranked) == 3


def test_embedding_index_is_stable_for_same_input():
    profiles = [make_profile("HK-001"), make_profile("HK-002")]
    embedder = FakeEmbedder()

    first = build_description_index(profiles, embedder)
    second = build_description_index(profiles, embedder)

    for pid in first:
        assert np.allclose(first[pid], second[pid])


def test_embedding_cache_roundtrip(tmp_path):
    """Кэш на диске должен возвращать те же векторы, а не пересчитывать
    молча что-то другое."""
    profiles = [make_profile("HK-001"), make_profile("HK-002")]
    embedder = FakeEmbedder()
    cache = tmp_path / "emb.npz"

    fresh = build_description_index(profiles, embedder, cache_path=cache, model_name="fake")
    assert cache.exists()

    cached = build_description_index(profiles, embedder, cache_path=cache, model_name="fake")

    for pid in fresh:
        assert np.allclose(fresh[pid], cached[pid])


def test_embedding_cache_invalidates_on_changed_description(tmp_path):
    """Изменили описание — кэш обязан протухнуть, иначе будем ранжировать
    по текстам, которых уже нет."""
    cache = tmp_path / "emb.npz"
    embedder = FakeEmbedder()

    original = [make_profile("HK-001", description="старое описание")]
    before = build_description_index(original, embedder, cache_path=cache, model_name="fake")

    changed = [make_profile("HK-001", description="новое описание")]
    after = build_description_index(changed, embedder, cache_path=cache, model_name="fake")

    assert not np.allclose(before["HK-001"], after["HK-001"])
