"""
Централизованная конфигурация проекта.

Правило: никаких магических чисел и списков допустимых значений в коде
пайплайна (filters.py, scoring.py, explain.py и т.д.) — только здесь,
через env-переменные с дефолтами. Списки городов/категорий/языков сюда
не попадают вообще: они вычисляются динамически из датасета в data_loader.py.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoringWeights(BaseSettings):
    """Веса компонентов детерминированного скоринга.

    Имя поля обязано совпадать с именем фичи в scoring.SCORING_FEATURES —
    так реестр фич остаётся декларативным. Сумма не обязана быть 1.0:
    неприменимые к запросу фичи выбывают, а веса перенормируются.
    """

    semantic_similarity: float = 0.55
    budget_headroom: float = 0.10
    """Намеренно мал: price_from_kzt — цена «от», дешевле не значит лучше.
    Запас по бюджету идёт в текст объяснения, а не в преимущество рейтинга."""

    duration_margin: float = 0.20
    language_breadth: float = 0.15


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEUROFORGE_", env_file=".env", extra="ignore")

    dataset_path: str = "data/raw/hackathon-dataset-anonymized.jsonl"
    synthetic_path: str = "data/synthetic/extra_profiles.jsonl"

    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    embeddings_cache_path: str = "data/processed/embeddings_cache.npz"

    llm_model: str = "claude-haiku-4-5-20251001"
    llm_timeout_s: float = 4.0
    llm_temperature: float = 0.0

    explanation_cache_path: str = "cache/explanations.sqlite"

    top_k: int = 3

    scoring: ScoringWeights = ScoringWeights()


settings = Settings()
