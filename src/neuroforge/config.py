"""
Централизованная конфигурация проекта.

Правило: никаких магических чисел и списков допустимых значений в коде
пайплайна (filters.py, scoring.py, explain.py и т.д.) — только здесь,
через env-переменные с дефолтами. Списки городов/категорий/языков сюда
не попадают вообще: они вычисляются динамически из датасета в data_loader.py.
"""
from datetime import date
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ключи провайдеров ищутся по имени переменной (LLMProvider.api_key_env),
# поэтому .env должен попасть именно в окружение, а не только в Settings.
load_dotenv()


class ScoringWeights(BaseSettings):
    """Веса компонентов детерминированного скоринга.

    Имя поля обязано совпадать с именем фичи в scoring.SCORING_FEATURES —
    так реестр фич остаётся декларативным. Сумма не обязана быть 1.0:
    неприменимые к запросу фичи выбывают, а веса перенормируются.
    """

    semantic_similarity: float = Field(default=0.55, ge=0)
    budget_headroom: float = Field(default=0.10, ge=0)
    """Намеренно мал: price_from_kzt — цена «от», дешевле не значит лучше.
    Запас по бюджету идёт в текст объяснения, а не в преимущество рейтинга."""

    duration_margin: float = Field(default=0.20, ge=0)
    language_breadth: float = Field(default=0.15, ge=0)


class LLMProvider(BaseModel):
    """Один провайдер в цепочке отказоустойчивости.

    OpenAI и NVIDIA NIM используют один протокол, поэтому провайдер — это
    данные (имя, модель, адрес, переменная с ключом), а не отдельный класс.
    Добавить провайдера значит дописать элемент списка, а не править код.
    """

    name: str
    model: str
    api_key_env: str
    base_url: str | None = None
    """None — адрес по умолчанию для OpenAI SDK."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEUROFORGE_", env_file=".env", extra="ignore", env_nested_delimiter="__")

    dataset_path: str = "data/raw/hackathon-dataset-anonymized.jsonl"
    synthetic_path: str = "data/synthetic/extra_profiles.jsonl"
    # Метаданные покрытия из ТЗ; отсутствие busy_dates вне окна не означает свободу.
    calendar_start: date = date(2026, 9, 23)
    calendar_end: date = date(2026, 12, 31)

    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    embeddings_cache_path: str = "data/processed/embeddings_cache.npz"

    llm_providers: list[LLMProvider] = [
        LLMProvider(
            name="openai",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
        ),
        LLMProvider(
            name="nvidia",
            model="nvidia/nemotron-3-super-120b-a12b",
            api_key_env="NVIDIA_API_KEY",
            base_url="https://integrate.api.nvidia.com/v1",
        ),
    ]

    llm_timeout_s: float = Field(default=2.0, gt=0, le=5)
    llm_total_timeout_s: float = Field(default=3.0, gt=0, le=6)
    """Общий бюджет ожидания объяснений; после него используются шаблоны."""

    llm_temperature: float = 0.0
    llm_max_tokens: int = 160
    llm_enabled: bool = True
    """Выключатель для демонстрации шаблонного режима без правки кода."""

    explanation_cache_path: str = "cache/explanations.sqlite"

    top_k: int = Field(default=3, ge=1, le=3)

    scoring: ScoringWeights = Field(default_factory=ScoringWeights)

    @model_validator(mode="after")
    def calendar_order(self):
        if self.calendar_end < self.calendar_start:
            raise ValueError("calendar_end должен быть не раньше calendar_start")
        return self


settings = Settings()
