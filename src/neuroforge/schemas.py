"""
Pydantic-схемы. Поля соответствуют реальному датасету (см. docs/architecture.md),
но допустимые значения (город, категория, язык, формат) НЕ перечисляются как enum —
они валидируются относительно множества, вычисленного из загруженного датасета
(см. data_loader.py), чтобы новые значения в данных не требовали правки схемы.
"""
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Profile(BaseModel):
    id: str
    anon_name: str
    categories: list[str]
    city: str
    price_from_kzt: int = Field(ge=0)
    event_formats: list[str]
    languages: list[str]
    max_hours: Optional[float] = Field(default=None, gt=0)
    busy_dates: list[date] = Field(default_factory=list)
    description: str
    synthetic: bool = False
    city_imputed: bool = False
    price_imputed: bool = False


class Query(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    city: str = Field(min_length=1, max_length=120)
    date: date
    event_type: str = Field(min_length=1, max_length=120)
    category: str = Field(min_length=1, max_length=120)
    budget_kzt: int = Field(ge=0)
    duration_h: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    language: Optional[str] = Field(default=None, max_length=120)
    brief: Optional[str] = Field(default=None, max_length=2000)
    """Свободное описание пожеланий заказчика. Единственный вход, с которым
    семантическое сходство имеет смысл: структурные поля у всех выживших
    кандидатов совпадают по определению (они прошли одни и те же фильтры),
    поэтому сравнивать описания не с чем, кроме брифа. Если брифа нет —
    семантическая фича отключается, а её вес перераспределяется между
    остальными (см. scoring.py)."""

    @field_validator("language", "brief", mode="before")
    @classmethod
    def empty_is_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value


class OutcomeType(str, Enum):
    FOUND = "FOUND"
    NO_CATEGORY = "NO_CATEGORY"          # такой категории в городе нет вообще
    NO_MATCH = "NO_MATCH"                # кандидаты есть, но никто не прошёл фильтры


class RejectionReason(str, Enum):
    BUSY_DATE = "busy_date"
    BUDGET = "budget"
    FORMAT = "format"
    DURATION = "duration"
    LANGUAGE = "language"


class Card(BaseModel):
    id: str
    name: str
    category: str
    """Именно запрошенная категория, а не первая из profile.categories:
    13 профилей в датасете мультикатегорийны (напр. Ресторан + Банкетный зал)."""
    city: str
    price_from_kzt: int
    is_synthetic: bool
    explanation: str
    score: float
    evidence_quote: Optional[str] = None
    city_imputed: bool = False
    price_imputed: bool = False


class PipelineTrace(BaseModel):
    """Промежуточные числа по стадиям — для прозрачности перед жюри.

    Счётчики заданы словарями, а не фиксированными полями: набор фильтров
    описан реестром в filters.py, и добавление нового фильтра не должно
    требовать правки схемы.
    """

    pool_size: int
    """Сколько профилей в городе и категории запроса — до остальных фильтров."""

    stage_counts: dict[str, int] = Field(default_factory=dict)
    """Воронка: причина фильтра -> сколько кандидатов осталось после него."""

    rejection_breakdown: dict[str, int] = Field(default_factory=dict)
    """Причина -> сколько кандидатов отсеяно по ней (первая сработавшая)."""

    top_scores: list[float] = Field(default_factory=list)
    feature_values: dict[str, dict[str, float]] = Field(default_factory=dict)
    explanation_sources: dict[str, str] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    outcome: OutcomeType
    pool_size: int
    found_count: int
    eligible_count: int = 0
    message: Optional[str] = None
    """Пользовательский текст для исходов NO_CATEGORY / NO_MATCH, а также
    для случая, когда подходящих нашлось меньше трёх (требование ТЗ №4)."""
    cards: list[Card] = Field(default_factory=list)
    trace: PipelineTrace
