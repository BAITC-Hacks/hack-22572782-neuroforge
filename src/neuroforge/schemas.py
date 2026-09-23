"""
Pydantic-схемы. Поля соответствуют реальному датасету (см. docs/architecture.md),
но допустимые значения (город, категория, язык, формат) НЕ перечисляются как enum —
они валидируются относительно множества, вычисленного из загруженного датасета
(см. data_loader.py), чтобы новые значения в данных не требовали правки схемы.
"""
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Profile(BaseModel):
    id: str
    anon_name: str
    categories: list[str]
    city: str
    price_from_kzt: int
    event_formats: list[str]
    languages: list[str]
    max_hours: Optional[int] = None
    busy_dates: list[date] = Field(default_factory=list)
    description: str
    synthetic: bool = False
    city_imputed: bool = False
    price_imputed: bool = False


class Query(BaseModel):
    city: str
    date: date
    event_type: str
    category: str
    budget_kzt: int
    duration_h: Optional[int] = None
    language: Optional[str] = None


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
    city: str
    price_from_kzt: int
    is_synthetic: bool
    explanation: str
    score: float


class PipelineTrace(BaseModel):
    """Промежуточные числа по стадиям — для прозрачности перед жюри."""

    pool_city_category: int
    after_busy_filter: int
    after_budget_filter: int
    after_format_filter: int
    after_duration_filter: int
    after_language_filter: int
    rejection_breakdown: dict[str, int] = Field(default_factory=dict)
    top_scores: list[float] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    outcome: OutcomeType
    pool_size: int
    found_count: int
    message: Optional[str] = None
    cards: list[Card] = Field(default_factory=list)
    trace: PipelineTrace
