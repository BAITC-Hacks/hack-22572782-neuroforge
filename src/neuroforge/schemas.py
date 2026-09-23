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
    brief: Optional[str] = None
    """Свободное описание пожеланий заказчика. Единственный вход, с которым
    семантическое сходство имеет смысл: структурные поля у всех выживших
    кандидатов совпадают по определению (они прошли одни и те же фильтры),
    поэтому сравнивать описания не с чем, кроме брифа. Если брифа нет —
    семантическая фича отключается, а её вес перераспределяется между
    остальными (см. scoring.py)."""


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


class RecommendResponse(BaseModel):
    outcome: OutcomeType
    pool_size: int
    found_count: int
    message: Optional[str] = None
    """Пользовательский текст для исходов NO_CATEGORY / NO_MATCH, а также
    для случая, когда подходящих нашлось меньше трёх (требование ТЗ №4)."""
    cards: list[Card] = Field(default_factory=list)
    trace: PipelineTrace
