"""
Объяснения к карточкам.

ExplanationFacts содержит сведения профиля и проверенные условия запроса.
Цитата и цена обязательны; LLM может выбрать дополнительные акценты только
по идентификаторам из разрешённого меню. Свободный ответ модели не выводится.
При ошибке используется детерминированный порядок тех же фактов.
"""
import json
import re
from dataclasses import dataclass
from datetime import date

import numpy as np

from neuroforge.embeddings import EmbeddingProvider, cosine_similarity
from neuroforge.schemas import Profile, Query

_MONTHS_GENITIVE = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\n•]+")


def format_date(value: date) -> str:
    return f"{value.day} {_MONTHS_GENITIVE[value.month - 1]}"


def format_kzt(value: int) -> str:
    return f"{value:,}".replace(",", " ") + " ₸"


def plural(n: int, one: str, few: str, many: str) -> str:
    """Русское согласование числительных: 1 профиль, 2 профиля, 5 профилей."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


@dataclass(frozen=True)
class ExplanationFacts:
    """Всё, на что объяснение имеет право ссылаться. Ничего сверх этого."""

    candidate_id: str
    name: str
    category: str
    city: str
    price_from_kzt: int
    budget_kzt: int
    budget_headroom_pct: int
    event_date: date
    duration_h: float | None
    max_hours: float | None
    languages: list[str]
    requested_language: str | None
    matched_sentence: str | None
    """Предложение из описания, ближайшее по смыслу к брифу заказчика."""

    busy_in_pool: int
    """Сколько подрядчиков той же категории заняты на эту дату."""

    pool_size: int
    is_synthetic: bool
    event_type: str = ""
    brief: str | None = None


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def find_matching_sentence(
    description: str,
    brief: str | None,
    embedder: EmbeddingProvider,
    other_descriptions: list[str] | None = None,
) -> str | None:
    """Цитата, отличающая профиль от соседних; с брифом — ближайшая по смыслу.

    Без брифа показываем характеристику профиля, не утверждая совпадение.
    """
    sentences = split_sentences(description)
    if not sentences:
        return None

    peers = [d.casefold() for d in other_descriptions or []]
    distinctive = [s for s in sentences if not any(s.casefold() in d for d in peers)]
    sentences = distinctive or sentences
    concise = [s for s in sentences if len(s) <= 350]
    sentences = concise or sentences
    if not brief or not brief.strip():
        return sentences[0]

    vectors = embedder.encode([brief] + sentences)
    brief_vector, sentence_vectors = vectors[0], vectors[1:]

    scores = [cosine_similarity(brief_vector, v) for v in sentence_vectors]
    return sentences[int(np.argmax(scores))]


def build_facts(
    profile: Profile,
    query: Query,
    busy_in_pool: int,
    pool_size: int,
    matched_sentence: str | None,
) -> ExplanationFacts:
    headroom_pct = 0
    if query.budget_kzt > 0:
        headroom_pct = round(
            (query.budget_kzt - profile.price_from_kzt) / query.budget_kzt * 100
        )

    return ExplanationFacts(
        candidate_id=profile.id,
        name=profile.anon_name,
        category=query.category,
        city=profile.city,
        price_from_kzt=profile.price_from_kzt,
        budget_kzt=query.budget_kzt,
        budget_headroom_pct=max(headroom_pct, 0),
        event_date=query.date,
        duration_h=query.duration_h,
        max_hours=profile.max_hours,
        languages=list(profile.languages),
        requested_language=query.language,
        matched_sentence=matched_sentence,
        busy_in_pool=busy_in_pool,
        pool_size=pool_size,
        is_synthetic=profile.synthetic,
        event_type=query.event_type,
        brief=query.brief,
    )


def _clause_semantic(facts: ExplanationFacts) -> str | None:
    if not facts.matched_sentence:
        return None
    sentence = facts.matched_sentence.rstrip(".")
    return f"в профиле указано: «{sentence}»"


def _clause_budget(facts: ExplanationFacts) -> str | None:
    if facts.budget_headroom_pct >= 15:
        return (
            f"цена от {format_kzt(facts.price_from_kzt)} — нижняя граница примерно на "
            f"{facts.budget_headroom_pct}% ниже бюджета"
        )
    if facts.price_from_kzt <= facts.budget_kzt:
        return f"начальная цена {format_kzt(facts.price_from_kzt)} не превышает бюджет"
    return None


def _clause_duration(facts: ExplanationFacts) -> str | None:
    if facts.duration_h is None:
        return None
    if facts.max_hours is None:
        return "работа не привязана к часам на площадке"
    if facts.max_hours > facts.duration_h:
        return f"работает до {facts.max_hours:g} ч при ваших {facts.duration_h:g} ч"
    return f"доступная длительность соответствует вашим {facts.duration_h:g} ч"


def prepositional(language: str) -> str:
    """«русский» -> «русском». Названия языков в датасете лежат в именительном
    падеже, а в тексте нужны в предложном. Правило общее для прилагательных
    на -ий/-ый, а не список конкретных языков: новый язык в данных не
    потребует правки кода."""
    if language.endswith(("ий", "ый")):
        return language[:-2] + "ом"
    return language


def _clause_language(facts: ExplanationFacts) -> str | None:
    if facts.requested_language is None:
        return None
    others = [lang for lang in facts.languages if lang != facts.requested_language]
    requested = prepositional(facts.requested_language)
    if others:
        joined = " и ".join(prepositional(lang) for lang in others)
        return f"работает на {requested}, а также на {joined}"
    return f"работает на {requested}"


def _clause_availability(facts: ExplanationFacts) -> str | None:
    if facts.busy_in_pool <= 0:
        return None
    return (
        f"свободен {format_date(facts.event_date)}, когда {facts.busy_in_pool} из "
        f"{facts.pool_size} в этой категории уже заняты"
    )


def render_explanation(facts: ExplanationFacts, fact_ids: list[str] | None = None) -> str:
    """Цитата, цена и максимум два дополнительных факта из разрешённого меню."""
    menu = explanation_options(facts)
    selected = fact_ids if fact_ids is not None else list(menu)[:2]
    clauses = [menu[key] for key in selected if key in menu]
    budget = _clause_budget(facts)
    if budget:
        clauses.insert(0, budget)
    if not clauses:
        clauses = [_clause_availability(facts) or "проходит по указанным условиям"]
    details = "; ".join(clauses)
    quote = _clause_semantic(facts)
    if quote:
        return f"{quote[0].upper() + quote[1:]}. {details[0].upper() + details[1:]}."
    return f"{details[0].upper() + details[1:]}."


def explanation_options(facts: ExplanationFacts) -> dict[str, str]:
    """LLM может выбрать только эти уже проверенные акценты."""
    options = {"language": _clause_language(facts), "duration": _clause_duration(facts)}
    if facts.event_type:
        options["format"] = f"принимает формат «{facts.event_type}»"
    return {key: value for key, value in options.items() if value}


def parse_fact_selection(text: str, facts: ExplanationFacts) -> list[str] | None:
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(result, dict) or set(result) != {"fact_ids"}:
        return None
    ids = result["fact_ids"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 2:
        return None
    if any(not isinstance(key, str) or key not in explanation_options(facts) for key in ids):
        return None
    return ids if len(ids) == len(set(ids)) else None

def facts_to_lines(facts: ExplanationFacts) -> list[str]:
    """Факты в виде строк для промпта. Ровно то же, из чего строится шаблон:
    LLM не получает ничего сверх проверенного пайплайном."""
    lines = [
        f"Цена от {format_kzt(facts.price_from_kzt)} при бюджете "
        f"{format_kzt(facts.budget_kzt)}"
    ]
    if facts.budget_headroom_pct >= 15:
        lines.append(f"Это на {facts.budget_headroom_pct}% ниже бюджета")

    if facts.requested_language:
        others = [l for l in facts.languages if l != facts.requested_language]
        if others:
            lines.append(
                f"Работает на запрошенном языке ({facts.requested_language}), "
                f"а также на: {', '.join(others)}"
            )
        else:
            lines.append(f"Работает на запрошенном языке ({facts.requested_language})")

    if facts.duration_h is not None:
        if facts.max_hours is None:
            lines.append("Работа не привязана к часам на площадке")
        else:
            lines.append(
                f"Берёт смены до {facts.max_hours} ч, запрошено {facts.duration_h} ч"
            )

    return lines


def build_llm_messages(facts: ExplanationFacts) -> tuple[str, str]:
    from neuroforge import prompts
    user = json.dumps({
        "candidate_id": facts.candidate_id,
        "request": {"city": facts.city, "category": facts.category,
                    "date": facts.event_date.isoformat(), "event_type": facts.event_type,
                    "brief": facts.brief, "budget_kzt": facts.budget_kzt,
                    "language": facts.requested_language, "duration_h": facts.duration_h},
        "profile_quote": facts.matched_sentence,
        "price_from_kzt": facts.price_from_kzt,
        "facts": explanation_options(facts),
    }, ensure_ascii=False, sort_keys=True)
    return prompts.EXPLANATION_SYSTEM, user
