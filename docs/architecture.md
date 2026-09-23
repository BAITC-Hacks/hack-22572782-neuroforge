# Архитектура

## Принцип
Ранжирование и фильтрация — детерминированный код. LLM используется только
для формулировки текста объяснения поверх уже посчитанных фактов. Это разом
даёт: детерминизм порядка карточек, объяснения без общих фраз и устойчивость
к сетевым сбоям (<10с ответ даже если LLM API недоступен).

## Пайплайн

```
Клиент (форма) ──POST /recommend──▶ API (FastAPI)
                                          │
                                          ▼
                              1. Query Normalizer      Pydantic: валидация, нормализация
                                          ▼
                              2. Candidate Pool         все профили city + category (real + synthetic)
                                          ▼
                              3. Hard Filters           busy_dates, budget, format, duration, language
                                 (чистые функции)
                                          ▼
                              4. Outcome Classifier  ──▶ NO_CATEGORY / NO_MATCH / FOUND
                                          │ (FOUND)
                                          ▼
                              5. Deterministic          semantic score (эмбеддинги) + soft-фичи
                                 Scorer & Top-3          сортировка + tie-break по id
                                          ▼
                              6. Explanation Agent      LLM temp=0, grounded фактами,
                                 (tool-grounded)         кэш + fallback на шаблон
                                          ▼
                              7. Response Assembler     карточки + outcome + trace для жюри
```

## Hard Filters
Все — чистые булевы функции над одним профилем (`src/neuroforge/filters.py`):

| Фильтр | Условие |
|---|---|
| city | `profile.city == query.city` |
| category | `query.category in profile.categories` |
| event_format | `query.event_type in profile.event_formats` |
| busy_dates | `query.date not in profile.busy_dates` |
| budget | `profile.price_from_kzt <= query.budget_kzt` |
| duration (опц.) | `profile.max_hours is None or profile.max_hours >= query.duration_h` |
| language (опц.) | `query.language in profile.languages` |

Каждый отфильтрованный кандидат сохраняет причину исключения — нужна для
честного объяснения в исходе NO_MATCH.

## Outcome Classifier
```
pool_city_category = filter(city, category)
if len(pool_city_category) == 0: → NO_CATEGORY
survivors = apply_remaining_filters(pool_city_category)
if len(survivors) == 0: → NO_MATCH   # + breakdown причин
else: → FOUND                        # found_count < 3 — отдельный флаг
```

## Deterministic Scorer
```
score = w.semantic_similarity * semantic_similarity(query_context, profile.description)
      + w.budget_headroom     * budget_headroom(profile.price_from_kzt, query.budget_kzt)
      + w.duration_margin     * duration_margin(profile.max_hours, query.duration_h)
      + w.language_breadth    * language_breadth(profile.languages, query.language)
```
Веса — в `config.Settings.scoring`, не в коде. Эмбеддинги считаются локально
(`sentence-transformers`, мультиязычная модель) — без сетевой зависимости на
критичном пути. Сортировка `desc(score)`, tie-break по `id` asc.

## Explanation Agent
Для каждого top-k кандидата собирается структурированный факт-объект
(budget_headroom_pct, duration_margin_h, language_match, top_semantic_sentence,
is_synthetic) и передаётся в LLM с инструкцией использовать только эти факты,
1-2 предложения, без общих фраз. temperature=0. Кэш по
`hash(candidate_id + normalized_query)`. При таймауте/ошибке — шаблонный
fallback на тех же фактах.

## Гибкость / принцип "ничего не хардкодить"
- Допустимые города/категории/форматы/языки вычисляются из датасета
  динамически (`data_loader.py`), не перечисляются в коде.
- Веса скоринга — конфиг, не константы в коде.
- Набор soft-фич скоринга — список функций, легко расширяемый.
- Промпт LLM — отдельный шаблон, не встроенная строка в логике.

## Данные
`hackathon-dataset-anonymized` — 66 профилей: id, anon_name, categories, city,
price_from_kzt, event_formats, languages, max_hours, busy_dates, description,
synthetic, city_imputed, price_imputed. 13 профилей полностью синтетические.
Можно дописывать свои синтетические профили в `data/synthetic/` тем же
форматом — в демо видно, где реальный профиль, а где добавленный
(`Card.is_synthetic`).

## Демо-сценарии (обязательны)
1. Плотная категория, осенняя дата (Ведущий — 15 / Фотограф — 12 / Банкетный зал — 8).
2. Редкая категория (Флорист/Декоратор/Сувениры/Ведущий церемонии/Фотобудки/Отель/Инструменталист — по 3).
3. Запрос без результата — с текстовым объяснением.
