"""
Загрузка датасета (реальные профили + наши синтетические дополнения).

Здесь же вычисляются множества допустимых значений (города, категории,
форматы, языки) — динамически, из фактических данных. Ничего не хардкодим:
добавление новой категории/города/языка в JSONL не требует правки кода.
"""
import json
from pathlib import Path

from neuroforge.config import settings
from neuroforge.schemas import Profile


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_profiles() -> list[Profile]:
    """Грузит real (dataset_path) + synthetic (synthetic_path) профили.

    Файл synthetic_path опционален — если его нет, датасет состоит только
    из реальных профилей. Синтетические профили обязаны иметь synthetic=true
    (проверяется явно, чтобы источник был виден в демо через Card.is_synthetic).
    """
    real_records = _read_jsonl(Path(settings.dataset_path))
    synthetic_records = _read_jsonl(Path(settings.synthetic_path))

    profiles = [Profile.model_validate(r) for r in real_records]

    for record in synthetic_records:
        profile = Profile.model_validate(record)
        if not profile.synthetic:
            raise ValueError(
                f"Профиль {profile.id} из {settings.synthetic_path} должен быть "
                "помечен synthetic: true"
            )
        profiles.append(profile)

    ids = [p.id for p in profiles]
    duplicate_ids = {i for i in ids if ids.count(i) > 1}
    if duplicate_ids:
        raise ValueError(f"Дублирующиеся id профилей в датасете: {duplicate_ids}")

    return profiles


def known_cities(profiles: list[Profile]) -> set[str]:
    return {p.city for p in profiles}


def known_categories(profiles: list[Profile]) -> set[str]:
    return {c for p in profiles for c in p.categories}


def known_event_formats(profiles: list[Profile]) -> set[str]:
    return {f for p in profiles for f in p.event_formats}


def known_languages(profiles: list[Profile]) -> set[str]:
    return {lang for p in profiles for lang in p.languages}
