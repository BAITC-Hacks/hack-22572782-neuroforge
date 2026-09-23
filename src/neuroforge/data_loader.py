"""
Загрузка датасета (реальные профили + наши синтетические дополнения).

Здесь же вычисляются множества допустимых значений (города, категории,
форматы, языки) — динамически, из фактических данных. Ничего не хардкодим.

TODO (шаг обработки датасета): реализовать load_profiles(), merge с
data/synthetic/extra_profiles.jsonl, валидацию через schemas.Profile.
"""
from functools import lru_cache
from pathlib import Path

from neuroforge.config import settings
from neuroforge.schemas import Profile


def load_profiles() -> list[Profile]:
    """Грузит real + synthetic профили, размечает источник (Profile.synthetic)."""
    raise NotImplementedError("TODO: реализовать при обработке датасета")


@lru_cache
def known_cities(profiles: tuple[Profile, ...]) -> set[str]:
    return {p.city for p in profiles}


@lru_cache
def known_categories(profiles: tuple[Profile, ...]) -> set[str]:
    return {c for p in profiles for c in p.categories}
