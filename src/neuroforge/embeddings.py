"""
Семантические эмбеддинги описаний подрядчиков.

Считаются локально (sentence-transformers): на критичном, ранжирующем пути
не должно быть сетевой зависимости — иначе демо падает вместе с сетью, а
ранжирование перестаёт быть воспроизводимым.

Провайдер вынесен за Protocol: ядро скоринга зависит от интерфейса, а не от
sentence-transformers. Это позволяет подменить его в тестах (без скачивания
модели) и заменить на API-провайдера, не трогая scoring.py.
"""
import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from neuroforge.schemas import Profile

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str:
        """Идентификатор модели. Провайдер обязан описывать себя сам: кэш
        подписывается этим значением, и если бы имя приходило со стороны,
        подмена провайдера молча вернула бы векторы чужой модели."""
        ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Возвращает матрицу (len(texts), dim). Одинаковый вход — одинаковый выход."""
        ...


class SentenceTransformerEmbedder:
    """Ленивая обёртка: модель весит сотни мегабайт и грузится секунды,
    поэтому загружается при первом обращении, а не при импорте."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model_id(self) -> str:
        return self.model_name

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        return np.asarray(model.encode(texts, normalize_embeddings=True))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинус в исходном диапазоне [-1, 1], приведённый к [0, 1], чтобы все
    фичи скоринга жили в одной шкале."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    raw = float(np.dot(a, b) / denominator)
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))


def _fingerprint(profiles: list[Profile], model_name: str) -> str:
    """Отпечаток входных данных: при смене модели или любого описания кэш
    считается протухшим и пересчитывается. Иначе легко получить эмбеддинги
    от старого датасета и молча ранжировать по несуществующим текстам."""
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for profile in sorted(profiles, key=lambda p: p.id):
        digest.update(profile.id.encode("utf-8"))
        digest.update(profile.description.encode("utf-8"))
    return digest.hexdigest()


def build_description_index(
    profiles: list[Profile],
    embedder: EmbeddingProvider,
    cache_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """id профиля -> вектор его описания.

    При наличии cache_path результат кэшируется на диск и переиспользуется,
    пока отпечаток датасета и модели не изменился. Идентификатор модели
    берётся у самого эмбеддера, а не из конфига: иначе подмена провайдера
    при неизменной настройке вернула бы векторы другой модели.
    """
    fingerprint = _fingerprint(profiles, embedder.model_id)

    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["fingerprint"]) == fingerprint:
                    ids, vectors = cached["ids"], cached["vectors"]
                    if (vectors.ndim == 2 and len(ids) == len(profiles) == len(vectors)
                            and set(ids) == {p.id for p in profiles}
                            and np.isfinite(vectors).all()):
                        return {pid: vectors[i] for i, pid in enumerate(ids)}
        except (OSError, ValueError, KeyError, EOFError):
            logger.warning("Кэш эмбеддингов повреждён; пересчитываем")

    ordered = sorted(profiles, key=lambda p: p.id)
    vectors = embedder.encode([p.description for p in ordered])
    index = {p.id: vectors[i] for i, p in enumerate(ordered)}

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                fingerprint=fingerprint,
                ids=np.array([p.id for p in ordered]),
                vectors=vectors,
            )
        except OSError:
            logger.warning("Не удалось сохранить кэш эмбеддингов; используем индекс в памяти")

    return index
