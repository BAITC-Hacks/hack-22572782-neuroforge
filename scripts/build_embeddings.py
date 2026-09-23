"""
Предпосчёт эмбеддингов описаний в кэш на диске.

Запускать после любого изменения датасета. Кэш самопроверяемый: он хранит
отпечаток (модель + все описания), и при несовпадении пересчитывается сам,
так что забытый запуск скрипта не приведёт к ранжированию по устаревшим
текстам — только к более медленному первому запросу.

    python scripts/build_embeddings.py
"""
import sys
import time
from pathlib import Path

from neuroforge.config import settings
from neuroforge.data_loader import load_profiles
from neuroforge.embeddings import SentenceTransformerEmbedder, build_description_index


def main() -> int:
    profiles = load_profiles()
    cache_path = Path(settings.embeddings_cache_path)

    print(f"Профилей: {len(profiles)}")
    print(f"Модель:   {settings.embedding_model}")
    print(f"Кэш:      {cache_path}")

    embedder = SentenceTransformerEmbedder(settings.embedding_model)

    started = time.perf_counter()
    index = build_description_index(
        profiles,
        embedder,
        cache_path=cache_path,
        model_name=settings.embedding_model,
    )
    elapsed = time.perf_counter() - started

    dimension = len(next(iter(index.values())))
    print(f"Готово: {len(index)} векторов, размерность {dimension}, {elapsed:.1f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
