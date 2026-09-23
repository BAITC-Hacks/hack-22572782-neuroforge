"""
Выбор акцентов объяснения через LLM; произвольный текст в карточки не попадает.

Устройство определяется тремя требованиями ТЗ, которые тянут в разные стороны:
ответ до 10 секунд, детерминизм выдачи и качество формулировок.

* **Цепочка провайдеров.** OpenAI и NVIDIA NIM говорят по одному протоколу
  (NIM специально OpenAI-совместим), поэтому клиент один, а провайдеры —
  список в конфиге. Отказ или лимит у одного не роняет сервис.
* **Никаких исключений наружу.** Любая ошибка возвращает None, и вызывающий
  код берёт шаблонный текст. Сеть не должна уметь сломать выдачу.
* **Кэш на диске.** Один и тот же кандидат в одном и том же запросе обязан
  получить один и тот же текст, иначе повторный прогон на защите даст другие
  формулировки. Заодно повторный запрос не тратит ни токенов, ни времени.
"""
import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path

from neuroforge.config import LLMProvider, settings

logger = logging.getLogger(__name__)


class ExplanationCache:
    """SQLite-кэш проверенных JSON-ответов. Ключ зависит от промпта и провайдеров."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS explanations ("
                "key TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=0.1)

    @staticmethod
    def make_key(*parts: str) -> str:
        digest = hashlib.sha256()
        for part in parts:
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT text FROM explanations WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO explanations (key, text) VALUES (?, ?)",
                (key, text),
            )


class LLMUnavailable(Exception):
    """Ни один провайдер не ответил. Не выходит за пределы модуля."""


class LLMClient:
    """Перебирает провайдеров по порядку до первого успешного ответа."""

    def __init__(self, providers: list[LLMProvider] | None = None) -> None:
        self.providers = providers if providers is not None else settings.llm_providers
        self._clients: dict[str, object] = {}

    def available_providers(self) -> list[LLMProvider]:
        """Провайдеры, для которых реально есть ключ в окружении."""
        return [p for p in self.providers if os.environ.get(p.api_key_env)]

    def _client_for(self, provider: LLMProvider):
        if provider.name not in self._clients:
            from openai import OpenAI

            self._clients[provider.name] = OpenAI(
                api_key=os.environ[provider.api_key_env],
                base_url=provider.base_url,
                timeout=settings.llm_timeout_s,
                max_retries=0,  # ретраи съели бы бюджет времени на ответ
            )
        return self._clients[provider.name]

    def complete(self, system: str, user: str) -> str | None:
        """Текст от первого ответившего провайдера, либо None.

        None — штатный исход, а не ошибка: вызывающий код берёт шаблон.
        """
        deadline = time.monotonic() + settings.llm_total_timeout_s
        for provider in self.available_providers():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                client = self._client_for(provider)
                response = client.chat.completions.create(
                    model=provider.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    timeout=min(settings.llm_timeout_s, remaining),
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text
                logger.warning("Провайдер %s вернул пустой текст", provider.name)
            except Exception as exc:
                logger.warning(
                    "Провайдер %s недоступен (%s, HTTP %s), пробуем следующий",
                    provider.name,
                    type(exc).__name__,
                    getattr(exc, "status_code", "нет ответа"),
                )
        return None
