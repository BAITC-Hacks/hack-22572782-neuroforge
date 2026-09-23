"""Небольшая платная проверка каждого настроенного провайдера без кэша.

Проверяет настоящий ответ и разрешённые ID фактов. Успешный fallback не
считается успешным вызовом модели. Ключи и содержимое ошибок не выводятся.
"""
import json
import time
from pathlib import Path

from neuroforge.explain import build_llm_messages, parse_fact_selection
from neuroforge.llm_client import LLMClient
from neuroforge.outcome import run_funnel
from neuroforge.pipeline import Recommender
from neuroforge.schemas import Query


def main() -> int:
    providers = LLMClient().available_providers()
    if not providers:
        print('Нет настроенных API-ключей; живые вызовы не выполнены.')
        return 1
    scenarios = json.loads((Path(__file__).resolve().parents[1] / 'data/demo_queries.json').read_text())
    query = Query.model_validate({**scenarios[0]['query'], 'language': 'русский', 'duration_h': 5})
    service = Recommender.bootstrap(use_llm=False)
    service.warm_up()
    try:
        funnel = run_funnel(service.profiles, query)
        facts = service._build_facts(funnel.survivors[0], query, funnel, funnel.survivors)
        system, user = build_llm_messages(facts)
        valid_count = 0
        for provider in providers:
            started = time.perf_counter()
            text = LLMClient(providers=[provider]).complete(system, user)
            selected = parse_fact_selection(text, facts) if text else None
            valid = selected is not None
            valid_count += int(valid)
            print(json.dumps({'provider': provider.name, 'model': provider.model,
                              'valid_response': valid, 'fact_ids': selected,
                              'seconds': round(time.perf_counter() - started, 3)}, ensure_ascii=False), flush=True)
        print(f'Проверенный ответ: {valid_count}/{len(providers)} провайдеров.')
        return 0 if valid_count == len(providers) else 1
    finally:
        service.close()


if __name__ == '__main__':
    raise SystemExit(main())
