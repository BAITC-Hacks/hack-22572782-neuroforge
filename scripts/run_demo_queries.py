"""Проверяемое демо на настоящих эмбеддингах. Без внешних LLM по умолчанию.

    python scripts/run_demo_queries.py
    python scripts/run_demo_queries.py --llm
"""
import argparse
import json
import time
from pathlib import Path

from neuroforge.pipeline import Recommender
from neuroforge.schemas import Query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm', action='store_true', help='Разрешить вызовы настроенных LLM-провайдеров')
    args = parser.parse_args()
    scenarios = json.loads((Path(__file__).resolve().parents[1] / 'data/demo_queries.json').read_text(encoding='utf-8'))
    started = time.perf_counter()
    service = Recommender.bootstrap(use_llm=args.llm)
    service.warm_up()
    print(f'Загрузка и прогрев: {time.perf_counter() - started:.2f} с')
    outputs = {}
    failed = False
    try:
        for scenario in scenarios:
            query = Query.model_validate(scenario['query'])
            started = time.perf_counter()
            response = service.recommend(query)
            elapsed = time.perf_counter() - started
            repeated = service.recommend(query)
            ids = [card.id for card in response.cards]
            stable = ids == [card.id for card in repeated.cards]
            valid = response.outcome.value == scenario['expected_outcome'] and stable and elapsed < 10
            failed |= not valid
            outputs[scenario['id']] = ids
            print(f'\n{"PASS" if valid else "FAIL"} {scenario["title"]}: {response.outcome.value}, {elapsed:.3f} с; порядок стабилен: {stable}')
            if response.message:
                print(response.message)
            for card in response.cards:
                print(f'  {card.id} · {card.name} · от {card.price_from_kzt:,} ₸ · synthetic={card.is_synthetic}')
                print(f'  {card.explanation}')
        changed = outputs['dense'] != outputs['date-change']
        failed |= not changed
        print(f'\nСмена даты изменила выдачу: {changed}')
    finally:
        service.close()
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
