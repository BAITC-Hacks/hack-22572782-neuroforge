"""Запуск из корня репозитория: uv run --frozen --extra dev python run.py.

--check выполняет автономные тесты и демо на настоящих эмбеддингах.
--llm включает настроенные внешние провайдеры для интерактивного сервиса.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Проверить проект без запуска сервера и без внешних LLM')
    parser.add_argument('--llm', action='store_true', help='Разрешить внешние LLM в интерфейсе; нужны ключи в .env')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    os.environ['NEUROFORGE_LLM_ENABLED'] = 'true' if args.llm and not args.check else 'false'
    if args.check:
        for command in ([sys.executable, '-m', 'pytest', '-q'],
                        [sys.executable, 'scripts/run_demo_queries.py']):
            result = subprocess.run(command)
            if result.returncode:
                return result.returncode
        return 0

    try:
        import uvicorn
    except ImportError:
        print('Установите зависимости: python -m pip install -e ".[dev]"', file=sys.stderr)
        return 1
    print(f'NeuroForge: http://{args.host}:{args.port}', flush=True)
    print('Первый запуск может скачивать модель. Дождитесь Application startup complete.', flush=True)
    uvicorn.run('api.main:app', host=args.host, port=args.port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
