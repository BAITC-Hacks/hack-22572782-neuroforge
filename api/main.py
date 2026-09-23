"""
FastAPI-приложение. Единственный содержательный эндпоинт — POST /recommend.
Логика полностью делегирована neuroforge.pipeline — здесь только HTTP-обвязка.
"""
from fastapi import FastAPI

from neuroforge.schemas import Query, RecommendResponse

app = FastAPI(title="NeuroForge — умный подбор подрядчиков")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(query: Query) -> RecommendResponse:
    raise NotImplementedError("TODO: вызвать neuroforge.pipeline после его реализации")
