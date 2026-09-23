"""
FastAPI-приложение. Единственный содержательный эндпоинт — POST /recommend.
Логика полностью делегирована neuroforge.pipeline — здесь только HTTP-обвязка.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from neuroforge.config import settings
from neuroforge.data_loader import known_categories, known_cities, known_event_formats, known_languages
from neuroforge.pipeline import Recommender
from neuroforge.schemas import Query, RecommendResponse

ROOT = Path(__file__).resolve().parents[1]


def create_app(recommender: Recommender | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = recommender if recommender is not None else Recommender.bootstrap()
        service.warm_up()
        app.state.recommender = service
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="NeuroForge — умный подбор подрядчиков", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=ROOT / "ui"), name="assets")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "ui" / "index.html")

    @app.get("/health")
    def health() -> dict:
        service = app.state.recommender
        return {"status": "ok", "profile_count": len(service.profiles),
                "embedding_model": service.embedder.model_id,
                "explanation_mode": "llm_fact_selection" if service.llm else "facts"}

    @app.get("/catalog")
    def catalog() -> dict:
        profiles = app.state.recommender.profiles
        return {
            "cities": sorted(known_cities(profiles)),
            "categories": sorted(known_categories(profiles)),
            "event_types": sorted(known_event_formats(profiles)),
            "languages": sorted(known_languages(profiles)),
            "calendar_start": settings.calendar_start.isoformat(),
            "calendar_end": settings.calendar_end.isoformat(),
            "profile_count": len(profiles),
            "synthetic_count": sum(p.synthetic for p in profiles),
        }

    @app.get("/demo-scenarios")
    def demo_scenarios() -> list:
        return json.loads((ROOT / "data" / "demo_queries.json").read_text(encoding="utf-8"))

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(query: Query) -> RecommendResponse:
        try:
            return app.state.recommender.recommend(query)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
