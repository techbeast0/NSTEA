"""NS-TEA Phase 5 FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nstea.api.middleware import CorrelationMiddleware
from nstea.api.routes import analysis, calculators, feedback, health
from nstea.core.logging import configure_logging

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    configure_logging()
    # Pre-warm vector store on startup
    from nstea.tools.guideline_search import _get_store
    _get_store()
    yield


app = FastAPI(
    title="NS-TEA Clinical Decision Support API",
    version="0.5.0",
    description=(
        "Neuro-Symbolic Temporal EHR Agent — "
        "evidence-based clinical reasoning with RAG + temporal analysis + symbolic constraints."
    ),
    lifespan=lifespan,
)

app.add_middleware(CorrelationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router, tags=["health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(calculators.router, prefix="/api/v1", tags=["calculators"])

# Serve static frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/app.js")
    async def serve_app_js():
        return FileResponse(str(FRONTEND_DIR / "app.js"), media_type="application/javascript")
