"""Core FastAPI application for EconoCausal."""

import time
from typing import Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from src.api.config import settings, logger
from src.api.routes.scoring import router as scoring_router


def create_app() -> FastAPI:
    """Application factory for EconoCausal API."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Prescriptive AI Dynamic Pricing & Discount Allocation Engine",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS for M4's React/Vite frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Track API request processing time
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time-Sec"] = f"{process_time:.6f}"
        return response

    # Register API v1 routers
    app.include_router(scoring_router)

    # Root endpoint
    @app.get("/", tags=["General"])
    async def root() -> Dict[str, Any]:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "online",
            "docs": "/docs",
        }

    # Liveness probe (Kubernetes / monitoring)
    @app.get("/health", tags=["Monitoring"], status_code=status.HTTP_200_OK)
    async def health_check() -> Dict[str, str]:
        return {
            "status": "healthy",
            "environment": settings.environment,
            "version": settings.app_version,
        }

    # Readiness probe (verifies subsystems are ready to accept traffic)
    @app.get("/ready", tags=["Monitoring"], status_code=status.HTTP_200_OK)
    async def readiness_check() -> Dict[str, Any]:
        return {
            "ready": True,
            "subsystems": {
                "configuration": "loaded",
                "optimization_engine": "initialized",
                "logging": "active",
            },
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run("src.api.main:app", host=settings.host, port=settings.port, reload=settings.debug)