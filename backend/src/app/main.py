from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.graphql.api import graphql_app
from app.api.router import router as api_router
from app.core.config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"MongoDB: {settings.MONGODB_HOST}:{settings.MONGODB_PORT}")
    print(f"MinIO: {settings.minio_endpoint}")
    yield
    # Shutdown
    print(f"Shutting down {settings.PROJECT_NAME}")

def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.PROJECT_VERSION,
        lifespan=lifespan,
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "project": settings.PROJECT_NAME,
            "description": settings.PROJECT_DESCRIPTION,
            "version": settings.PROJECT_VERSION,
            "environment": settings.ENVIRONMENT,
            "docs": "/docs",
            "services": settings.get_service_urls()
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "project": settings.PROJECT_NAME,
            "version": settings.PROJECT_VERSION,
            "environment": settings.ENVIRONMENT
        }

    app.include_router(api_router, prefix="/api", tags=["api"])

    app.include_router(graphql_app, prefix="/graphql", tags=["graphql"])

    return app

app = create_application()