from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.config.settings import settings
from app.db.connection import init_db, close_connections, get_redis
from app.api.middleware.logging import RequestLoggingMiddleware
from app.api.middleware.rate_limiter import RateLimitMiddleware
from app.api.middleware.error_handler import ErrorHandlerMiddleware, http_exception_handler
from app.api.routes import auth, predictions, models, admin, llm, rag, batch

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])

start_time = datetime.utcnow()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown
    await close_connections()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise AI Platform with ML/NLP capabilities",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)

# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(llm.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(batch.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "uptime_seconds": (datetime.utcnow() - start_time).total_seconds()
    }
    
    # Check database
    try:
        from sqlalchemy import text
        from app.db.connection import async_engine
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health["database"] = "connected"
    except Exception as e:
        health["database"] = f"error: {str(e)}"
        health["status"] = "degraded"
    
    # Check Redis
    try:
        redis = await get_redis()
        await redis.ping()
        health["redis"] = "connected"
    except Exception as e:
        health["redis"] = f"error: {str(e)}"
        health["status"] = "degraded"
    
    return health

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/v1/info")
async def api_info():
    """API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "auth": "/api/v1/auth",
            "predictions": "/api/v1/predictions",
            "models": "/api/v1/models",
            "admin": "/api/v1/admin",
            "llm": "/api/v1/llm",
            "rag": "/api/v1/rag",
            "batch": "/api/v1/batch"
        },
        "features": [
            "Sentiment Analysis",
            "Named Entity Recognition",
            "Zero-shot Classification",
            "Semantic Similarity",
            "Text Embeddings",
            "RAG System",
            "ML Pipeline",
            "LLM Chat (GPT-4, Claude)",
            "Advanced RAG with Re-ranking",
            "Batch Processing",
            "Vector Database (Qdrant)"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
