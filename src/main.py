"""
Enterprise Agent Platform - Main Application
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi. middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.core.config import settings
from src.core.database import db_pool
from src.core. cache import cache_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("🚀 Starting Enterprise Agent Platform...")
    
    try:
        # Initialize database pool
        await db_pool.initialize(
            dsn=settings.DATABASE_URL,
            min_size=settings.DB_POOL_MIN_SIZE,
            max_size=settings.DB_POOL_MAX_SIZE
        )
        logger.info("✅ Database pool initialized")
        
        # Initialize Redis cache
        await cache_manager. initialize(settings.REDIS_URL)
        logger.info("✅ Redis cache initialized")
        
        logger.info("✅ Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enterprise Agent Platform...")
    
    try:
        await db_pool.close()
        await cache_manager.close()
        logger.info("✅ Cleanup complete")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Agent Platform - High Availability, High Performance, Secure, Compliant",
    version=settings. API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# Trusted host middleware for production
if not settings.DEBUG:
    app. add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure specific domains in production
    )


# Request tracking middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time to response headers"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response. headers["X-Process-Time"] = str(process_time)
    
    # Record Prometheus metrics
    request_count. labels(
        method=request. method,
        endpoint=request. url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    return response


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID"""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "request_id": getattr(request. state, "request_id", None)
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Value error handler"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# Import and register API routes
try:
    from src.api. v1 import health
    app.include_router(health.router, tags=["Health"])
except ImportError:
    logger.warning("Health router not found, using minimal health endpoint")
    
    @app.get("/health")
    async def minimal_health():
        """Minimal health check"""
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.API_VERSION
        }


# Try to import other routers
try:
    from src.api.v1 import agent
    app.include_router(
        agent.router,
        prefix=f"/api/{settings.API_VERSION}/agent",
        tags=["Agent"]
    )
except ImportError: 
    logger.warning("Agent router not available")

try:
    from src.api.v1 import session
    app.include_router(
        session.router,
        prefix=f"/api/{settings.API_VERSION}/session",
        tags=["Session"]
    )
except ImportError:
    logger.warning("Session router not available")

try:
    from src.api.v1 import audit
    app.include_router(
        audit.router,
        prefix=f"/api/{settings.API_VERSION}/audit",
        tags=["Audit"]
    )
except ImportError:
    logger. warning("Audit router not available")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.API_VERSION,
        "status": "running",
        "docs":  "/docs",
        "health": "/health"
    }


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics export"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Startup message
@app.on_event("startup")
async def startup_message():
    """Display startup message"""
    logger.info("=" * 60)
    logger.info(f"🚀 {settings. APP_NAME} Started")
    logger.info(f"📝 Environment: {settings.APP_ENV}")
    logger.info(f"🐛 Debug Mode: {settings.DEBUG}")
    logger.info(f"📚 API Docs: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"💚 Health:  http://{settings.HOST}:{settings.PORT}/health")
    logger.info(f"📊 Metrics: http://{settings.HOST}:{settings.PORT}/metrics")
    logger.info("=" * 60)


if __name__ == "__main__": 
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL. lower()
    )
