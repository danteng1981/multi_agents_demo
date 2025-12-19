"""FastAPI Application Main Entry Point

Complete FastAPI application with middleware, routes, metrics, and lifecycle management.
Includes health checks, error handling, CORS, request logging, and Prometheus metrics.

Author: danteng1981
Created: 2025-12-19
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)
ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Number of active HTTP requests",
)
ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP errors",
    ["method", "endpoint", "error_type"],
)

# Application state
app_state = {
    "startup_time": None,
    "request_count": 0,
    "is_ready": False,
}


# Pydantic Models
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service health status")
    timestamp: float = Field(..., description="Current timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    version: str = Field(default="1.0.0", description="Application version")


class ReadinessResponse(BaseModel):
    """Readiness check response model"""
    ready: bool = Field(..., description="Service readiness status")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual readiness checks")


class MetricsResponse(BaseModel):
    """Metrics summary response model"""
    total_requests: int = Field(..., description="Total number of requests processed")
    active_requests: int = Field(..., description="Current active requests")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class AgentRequest(BaseModel):
    """Agent task request model"""
    task: str = Field(..., description="Task description", min_length=1)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task parameters")
    timeout: Optional[int] = Field(default=30, description="Task timeout in seconds", ge=1, le=300)


class AgentResponse(BaseModel):
    """Agent task response model"""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    result: Optional[Any] = Field(None, description="Task result")
    execution_time: float = Field(..., description="Execution time in seconds")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: float = Field(..., description="Error timestamp")


# Lifecycle Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    # Startup
    logger.info("Starting FastAPI application...")
    app_state["startup_time"] = time.time()
    
    # Simulate initialization tasks
    await asyncio.sleep(0.1)
    
    # Perform health checks
    try:
        # Add your initialization logic here
        # Example: database connections, cache warming, etc.
        logger.info("Initializing application components...")
        await asyncio.sleep(0.5)
        
        app_state["is_ready"] = True
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app_state["is_ready"] = False
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    app_state["is_ready"] = False
    
    # Cleanup tasks
    try:
        # Add your cleanup logic here
        # Example: close database connections, flush caches, etc.
        logger.info("Performing cleanup tasks...")
        await asyncio.sleep(0.1)
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Demo API",
    description="Complete FastAPI application with middleware, metrics, and lifecycle management",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Middleware Configuration

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host Middleware (uncomment and configure for production)
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["example.com", "*.example.com"]
# )


# Custom Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Metrics collection middleware"""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        ERROR_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            error_type=type(e).__name__,
        ).inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request/response logging middleware"""
    request_id = f"{time.time()}-{id(request)}"
    
    logger.info(
        f"Request started: {request.method} {request.url.path} "
        f"[ID: {request_id}] [Client: {request.client.host if request.client else 'unknown'}]"
    )
    
    try:
        response = await call_next(request)
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"[ID: {request_id}] [Status: {response.status_code}]"
        )
        return response
    except Exception as e:
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"[ID: {request_id}] [Error: {str(e)}]"
        )
        raise


# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=time.time(),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            detail=str(exc),
            timestamp=time.time(),
        ).dict(),
    )


# Health Check Routes
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
    description="Returns the health status of the application",
)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    current_time = time.time()
    uptime = None
    
    if app_state["startup_time"]:
        uptime = current_time - app_state["startup_time"]
    
    return HealthResponse(
        status="healthy",
        timestamp=current_time,
        uptime_seconds=uptime,
        version="1.0.0",
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness check endpoint",
    description="Returns whether the application is ready to serve requests",
)
async def readiness_check() -> ReadinessResponse:
    """Readiness check endpoint"""
    checks = {
        "application": app_state["is_ready"],
        "startup_complete": app_state["startup_time"] is not None,
    }
    
    is_ready = all(checks.values())
    
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )
    
    return ReadinessResponse(
        ready=is_ready,
        checks=checks,
    )


@app.get(
    "/live",
    tags=["Health"],
    summary="Liveness check endpoint",
    description="Returns whether the application is alive",
)
async def liveness_check() -> Dict[str, str]:
    """Liveness check endpoint"""
    return {"status": "alive"}


# Metrics Routes
@app.get(
    "/metrics",
    tags=["Metrics"],
    summary="Prometheus metrics endpoint",
    description="Returns Prometheus-formatted metrics",
)
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get(
    "/metrics/summary",
    response_model=MetricsResponse,
    tags=["Metrics"],
    summary="Metrics summary endpoint",
    description="Returns a summary of application metrics",
)
async def metrics_summary() -> MetricsResponse:
    """Metrics summary endpoint"""
    uptime = 0.0
    if app_state["startup_time"]:
        uptime = time.time() - app_state["startup_time"]
    
    return MetricsResponse(
        total_requests=app_state["request_count"],
        active_requests=int(ACTIVE_REQUESTS._value.get()),
        uptime_seconds=uptime,
    )


# Agent Routes
@app.post(
    "/api/v1/agent/execute",
    response_model=AgentResponse,
    tags=["Agent"],
    summary="Execute agent task",
    description="Execute a task using the multi-agent system",
    status_code=status.HTTP_200_OK,
)
async def execute_agent_task(request: AgentRequest) -> AgentResponse:
    """Execute an agent task"""
    start_time = time.time()
    task_id = f"task-{int(start_time * 1000)}"
    
    logger.info(f"Executing task: {task_id} - {request.task}")
    
    try:
        # Simulate task execution
        await asyncio.sleep(0.1)
        
        # Your agent logic here
        result = {
            "message": f"Task '{request.task}' completed successfully",
            "parameters": request.parameters,
        }
        
        execution_time = time.time() - start_time
        app_state["request_count"] += 1
        
        return AgentResponse(
            task_id=task_id,
            status="completed",
            result=result,
            execution_time=execution_time,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Task execution timeout after {request.timeout} seconds",
        )
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task execution failed: {str(e)}",
        )


@app.get(
    "/api/v1/agent/status/{task_id}",
    response_model=AgentResponse,
    tags=["Agent"],
    summary="Get task status",
    description="Retrieve the status of a specific task",
)
async def get_task_status(task_id: str) -> AgentResponse:
    """Get task status"""
    # In a real application, you would query a database or cache
    # This is a simplified example
    
    if not task_id.startswith("task-"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    
    return AgentResponse(
        task_id=task_id,
        status="completed",
        result={"message": "Task status retrieved"},
        execution_time=0.1,
    )


# Root Route
@app.get(
    "/",
    tags=["Root"],
    summary="API root endpoint",
    description="Returns basic API information",
)
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    uptime = None
    if app_state["startup_time"]:
        uptime = time.time() - app_state["startup_time"]
    
    return {
        "name": "Multi-Agent Demo API",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": uptime,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
        },
    }


# Development server configuration
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )
