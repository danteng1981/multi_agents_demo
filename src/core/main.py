"""
Enterprise Agent Platform - Main Application
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.core.config import settings
from src.core.database import db_pool
from src.core.cache import cache_manager
from src.middleware.auth import AuthMiddleware
from src. middleware.audit import AuditMiddleware
from src.middleware.rate_limit import RateLimitMiddleware
from src.api.v1 import agent, health, session, audit

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus指标
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

agent_execution_count = Counter(
    'agent_executions_total',
    'Total agent executions',
    ['agent_type', 'status']
)

agent_execution_duration = Histogram(
    'agent_execution_duration_seconds',
    'Agent execution duration',
    ['agent_type']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("🚀 Starting Enterprise Agent Platform...")
    
    try:
        # 初始化数据库连接池
        await db_pool.initialize(
            dsn=settings.DATABASE_URL,
            min_size=settings. DB_POOL_MIN_SIZE,
            max_size=settings.DB_POOL_MAX_SIZE
        )
        logger.info("✅ Database pool initialized")
        
        # 初始化Redis缓存
        await cache_manager.initialize(settings.REDIS_URL)
        logger.info("✅ Redis cache initialized")
        
        logger.info("✅ Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("🛑 Shutting down Enterprise Agent Platform...")
    
    try:
        await db_pool. close()
        await cache_manager.close()
        logger.info("✅ Cleanup complete")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description="企业级智能体平台 - 高可用、高性能、安全、合规",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# 可信主机中间件
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # 生产环境应配置具体域名
    )


# 请求追踪中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加请求处理时间"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # 记录Prometheus指标
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    return response


# 请求ID中间件
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """添加请求ID"""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# 添加自定义中间件（按顺序）
if settings.ENABLE_AUDIT_LOG:
    app.add_middleware(AuditMiddleware)

if settings.RATE_LIMIT_ENABLED: 
    app.add_middleware(RateLimitMiddleware)

# 认证中间件（某些路由需要）
# app.add_middleware(AuthMiddleware)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc:  Exception):
    """全局异常处理器"""
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
    """参数错误处理"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# 注册路由
app.include_router(
    health.router,
    tags=["健康检查"]
)

app.include_router(
    agent.router,
    prefix=f"/api/{settings.API_VERSION}/agent",
    tags=["智能体"]
)

app.include_router(
    session. router,
    prefix=f"/api/{settings.API_VERSION}/session",
    tags=["会话管理"]
)

app.include_router(
    audit.router,
    prefix=f"/api/{settings.API_VERSION}/audit",
    tags=["审计日志"]
)


# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "version": settings.API_VERSION,
        "status": "running",
        "docs":  "/docs",
        "health": "/health"
    }


# Prometheus指标端点
@app.get("/metrics")
async def metrics():
    """Prometheus指标导出"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# 启动消息
@app.on_event("startup")
async def startup_message():
    """启动消息"""
    logger.info("=" * 50)
    logger.info(f"🚀 {settings.APP_NAME} Started")
    logger.info(f"📝 Environment: {settings.APP_ENV}")
    logger.info(f"🐛 Debug Mode: {settings.DEBUG}")
    logger.info(f"📚 API Docs: http://{settings.HOST}:{settings. PORT}/docs")
    logger.info(f"💚 Health Check: http://{settings.HOST}:{settings.PORT}/health")
    logger.info(f"📊 Metrics: http://{settings.HOST}:{settings.PORT}/metrics")
    logger.info("=" * 50)


if __name__ == "__main__": 
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL. lower()
    )
