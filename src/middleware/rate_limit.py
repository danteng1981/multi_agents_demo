"""
Rate Limiting Middleware
"""
import time
import logging
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings
from src.core.cache import cache_manager

logger = logging.getLogger(__name__)


class TokenBucket:
    """令牌桶算法"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        初始化令牌桶
        
        Args: 
            capacity: 桶容量
            refill_rate: 令牌填充速率（每秒）
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill_time = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        消费令牌
        
        Args:
            tokens: 需要消费的令牌数
            
        Returns:
            是否成功消费
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """填充令牌"""
        now = time.time()
        elapsed = now - self.last_refill_time
        
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill_time = now


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件"""
    
    # 白名单路径
    WHITELIST_PATHS = [
        "/health",
        "/readiness",
        "/liveness",
        "/metrics",
    ]
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # 白名单跳过限流
        if any(request.url.path. startswith(path) for path in self.WHITELIST_PATHS):
            return await call_next(request)
        
        # 获取限流键
        rate_limit_key = self._get_rate_limit_key(request)
        
        # 检查限流
        allowed = await self._check_rate_limit(rate_limit_key)
        
        if not allowed:
            return self._rate_limit_exceeded_response(rate_limit_key)
        
        response = await call_next(request)
        
        # 添加限流信息到响应头
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_PER_MINUTE)
        
        return response
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """生成限流键"""
        # 优先使用用户ID
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"rate_limit: user:{user_id}"
        
        # 使用IP地址
        client_ip = request.client.host if request.client else "unknown"
        return f"rate_limit:ip:{client_ip}"
    
    async def _check_rate_limit(self, key: str) -> bool:
        """
        检查是否超过限流
        
        Args:
            key: 限流键
            
        Returns:
            是否允许请求
        """
        try:
            if not cache_manager.redis_client:
                # Redis不可用，允许请求
                return True
            
            # 获取当前计数
            current = await cache_manager.redis_client.get(key)
            
            if current is None:
                # 首次请求，设置计数为1
                await cache_manager. redis_client.setex(
                    key,
                    60,  # 1分钟过期
                    1
                )
                return True
            
            current_count = int(current)
            
            if current_count >= settings.RATE_LIMIT_PER_MINUTE:
                logger.warning(f"Rate limit exceeded for {key}")
                return False
            
            # 增加计数
            await cache_manager.redis_client.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # 出错时允许请求
            return True
    
    def _rate_limit_exceeded_response(self, key: str):
        """返回限流响应"""
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate Limit Exceeded",
                "message": f"Too many requests.  Limit: {settings.RATE_LIMIT_PER_MINUTE}/minute",
                "retry_after": 60
            },
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_PER_MINUTE),
                "X-RateLimit-Remaining": "0"
            }
        )
