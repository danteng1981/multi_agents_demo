"""
Audit Logging Middleware
"""
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """审计动作"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    LOGIN = "login"
    LOGOUT = "logout"


class AuditLevel(Enum):
    """审计级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLogger:
    """审计日志记录器"""
    
    async def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
        level:  str = "info",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        记录审计日志
        
        Args:
            user_id: 用户ID
            action: 操作动作
            resource_type: 资源类型
            resource_id: 资源ID
            details: 详细信息
            level: 日志级别
            ip_address: IP地址
            user_agent:  User Agent
            request_id: 请求ID
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action":  action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
            "level": level,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "request_id": request_id,
        }
        
        # 记录到日志
        log_message = (
            f"AUDIT: user={user_id} action={action} "
            f"resource={resource_type}/{resource_id} "
            f"ip={ip_address}"
        )
        
        if level == "error" or level == "critical":
            logger.error(log_message, extra=log_entry)
        else:
            logger.info(log_message, extra=log_entry)
        
        # TODO: 存储到数据库或Elasticsearch
        # await self._store_to_database(log_entry)


class AuditMiddleware(BaseHTTPMiddleware):
    """审计日志中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.audit_logger = AuditLogger()
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        
        if not settings.ENABLE_AUDIT_LOG:
            return await call_next(request)
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 提取请求信息
        user_id = getattr(request. state, "user_id", "anonymous")
        request_id = getattr(request. state, "request_id", None)
        ip_address = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # 执行请求
        try:
            response = await call_next(request)
            
            # 记录成功的请求
            await self._log_request(
                request=request,
                response=response,
                user_id=user_id,
                request_id=request_id,
                ip_address=ip_address,
                user_agent=user_agent,
                duration=time.time() - start_time,
                status="success"
            )
            
            return response
            
        except Exception as e:
            # 记录失败的请求
            await self._log_request(
                request=request,
                response=None,
                user_id=user_id,
                request_id=request_id,
                ip_address=ip_address,
                user_agent=user_agent,
                duration=time.time() - start_time,
                status="error",
                error=str(e)
            )
            raise
    
    async def _log_request(
        self,
        request: Request,
        response: Optional[Any],
        user_id: str,
        request_id: Optional[str],
        ip_address: str,
        user_agent: str,
        duration: float,
        status: str,
        error: Optional[str] = None
    ):
        """记录请求日志"""
        
        # 确定动作类型
        action = self._determine_action(request. method)
        
        # 提取资源信息
        resource_type, resource_id = self._extract_resource(request. url. path)
        
        # 详细信息
        details = {
            "method": request.method,
            "path": request.url.path,
            "duration": duration,
            "status_code": response.status_code if response else None,
            "error": error
        }
        
        # 记录审计日志
        await self. audit_logger.log(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            level="error" if error else "info",
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id
        )
    
    def _determine_action(self, method: str) -> str:
        """根据HTTP方法确定动作"""
        action_map = {
            "GET":  "read",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete",
        }
        return action_map. get(method, "unknown")
    
    def _extract_resource(self, path: str) -> tuple:
        """从路径提取资源类型和ID"""
        parts = path.strip("/").split("/")
        
        if len(parts) >= 3:
            # /api/v1/agent/123 -> (agent, 123)
            return parts[2], parts[3] if len(parts) > 3 else "unknown"
        
        return "unknown", "unknown"


# 全局审计日志实例
audit_logger = AuditLogger()
