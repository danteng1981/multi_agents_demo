"""
Authentication Middleware
"""
import logging
from typing import Optional, List
from functools import wraps

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.security import jwt_manager, rbac_manager, Permission, Role

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    # 无需认证的路径
    WHITELIST_PATHS = [
        "/",
        "/health",
        "/readiness",
        "/liveness",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi. json",
    ]
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        
        # 白名单路径跳过认证
        if any(request.url.path. startswith(path) for path in self.WHITELIST_PATHS):
            return await call_next(request)
        
        # 获取token
        token = self._extract_token(request)
        
        if not token:
            return self._unauthorized_response("Missing authentication token")
        
        try: 
            # 验证token
            payload = jwt_manager.verify_token(token)
            
            # 将用户信息存储到request.state
            request.state. user_id = payload.get("sub")
            request.state.user_roles = [
                Role(role) for role in payload.get("roles", [Role.GUEST. value])
            ]
            request.state.token_payload = payload
            
            logger.debug(f"User authenticated: {request.state.user_id}")
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return self._unauthorized_response(str(e))
        
        response = await call_next(request)
        return response
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """从请求中提取token"""
        # 从Header获取
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]
        
        # 从Query参数获取（不推荐，仅用于特殊场景）
        token = request. query_params.get("token")
        if token:
            return token
        
        return None
    
    def _unauthorized_response(self, message: str):
        """返回未授权响应"""
        from fastapi. responses import JSONResponse
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "Unauthorized",
                "message": message
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


def require_permission(permission: Permission):
    """
    权限检查装饰器
    
    Usage:
        @require_permission(Permission.WRITE_AGENT)
        async def create_agent(... ):
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 从kwargs获取request对象
            request = kwargs.get('request')
            
            if not request:
                # 尝试从args获取
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            # 获取用户角色
            user_roles = getattr(request.state, "user_roles", [Role.GUEST])
            
            # 检查权限
            if not rbac_manager.has_permission(user_roles, permission):
                raise HTTPException(
                    status_code=status. HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission.value}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(required_role: Role):
    """
    角色检查装饰器
    
    Usage:
        @require_role(Role.ADMIN)
        async def admin_function(...):
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            
            if not request: 
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request: 
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            user_roles = getattr(request. state, "user_roles", [])
            
            if required_role not in user_roles: 
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {required_role.value}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_current_user(request: Request) -> dict:
    """
    获取当前用户信息
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        用户信息字典
    """
    return {
        "user_id": getattr(request.state, "user_id", None),
        "roles": getattr(request.state, "user_roles", []),
        "permissions": rbac_manager.get_user_permissions(
            getattr(request.state, "user_roles", [])
        )
    }
