"""
Security Module - Encryption, Authentication, Authorization
"""
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum

from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf. pbkdf2 import PBKDF2

from src.core.config import settings


# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Permission(Enum):
    """权限定义"""
    # Agent权限
    READ_AGENT = "read:agent"
    WRITE_AGENT = "write: agent"
    DELETE_AGENT = "delete:agent"
    ADMIN_AGENT = "admin: agent"
    
    # 数据权限
    READ_DATA = "read:data"
    WRITE_DATA = "write:data"
    EXPORT_DATA = "export:data"
    
    # 系统权限
    VIEW_AUDIT_LOG = "view:audit_log"
    MANAGE_USERS = "manage:users"
    ADMIN_SYSTEM = "admin:system"


class Role(Enum):
    """角色定义"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"


class DataEncryption: 
    """数据加密管理"""
    
    def __init__(self):
        """初始化加密器"""
        # 确保密钥长度正确
        key = settings.ENCRYPTION_KEY.encode()
        if len(key) != 32:
            # 使用PBKDF2派生32字节密钥
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'enterprise-agent',  # 生产环境应使用随机salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key))
        else:
            key = base64.urlsafe_b64encode(key)
        
        self. fernet = Fernet(key)
    
    def encrypt_field(self, data: str) -> str:
        """
        字段级加密
        
        Args:
            data: 要加密的数据
            
        Returns:
            加密后的字符串
        """
        if not data:
            return data
        
        encrypted = self.fernet.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_field(self, encrypted_data: str) -> str:
        """
        字段级解密
        
        Args:
            encrypted_data: 加密的数据
            
        Returns: 
            解密后的字符串
        """
        if not encrypted_data:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data. encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def hash_sensitive_data(self, data: str) -> str:
        """
        单向哈希（用于查询但不需要解密的数据）
        
        Args:
            data: 要哈希的数据
            
        Returns:
            哈希值
        """
        digest = hashes.Hash(hashes.SHA256())
        digest.update(data. encode())
        return base64.b64encode(digest. finalize()).decode()


class JWTManager:
    """JWT令牌管理"""
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建访问令牌
        
        Args:
            data: 要编码的数据
            expires_delta: 过期时间
            
        Returns:
            JWT令牌
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(seconds=settings.JWT_EXPIRATION)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """
        验证令牌
        
        Args: 
            token: JWT令牌
            
        Returns:
            解码后的数据
            
        Raises:
            JWTError: 令牌无效
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings. JWT_ALGORITHM]
            )
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")
    
    @staticmethod
    def create_refresh_token(user_id: str) -> str:
        """创建刷新令牌"""
        data = {
            "sub": user_id,
            "type":  "refresh"
        }
        expire = datetime.utcnow() + timedelta(days=7)
        data.update({"exp": expire})
        
        return jwt.encode(
            data,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )


class RBACManager:
    """基于角色的访问控制"""
    
    # 角色权限映射
    ROLE_PERMISSIONS = {
        Role. ADMIN: {
            Permission.READ_AGENT,
            Permission.WRITE_AGENT,
            Permission.DELETE_AGENT,
            Permission. ADMIN_AGENT,
            Permission.READ_DATA,
            Permission. WRITE_DATA,
            Permission. EXPORT_DATA,
            Permission.VIEW_AUDIT_LOG,
            Permission.MANAGE_USERS,
            Permission.ADMIN_SYSTEM,
        },
        Role. DEVELOPER: {
            Permission.READ_AGENT,
            Permission.WRITE_AGENT,
            Permission. READ_DATA,
            Permission. WRITE_DATA,
            Permission. VIEW_AUDIT_LOG,
        },
        Role.OPERATOR:  {
            Permission.READ_AGENT,
            Permission.READ_DATA,
            Permission.VIEW_AUDIT_LOG,
        },
        Role.VIEWER:  {
            Permission.READ_AGENT,
            Permission.READ_DATA,
        },
        Role.GUEST:  {
            Permission.READ_AGENT,
        }
    }
    
    @classmethod
    def get_user_permissions(cls, user_roles: list[Role]) -> set[Permission]:
        """
        获取用户的所有权限
        
        Args:
            user_roles: 用户角色列表
            
        Returns:
            权限集合
        """
        permissions = set()
        for role in user_roles:
            permissions.update(cls.ROLE_PERMISSIONS. get(role, set()))
        return permissions
    
    @classmethod
    def has_permission(
        cls,
        user_roles: list[Role],
        required_permission: Permission
    ) -> bool:
        """
        检查用户是否有指定权限
        
        Args:
            user_roles:  用户角色列表
            required_permission: 需要的权限
            
        Returns:
            是否有权限
        """
        user_permissions = cls.get_user_permissions(user_roles)
        return required_permission in user_permissions


class PasswordManager:
    """密码管理"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """哈希密码"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)


class RequestSigner:
    """请求签名验证"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def sign_request(self, data: Dict[str, Any], timestamp: int) -> str:
        """
        生成请求签名
        
        Args:
            data: 请求数据
            timestamp:  时间戳
            
        Returns:
            签名
        """
        # 排序参数
        sorted_params = sorted(data.items())
        param_str = '&'.join([f"{k}={v}" for k, v in sorted_params])
        
        # 添加时间戳和密钥
        sign_str = f"{param_str}&timestamp={timestamp}&key={self.secret_key}"
        
        # 生成HMAC签名
        signature = hmac. new(
            self.secret_key.encode(),
            sign_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_signature(
        self,
        data: Dict[str, Any],
        timestamp: int,
        signature: str,
        max_age: int = 300
    ) -> bool:
        """
        验证请求签名
        
        Args: 
            data: 请求数据
            timestamp: 时间戳
            signature: 签名
            max_age: 最大有效期（秒）
            
        Returns:
            是否有效
        """
        import time
        
        # 检查时间戳
        current_time = int(time.time())
        if abs(current_time - timestamp) > max_age:
            return False
        
        # 验证签名
        expected_signature = self.sign_request(data, timestamp)
        return hmac.compare_digest(signature, expected_signature)


# 全局实例
encryption = DataEncryption()
jwt_manager = JWTManager()
rbac_manager = RBACManager()
password_manager = PasswordManager()
