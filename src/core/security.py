"""
Security Module - Encryption, Authentication, Authorization
"""
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Set
from enum import Enum

from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf. pbkdf2 import PBKDF2

from src.core.config import settings


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Permission(Enum):
    """Permission definitions"""
    # Agent permissions
    READ_AGENT = "read: agent"
    WRITE_AGENT = "write:agent"
    DELETE_AGENT = "delete:agent"
    ADMIN_AGENT = "admin: agent"
    
    # Data permissions
    READ_DATA = "read:data"
    WRITE_DATA = "write:data"
    EXPORT_DATA = "export:data"
    
    # System permissions
    VIEW_AUDIT_LOG = "view:audit_log"
    MANAGE_USERS = "manage:users"
    ADMIN_SYSTEM = "admin:system"


class Role(Enum):
    """Role definitions"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"


class DataEncryption:
    """Data encryption manager"""
    
    def __init__(self):
        """Initialize encryptor"""
        key = settings.ENCRYPTION_KEY. encode()
        if len(key) != 32:
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'enterprise-agent',
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key))
        else:
            key = base64.urlsafe_b64encode(key)
        
        self.fernet = Fernet(key)
    
    def encrypt_field(self, data: str) -> str:
        """Field-level encryption"""
        if not data:
            return data
        
        encrypted = self.fernet.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_field(self, encrypted_data: str) -> str:
        """Field-level decryption"""
        if not encrypted_data:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data. encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def hash_sensitive_data(self, data: str) -> str:
        """One-way hash for sensitive data"""
        return hashlib.sha256(data.encode()).hexdigest()


class JWTManager:
    """JWT token manager"""
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create access token"""
        to_encode = data. copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(seconds=settings.JWT_EXPIRATION)
        
        to_encode. update({
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
        """Verify token"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")
    
    @staticmethod
    def create_refresh_token(user_id: str) -> str:
        """Create refresh token"""
        data = {"sub": user_id, "type": "refresh"}
        expire = datetime.utcnow() + timedelta(days=7)
        data.update({"exp": expire})
        
        return jwt.encode(
            data,
            settings.SECRET_KEY,
            algorithm=settings. JWT_ALGORITHM
        )


class RBACManager:
    """Role-Based Access Control manager"""
    
    ROLE_PERMISSIONS = {
        Role.ADMIN: {
            Permission.READ_AGENT, Permission.WRITE_AGENT, Permission.DELETE_AGENT,
            Permission.ADMIN_AGENT, Permission.READ_DATA, Permission.WRITE_DATA,
            Permission.EXPORT_DATA, Permission.VIEW_AUDIT_LOG, Permission.MANAGE_USERS,
            Permission.ADMIN_SYSTEM,
        },
        Role. DEVELOPER: {
            Permission.READ_AGENT, Permission.WRITE_AGENT,
            Permission.READ_DATA, Permission.WRITE_DATA, Permission.VIEW_AUDIT_LOG,
        },
        Role.OPERATOR:  {
            Permission.READ_AGENT, Permission.READ_DATA, Permission.VIEW_AUDIT_LOG,
        },
        Role.VIEWER:  {
            Permission.READ_AGENT, Permission.READ_DATA,
        },
        Role. GUEST: {
            Permission.READ_AGENT,
        }
    }
    
    @classmethod
    def get_user_permissions(cls, user_roles: list) -> Set[Permission]:
        """Get all permissions for user roles"""
        permissions = set()
        for role in user_roles: 
            if isinstance(role, str):
                try:
                    role = Role(role)
                except ValueError:
                    continue
            permissions.update(cls.ROLE_PERMISSIONS.get(role, set()))
        return permissions
    
    @classmethod
    def has_permission(cls, user_roles: list, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        user_permissions = cls.get_user_permissions(user_roles)
        return required_permission in user_permissions


class PasswordManager:
    """Password management"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)


# Global instances
encryption = DataEncryption()
jwt_manager = JWTManager()
rbac_manager = RBACManager()
password_manager = PasswordManager()
