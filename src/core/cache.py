"""
Multi-Level Cache System
"""
import asyncio
import hashlib
import json
import logging
from functools import wraps
from typing import Optional, Callable, Any
from collections import OrderedDict

import redis.asyncio as redis

from src.core.config import settings

logger = logging.getLogger(__name__)


class LRUCache:
    """本地LRU缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key not in self. cache:
            return None
        
        # 移到末尾（最近使用）
        self.cache.move_to_end(key)
        return self. cache[key]
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        if key in self. cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # 超过最大容量，删除最久未使用的
        if len(self.cache) > self.max_size:
            self. cache.popitem(last=False)
    
    def delete(self, key: str):
        """删除缓存"""
        self.cache.pop(key, None)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def size(self) -> int:
        """缓存大小"""
        return len(self.cache)


class MultiLevelCache:
    """多级缓存系统"""
    
    def __init__(self):
        # L1: 本地内存缓存
        self. local_cache = LRUCache(max_size=settings. CACHE_MAX_SIZE)
        
        # L2: Redis缓存
        self.redis_client:  Optional[redis.Redis] = None
        
        self.enabled = settings.CACHE_ENABLED
    
    async def initialize(self, redis_url: str):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.REDIS_MAX_CONNECTIONS
            )
            
            # 测试连接
            await self. redis_client.ping()
            logger.info("✅ Redis cache initialized")
            
        except Exception as e: 
            logger.error(f"❌ Redis initialization failed: {e}")
            self.redis_client = None
    
    async def close(self):
        """关闭Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    def _generate_key(self, func_name: str, args, kwargs) -> str:
        """生成缓存键"""
        key_data = {
            "func":  func_name,
            "args": str(args),
            "kwargs":  str(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(
        self,
        key: str,
        use_local:  bool = True,
        use_redis: bool = True
    ) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key:  缓存键
            use_local: 是否使用本地缓存
            use_redis: 是否使用Redis缓存
            
        Returns: 
            缓存值
        """
        if not self.enabled:
            return None
        
        # L1: 本地缓存
        if use_local: 
            value = self.local_cache.get(key)
            if value is not None:
                logger.debug(f"L1 Cache hit: {key}")
                return value
        
        # L2: Redis缓存
        if use_redis and self.redis_client:
            try:
                value = await self. redis_client.get(key)
                if value is not None: 
                    logger.debug(f"L2 Cache hit: {key}")
                    
                    # 解析JSON
                    value = json.loads(value)
                    
                    # 回写到L1
                    if use_local:
                        self.local_cache.set(key, value)
                    
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
        use_local: bool = True,
        use_redis: bool = True
    ):
        """
        设置缓存
        
        Args: 
            key: 缓存键
            value: 缓存值
            ttl:  过期时间（秒）
            use_local: 是否使用本地缓存
            use_redis: 是否使用Redis缓存
        """
        if not self.enabled:
            return
        
        ttl = ttl or settings.CACHE_TTL
        
        # L1: 本地缓存
        if use_local:
            self. local_cache.set(key, value)
        
        # L2: Redis缓存
        if use_redis and self.redis_client:
            try:
                # 序列化为JSON
                value_json = json.dumps(value)
                await self.redis_client.setex(key, ttl, value_json)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
    
    async def delete(self, key:  str):
        """删除缓存"""
        # L1
        self.local_cache.delete(key)
        
        # L2
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
    
    async def clear(self, pattern: str = "*"):
        """清空缓存"""
        # L1
        self.local_cache.clear()
        
        # L2
        if self.redis_client:
            try:
                keys = await self.redis_client. keys(pattern)
                if keys:
                    await self. redis_client.delete(*keys)
            except Exception as e: 
                logger.error(f"Redis clear error: {e}")
    
    def cache(
        self,
        ttl: int = None,
        use_local: bool = True,
        use_redis: bool = True,
        key_prefix: str = ""
    ):
        """
        缓存装饰器
        
        Args:
            ttl:  过期时间
            use_local: 是否使用本地缓存
            use_redis: 是否使用Redis缓存
            key_prefix: 键前缀
        """
        def decorator(func:  Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{key_prefix}:{self._generate_key(func.__name__, args, kwargs)}"
                
                # 尝试从缓存获取
                cached_value = await self.get(cache_key, use_local, use_redis)
                if cached_value is not None:
                    return cached_value
                
                # 执行函数
                result = await func(*args, **kwargs)
                
                # 写入缓存
                await self.set(cache_key, result, ttl, use_local, use_redis)
                
                return result
            
            return wrapper
        return decorator
    
    async def get_stats(self) -> dict:
        """获取缓存统计"""
        stats = {
            "l1_size": self.local_cache.size(),
            "l1_max_size": self.local_cache.max_size,
        }
        
        if self.redis_client:
            try:
                info = await self.redis_client. info("stats")
                stats.update({
                    "l2_hits": info.get("keyspace_hits", 0),
                    "l2_misses": info.get("keyspace_misses", 0),
                })
            except Exception as e: 
                logger.error(f"Get Redis stats error: {e}")
        
        return stats


# 全局缓存管理器实例
cache_manager = MultiLevelCache()
