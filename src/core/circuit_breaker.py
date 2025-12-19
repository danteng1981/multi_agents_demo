"""
Database Connection Pool Management
"""
import logging
from typing import Optional, List, Dict, Any

import asyncpg

from src.core.config import settings

logger = logging.getLogger(__name__)


class DatabasePool:
    """数据库连接池管理器"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(
        self,
        dsn: str,
        min_size: int = 10,
        max_size: int = 50,
        timeout: int = 30
    ):
        """
        初始化连接池
        
        Args: 
            dsn: 数据库连接字符串
            min_size: 最小连接数
            max_size: 最大连接数
            timeout:  连接超时时间
        """
        try:
            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=min_size,
                max_size=max_size,
                command_timeout=timeout,
                max_queries=50000,
                max_inactive_connection_lifetime=300
            )
            
            # 测试连接
            async with self.pool.acquire() as connection:
                await connection.fetchval("SELECT 1")
            
            logger.info(f"✅ Database pool initialized (min={min_size}, max={max_size})")
            
        except Exception as e:
            logger. error(f"❌ Database pool initialization failed: {e}")
            raise
    
    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
    
    async def execute(self, query: str, *args) -> str:
        """
        执行SQL命令（INSERT, UPDATE, DELETE）
        
        Args:
            query: SQL查询
            *args: 查询参数
            
        Returns:
            执行结果
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """
        查询多行数据
        
        Args:
            query: SQL查询
            *args: 查询参数
            
        Returns: 
            查询结果列表
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool. acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        查询单行数据
        
        Args:
            query:  SQL查询
            *args:  查询参数
            
        Returns:
            查询结果
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """
        查询单个值
        
        Args: 
            query: SQL查询
            *args: 查询参数
            
        Returns:
            查询结果值
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool. acquire() as connection:
            return await connection.fetchval(query, *args)
    
    async def transaction(self):
        """
        获取事务上下文
        
        Usage:
            async with db_pool.transaction() as conn:
                await conn.execute("INSERT ...")
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        connection = await self.pool.acquire()
        transaction = connection.transaction()
        
        try:
            await transaction.start()
            yield connection
            await transaction.commit()
        except Exception: 
            await transaction.rollback()
            raise
        finally: 
            await self.pool.release(connection)
    
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            是否健康
        """
        try:
            if not self.pool:
                return False
            
            async with self.pool.acquire() as connection:
                result = await connection.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """
        获取连接池统计信息
        
        Returns: 
            连接池统计
        """
        if not self. pool:
            return {"status": "not_initialized"}
        
        return {
            "size": self. pool.get_size(),
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool. get_max_size(),
            "free_connections": self.pool.get_idle_size(),
        }


# 全局数据库连接池实例
db_pool = DatabasePool()
