"""Database connection pool management module.

This module provides a DatabasePool class for managing PostgreSQL connections
using asyncpg with connection pooling, transaction support, and health monitoring.
"""

import asyncpg
import logging
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
import asyncio


logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages PostgreSQL database connection pool using asyncpg.
    
    This class provides methods for database operations including:
    - Connection pool initialization and cleanup
    - SQL query execution (execute, fetch, fetchrow, fetchval)
    - Transaction management
    - Health checking and monitoring
    """

    def __init__(self):
        """Initialize DatabasePool instance."""
        self._pool: Optional[asyncpg.Pool] = None
        self._dsn: Optional[str] = None
        self._min_size: int = 10
        self._max_size: int = 20

    async def initialize(
        self,
        dsn: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = 5432,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        min_size: int = 10,
        max_size: int = 20,
        command_timeout: float = 60.0,
        **kwargs
    ) -> None:
        """Initialize the database connection pool.
        
        Args:
            dsn: Database connection string (postgresql://user:pass@host:port/db)
            host: Database host
            port: Database port (default: 5432)
            user: Database user
            password: Database password
            database: Database name
            min_size: Minimum number of connections in pool (default: 10)
            max_size: Maximum number of connections in pool (default: 20)
            command_timeout: Command timeout in seconds (default: 60.0)
            **kwargs: Additional asyncpg connection parameters
            
        Raises:
            Exception: If pool initialization fails
        """
        if self._pool is not None:
            logger.warning("Database pool already initialized")
            return

        try:
            self._min_size = min_size
            self._max_size = max_size
            
            if dsn:
                self._dsn = dsn
                self._pool = await asyncpg.create_pool(
                    dsn=dsn,
                    min_size=min_size,
                    max_size=max_size,
                    command_timeout=command_timeout,
                    **kwargs
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database,
                    min_size=min_size,
                    max_size=max_size,
                    command_timeout=command_timeout,
                    **kwargs
                )
            
            logger.info(
                f"Database pool initialized successfully "
                f"(min_size={min_size}, max_size={max_size})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self) -> None:
        """Close the database connection pool.
        
        Gracefully closes all connections in the pool.
        """
        if self._pool is not None:
            try:
                await self._pool.close()
                logger.info("Database pool closed successfully")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}")
            finally:
                self._pool = None
        else:
            logger.warning("Database pool is not initialized")

    def _check_pool(self) -> None:
        """Check if pool is initialized.
        
        Raises:
            RuntimeError: If pool is not initialized
        """
        if self._pool is None:
            raise RuntimeError(
                "Database pool is not initialized. "
                "Call initialize() first."
            )

    async def execute(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """Execute a SQL command (INSERT, UPDATE, DELETE, etc.).
        
        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Status string from the SQL command (e.g., 'INSERT 0 1')
            
        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.exceptions.*: Various database errors
        """
        self._check_pool()
        try:
            result = await self._pool.execute(query, *args, timeout=timeout)
            return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[asyncpg.Record]:
        """Fetch all rows from a SQL query.
        
        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            List of Record objects representing rows
            
        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.exceptions.*: Various database errors
        """
        self._check_pool()
        try:
            result = await self._pool.fetch(query, *args, timeout=timeout)
            return result
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[asyncpg.Record]:
        """Fetch a single row from a SQL query.
        
        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Record object representing the row, or None if no rows found
            
        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.exceptions.*: Various database errors
        """
        self._check_pool()
        try:
            result = await self._pool.fetchrow(query, *args, timeout=timeout)
            return result
        except Exception as e:
            logger.error(f"Error fetching row: {e}")
            raise

    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """Fetch a single value from a SQL query.
        
        Args:
            query: SQL query string
            *args: Query parameters
            column: Column index to return (default: 0)
            timeout: Query timeout in seconds
            
        Returns:
            Single value from the specified column, or None if no rows found
            
        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.exceptions.*: Various database errors
        """
        self._check_pool()
        try:
            result = await self._pool.fetchval(
                query, *args, column=column, timeout=timeout
            )
            return result
        except Exception as e:
            logger.error(f"Error fetching value: {e}")
            raise

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions.
        
        Usage:
            async with db_pool.transaction():
                await db_pool.execute("INSERT INTO ...")
                await db_pool.execute("UPDATE ...")
        
        The transaction is automatically committed on success or rolled back on error.
        
        Yields:
            asyncpg.connection.Connection: Database connection with active transaction
            
        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.exceptions.*: Various database errors
        """
        self._check_pool()
        
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                try:
                    yield connection
                    logger.debug("Transaction committed successfully")
                except Exception as e:
                    logger.error(f"Transaction rolled back due to error: {e}")
                    raise

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the database connection pool.
        
        Returns:
            Dictionary containing health check results:
                - healthy (bool): Overall health status
                - pool_initialized (bool): Whether pool is initialized
                - connection_test (bool): Whether a test query succeeded
                - error (str, optional): Error message if unhealthy
                
        """
        health_status = {
            "healthy": False,
            "pool_initialized": self._pool is not None,
            "connection_test": False
        }

        if self._pool is None:
            health_status["error"] = "Database pool is not initialized"
            return health_status

        try:
            # Test query to verify connection
            result = await self._pool.fetchval("SELECT 1")
            if result == 1:
                health_status["connection_test"] = True
                health_status["healthy"] = True
        except Exception as e:
            health_status["error"] = f"Connection test failed: {str(e)}"
            logger.error(f"Health check failed: {e}")

        return health_status

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection pool.
        
        Returns:
            Dictionary containing pool statistics:
                - size (int): Current number of connections in the pool
                - min_size (int): Minimum pool size
                - max_size (int): Maximum pool size
                - free_size (int): Number of free connections
                - initialized (bool): Whether pool is initialized
                
        """
        if self._pool is None:
            return {
                "initialized": False,
                "size": 0,
                "min_size": self._min_size,
                "max_size": self._max_size,
                "free_size": 0
            }

        return {
            "initialized": True,
            "size": self._pool.get_size(),
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size(),
            "free_size": self._pool.get_idle_size()
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global database pool instance
db_pool = DatabasePool()


# Convenience functions for backward compatibility
async def get_db_pool() -> DatabasePool:
    """Get the global database pool instance.
    
    Returns:
        DatabasePool: Global database pool instance
    """
    return db_pool


async def initialize_db(dsn: Optional[str] = None, **kwargs) -> None:
    """Initialize the global database pool.
    
    Args:
        dsn: Database connection string
        **kwargs: Additional connection parameters
    """
    await db_pool.initialize(dsn=dsn, **kwargs)


async def close_db() -> None:
    """Close the global database pool."""
    await db_pool.close()
