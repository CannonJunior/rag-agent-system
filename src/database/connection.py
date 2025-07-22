from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from loguru import logger
from config.settings import settings
from .models import Base


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self):
        self.sync_engine = None
        self.async_engine = None
        self.sync_session_factory = None
        self.async_session_factory = None
        self._initialized = False
    
    def initialize_sync(self):
        """Initialize synchronous database connections."""
        db_url = (
            f"postgresql://{settings.database.username}:{settings.database.password}"
            f"@{settings.database.host}:{settings.database.port}/{settings.database.database}"
        )
        
        self.sync_engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            echo=settings.database.echo,
        )
        
        self.sync_session_factory = sessionmaker(
            bind=self.sync_engine,
            autocommit=False,
            autoflush=False,
        )
        
        logger.info("Synchronous database connection initialized")
    
    def initialize_async(self):
        """Initialize asynchronous database connections."""
        db_url = (
            f"postgresql+asyncpg://{settings.database.username}:{settings.database.password}"
            f"@{settings.database.host}:{settings.database.port}/{settings.database.database}"
        )
        
        self.async_engine = create_async_engine(
            db_url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            echo=settings.database.echo,
        )
        
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
        )
        
        logger.info("Asynchronous database connection initialized")
    
    def initialize_all(self):
        """Initialize both sync and async connections."""
        if self._initialized:
            return
        
        self.initialize_sync()
        self.initialize_async()
        self._initialized = True
        logger.info("Database manager fully initialized")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session context manager."""
        if not self.async_session_factory:
            raise RuntimeError("Async database not initialized")
        
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_sync_session(self) -> AsyncGenerator[Session, None]:
        """Get sync database session context manager."""
        if not self.sync_session_factory:
            raise RuntimeError("Sync database not initialized")
        
        session = self.sync_session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables."""
        if not self.sync_engine:
            raise RuntimeError("Sync database not initialized")
        
        Base.metadata.create_all(bind=self.sync_engine)
        logger.info("Database tables created")
    
    async def check_connection(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    async def check_vector_extension(self) -> bool:
        """Check if pgvector extension is available."""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                )
                return result.scalar() is not None
        except Exception as e:
            logger.error(f"Vector extension check failed: {e}")
            return False
    
    async def close(self):
        """Close all database connections."""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.sync_engine:
            self.sync_engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()