import asyncio
from typing import AsyncGenerator
from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# Metadata for table creation
metadata = MetaData()

# Global variables for database connections
engine: AsyncEngine = None
async_session_maker: async_sessionmaker = None
sync_engine = None
sync_session_maker = None


def create_sync_engine():
    """Create synchronous database engine for migrations and initial setup"""
    global sync_engine, sync_session_maker
    
    if sync_engine is None:
        sync_engine = create_engine(
            settings.DATABASE_URL,
            echo=settings.DATABASE_ECHO,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        
        sync_session_maker = sessionmaker(
            bind=sync_engine,
            autocommit=False,
            autoflush=False,
        )
        
        # Enable PostGIS extension
        @event.listens_for(sync_engine, "connect")
        def enable_postgis(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cursor:
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology;")
                    dbapi_connection.commit()
                    logger.info("PostGIS extensions enabled")
                except Exception as e:
                    logger.warning(f"Could not enable PostGIS extensions: {e}")
                    dbapi_connection.rollback()
    
    return sync_engine


def create_async_engine_instance():
    """Create asynchronous database engine"""
    global engine, async_session_maker
    
    if engine is None:
        engine = create_async_engine(
            settings.database_url_async,
            echo=settings.DATABASE_ECHO,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=20,
            max_overflow=0,
        )
        
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        logger.info("Async database engine created")
    
    return engine


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get async database session.
    Use this in FastAPI endpoints.
    """
    if async_session_maker is None:
        create_async_engine_instance()
    
    async with async_session_maker() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """
    Get synchronous database session.
    Use this for migrations and setup.
    """
    if sync_session_maker is None:
        create_sync_engine()
    
    return sync_session_maker()


async def init_database():
    """Initialize database tables and extensions"""
    try:
        # Create sync engine first to enable extensions
        sync_engine = create_sync_engine()
        
        # Create all tables using sync engine
        Base.metadata.create_all(bind=sync_engine)
        logger.info("Database tables created successfully")
        
        # Create async engine
        create_async_engine_instance()
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_database():
    """Close database connections"""
    global engine, sync_engine
    
    try:
        if engine:
            await engine.dispose()
            logger.info("Async database engine disposed")
        
        if sync_engine:
            sync_engine.dispose()
            logger.info("Sync database engine disposed")
            
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


class DatabaseManager:
    """Database manager for handling connections and transactions"""
    
    def __init__(self):
        self.engine = None
        self.session_maker = None
    
    async def init(self):
        """Initialize database manager"""
        self.engine = create_async_engine_instance()
        self.session_maker = async_session_maker
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if self.session_maker is None:
            await self.init()
        
        return self.session_maker()
    
    async def execute_transaction(self, operation):
        """Execute operation within a transaction"""
        async with self.get_session() as session:
            try:
                result = await operation(session)
                await session.commit()
                return result
            except Exception as e:
                await session.rollback()
                logger.error(f"Transaction failed: {e}")
                raise


# Global database manager instance
db_manager = DatabaseManager()


# Database health check
async def check_database_health() -> bool:
    """Check if database is accessible"""
    try:
        if engine is None:
            create_async_engine_instance()
        
        async with async_session_maker() as session:
            result = await session.execute("SELECT 1")
            return result.scalar() == 1
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Testing utilities
class TestDatabase:
    """Database utilities for testing"""
    
    @staticmethod
    def create_test_engine():
        """Create test database engine"""
        test_engine = create_async_engine(
            "postgresql+asyncpg://test:test@localhost:5432/test_fra_ocr_db",
            echo=True,
            poolclass=StaticPool,
            pool_pre_ping=True,
        )
        return test_engine
    
    @staticmethod
    async def create_test_tables(engine):
        """Create tables for testing"""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    @staticmethod
    async def drop_test_tables(engine):
        """Drop tables after testing"""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# Connection event handlers
@event.listens_for(engine, "connect", once=True)
def enable_postgis_async(dbapi_connection, connection_record):
    """Enable PostGIS for async connections"""
    try:
        with dbapi_connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology;")
            dbapi_connection.commit()
            logger.info("PostGIS extensions enabled for async connection")
    except Exception as e:
        logger.warning(f"Could not enable PostGIS for async connection: {e}")
        dbapi_connection.rollback()


# Database utilities
async def execute_raw_sql(sql: str, params: dict = None):
    """Execute raw SQL query"""
    async with async_session_maker() as session:
        try:
            result = await session.execute(sql, params or {})
            await session.commit()
            return result
        except Exception as e:
            await session.rollback()
            logger.error(f"Raw SQL execution failed: {e}")
            raise


async def get_database_info():
    """Get database information"""
    try:
        async with async_session_maker() as session:
            # Check PostgreSQL version
            pg_version = await session.execute("SELECT version();")
            
            # Check PostGIS version
            try:
                postgis_version = await session.execute("SELECT PostGIS_Version();")
                postgis_info = postgis_version.scalar()
            except:
                postgis_info = "Not installed"
            
            return {
                "postgresql_version": pg_version.scalar(),
                "postgis_version": postgis_info,
                "connection_url": settings.DATABASE_URL.replace(
                    settings.DATABASE_URL.split("@")[0].split("://")[1], "***"
                )
            }
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {"error": str(e)}


# Export commonly used items
__all__ = [
    "Base",
    "metadata",
    "get_async_session",
    "get_sync_session",
    "init_database",
    "close_database",
    "db_manager",
    "check_database_health",
    "TestDatabase",
    "execute_raw_sql",
    "get_database_info"
]