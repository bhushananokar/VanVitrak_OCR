import asyncio
import logging
from typing import AsyncGenerator, Optional
from pathlib import Path

from sqlalchemy import create_engine, MetaData, event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine

from .config import settings

logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# Note: metadata is available as Base.metadata - no need to create separate instance

# Global variables for database connections
engine: Optional[AsyncEngine] = None
async_session_maker: Optional[async_sessionmaker] = None
sync_engine: Optional[Engine] = None
sync_session_maker: Optional[sessionmaker] = None


def create_sync_engine():
    """Create synchronous database engine"""
    global sync_engine, sync_session_maker
    
    if sync_engine is None:
        logger.info(f"Creating sync engine for: {settings.DATABASE_URL}")
        
        if settings.is_sqlite:
            # SQLite configuration
            sync_engine = create_engine(
                settings.DATABASE_URL,
                echo=settings.DATABASE_ECHO,
                pool_pre_ping=True,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20
                }
            )
        else:
            # PostgreSQL configuration
            sync_engine = create_engine(
                settings.DATABASE_URL,
                echo=settings.DATABASE_ECHO,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=20,
                max_overflow=0
            )
        
        sync_session_maker = sessionmaker(
            bind=sync_engine,
            autocommit=False,
            autoflush=False,
        )
        
        # Setup database-specific configurations
        if settings.is_postgresql:
            setup_postgresql_extensions(sync_engine)
        elif settings.is_sqlite:
            setup_sqlite_extensions(sync_engine)
        
        logger.info("Sync database engine created successfully")
    
    return sync_engine


def create_async_engine_instance():
    """Create asynchronous database engine"""
    global engine, async_session_maker
    
    if engine is None:
        logger.info(f"Creating async engine for: {settings.database_url_async}")
        
        if settings.is_sqlite:
            # SQLite async configuration
            engine = create_async_engine(
                settings.database_url_async,
                echo=settings.DATABASE_ECHO,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20
                }
            )
        else:
            # PostgreSQL async configuration
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
        
        logger.info("Async database engine created successfully")
    
    return engine


def setup_postgresql_extensions(engine: Engine):
    """Setup PostgreSQL specific extensions and configurations"""
    @event.listens_for(engine, "connect")
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


def setup_sqlite_extensions(engine: Engine):
    """Setup SQLite specific configurations"""
    @event.listens_for(engine, "connect")
    def enable_sqlite_features(dbapi_connection, connection_record):
        # Enable foreign key support
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
        
        # Enable WAL mode for better concurrency
        dbapi_connection.execute("PRAGMA journal_mode=WAL")
        
        # Optimize SQLite settings
        dbapi_connection.execute("PRAGMA synchronous=NORMAL")
        dbapi_connection.execute("PRAGMA cache_size=10000")
        dbapi_connection.execute("PRAGMA temp_store=MEMORY")
        
        logger.debug("SQLite optimizations applied")


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


def get_sync_session() -> Session:
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
        logger.info("Initializing database...")
        
        # Create sync engine first
        sync_engine = create_sync_engine()
        
        # Import all models to ensure they're registered
        from app.models.database import (
            Document, OCRResult, ExtractedCoordinate, 
            Claim, ClaimGeometry, ProcessingLog, BatchJob
        )
        
        # Create all tables using sync engine
        Base.metadata.create_all(bind=sync_engine)
        logger.info("Database tables created successfully")
        
        # Test database connection
        with sync_engine.connect() as conn:
            if settings.is_sqlite:
                result = conn.execute(text("SELECT sqlite_version();"))
                version = result.scalar()
                logger.info(f"SQLite version: {version}")
                
                # Count tables
                result = conn.execute(text("SELECT COUNT(*) FROM sqlite_master WHERE type='table';"))
                table_count = result.scalar()
                logger.info(f"Created {table_count} tables")
                
            else:
                result = conn.execute(text("SELECT version();"))
                version = result.scalar()
                logger.info(f"PostgreSQL version: {version[:50]}...")
        
        # Create async engine
        create_async_engine_instance()
        logger.info("Database initialization completed successfully")
        
        # Run post-initialization tasks
        await post_init_tasks()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def post_init_tasks():
    """Run tasks after database initialization"""
    try:
        # Create upload directory
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Upload directory ensured: {upload_dir.absolute()}")
        
        # Additional initialization tasks can be added here
        
    except Exception as e:
        logger.warning(f"Post-init tasks failed: {e}")


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


async def check_database_health() -> bool:
    """Check if database is accessible"""
    try:
        if engine is None:
            create_async_engine_instance()
        
        async with async_session_maker() as session:
            if settings.is_sqlite:
                result = await session.execute(text("SELECT 1"))
            else:
                result = await session.execute(text("SELECT 1"))
            
            return result.scalar() == 1
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


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
        async with await self.get_session() as session:
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


# Testing utilities
class TestDatabase:
    """Database utilities for testing"""
    
    @staticmethod
    def create_test_engine():
        """Create test database engine"""
        test_url = "sqlite:///./test_fra_ocr.db"
        
        if "sqlite" in test_url:
            test_engine = create_async_engine(
                test_url.replace("sqlite:///", "sqlite+aiosqlite:///"),
                echo=True,
                connect_args={"check_same_thread": False}
            )
        else:
            test_engine = create_async_engine(
                test_url,
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


# Database utilities
async def execute_raw_sql(sql: str, params: dict = None):
    """Execute raw SQL query"""
    if async_session_maker is None:
        create_async_engine_instance()
    
    async with async_session_maker() as session:
        try:
            result = await session.execute(text(sql), params or {})
            await session.commit()
            return result
        except Exception as e:
            await session.rollback()
            logger.error(f"Raw SQL execution failed: {e}")
            raise


async def get_database_info():
    """Get database information"""
    try:
        if async_session_maker is None:
            create_async_engine_instance()
        
        async with async_session_maker() as session:
            if settings.is_sqlite:
                # SQLite version
                result = await session.execute(text("SELECT sqlite_version();"))
                version = result.scalar()
                
                # Table count
                result = await session.execute(
                    text("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
                )
                table_count = result.scalar()
                
                return {
                    "database_type": "SQLite",
                    "version": version,
                    "table_count": table_count,
                    "file_path": settings.DATABASE_URL.replace("sqlite:///", "")
                }
            else:
                # PostgreSQL version
                result = await session.execute(text("SELECT version();"))
                version = result.scalar()
                
                # PostGIS version
                try:
                    result = await session.execute(text("SELECT PostGIS_Version();"))
                    postgis_info = result.scalar()
                except:
                    postgis_info = "Not installed"
                
                return {
                    "database_type": "PostgreSQL",
                    "postgresql_version": version,
                    "postgis_version": postgis_info,
                    "connection_url": settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "local"
                }
                
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {"error": str(e)}


def get_database_url() -> str:
    """Get the current database URL (safe for logging)"""
    url = settings.DATABASE_URL
    if "@" in url:
        # Hide password in URL
        parts = url.split("@")
        user_pass = parts[0].split("://")[1]
        if ":" in user_pass:
            user = user_pass.split(":")[0] 
            return f"postgresql://{user}:***@{parts[1]}"
    return url


def reset_database():
    """Reset database (for testing/development)"""
    global engine, sync_engine, async_session_maker, sync_session_maker
    
    engine = None
    sync_engine = None
    async_session_maker = None
    sync_session_maker = None
    
    logger.info("Database connections reset")


# Export commonly used items
__all__ = [
    "Base",
    "get_async_session",
    "get_sync_session", 
    "init_database",
    "close_database",
    "db_manager",
    "check_database_health",
    "TestDatabase",
    "execute_raw_sql",
    "get_database_info",
    "get_database_url",
    "reset_database"
]