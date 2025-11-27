"""
Database connection and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from contextlib import contextmanager, asynccontextmanager
from typing import AsyncGenerator, Generator
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Create base class for ORM models
Base = declarative_base()

# Convert database URL for async if needed
def get_async_database_url() -> str:
    """Convert database URL to async version if needed."""
    db_url = settings.database_url
    
    # Convert postgresql:// to postgresql+asyncpg://
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+asyncpg://")
    # Convert sqlite:// to sqlite+aiosqlite://
    elif db_url.startswith("sqlite://"):
        return db_url.replace("sqlite://", "sqlite+aiosqlite://")
    
    return db_url

# Create async engine
async_engine = create_async_engine(
    get_async_database_url(),
    echo=settings.debug,
    pool_size=settings.connection_pool_size,
    max_overflow=10,
    pool_pre_ping=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create sync engine for migrations and admin tasks
sync_engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=settings.connection_pool_size,
    max_overflow=10,
    pool_pre_ping=True,
)

# Create sync session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Get a sync database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Database session error", error=str(e))
        raise
    finally:
        session.close()


async def init_database():
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def close_database():
    """Close database connections."""
    await async_engine.dispose()
    logger.info("Database connections closed")
