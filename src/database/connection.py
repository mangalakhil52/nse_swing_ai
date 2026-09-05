"""
Database Connection and Session Factory Module.
Provides robust connection management using standard built-in SQLite or PostgreSQL.
"""

from contextlib import contextmanager
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, scoped_session

from config.settings import settings
from src.database.schema import Base

logger = logging.getLogger(__name__)

# Normalize URL for standard sqlite if async prefix was provided
db_url = settings.DATABASE_URL
if db_url.startswith("sqlite+aiosqlite://"):
    db_url = db_url.replace("sqlite+aiosqlite://", "sqlite://")
elif db_url.startswith("postgresql+asyncpg://"):
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

# Create engine with thread-safe settings for SQLite
connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
engine = create_engine(
    db_url,
    echo=False,
    future=True,
    connect_args=connect_args,
    pool_pre_ping=True,
)

# Thread-local session factory
session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
ScopedSession = scoped_session(session_factory)


def init_db() -> None:
    """Initializes all database tables defined in SQLAlchemy schema and handles column migrations."""
    try:
        Base.metadata.create_all(bind=engine)
        # Migrate SQLite schema columns if table exists without new target/stop columns
        if db_url.startswith("sqlite"):
            from sqlalchemy import text
            with engine.connect() as conn:
                res = conn.execute(text("PRAGMA table_info(shadow_trades)"))
                existing_cols = {row[1] for row in res.fetchall()}
                new_cols = {
                    "stop_loss": "FLOAT",
                    "target_1": "FLOAT",
                    "target_2": "FLOAT",
                    "target_3": "FLOAT",
                    "t1_hit": "BOOLEAN DEFAULT 0",
                    "t2_hit": "BOOLEAN DEFAULT 0",
                    "t3_hit": "BOOLEAN DEFAULT 0",
                    "position_size_shares": "INTEGER",
                }
                for col_name, col_type in new_cols.items():
                    if col_name not in existing_cols:
                        try:
                            conn.execute(text(f"ALTER TABLE shadow_trades ADD COLUMN {col_name} {col_type}"))
                        except Exception:
                            pass
                conn.commit()
        logger.info(f"Database schema initialized successfully using {db_url.split('://')[0]}.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager providing a transactional database session."""
    session = ScopedSession()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rolled back due to error: {e}")
        raise
    finally:
        session.close()
