from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from loguru import logger

from config import DB_URL
from models.schema import Base


IS_SQLITE = DB_URL.startswith("sqlite")

if IS_SQLITE:
    engine = create_engine(
        DB_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # ~64MB page cache
        cursor.close()
else:
    # Postgres (e.g. Supabase) — pre-ping avoids stale-connection errors
    # after the pooler closes idle connections.
    engine = create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False,
    )


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialised at {DB_URL}")


@contextmanager
def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
