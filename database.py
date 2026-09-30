from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from config import DB_PATH
from models import Base


def _make_engine():
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


engine = _make_engine()
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    # Ensure newly added columns exist in existing SQLite databases
    with engine.connect() as conn:
        for table, col, col_type, default in [
            ("manuals", "allowed_exits", "INTEGER", "3"),
            ("manuals", "question_count", "INTEGER", "5"),
            ("viva_sessions", "allowed_exits", "INTEGER", "3"),
            ("viva_sessions", "exit_count", "INTEGER", "0"),
            ("viva_sessions", "exit_violations", "TEXT", "'[]'"),
            ("viva_sessions", "terminated_reason", "VARCHAR(255)", "NULL"),
        ]:
            try:
                conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {default}")
                )
                conn.commit()
            except Exception:
                pass


@contextmanager
def session_scope() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
