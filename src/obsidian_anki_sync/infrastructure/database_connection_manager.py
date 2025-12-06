"""Database connection management for SQLite database.

Handles thread-local connections, WAL mode configuration, and connection pooling.
"""

import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseConnectionManager:
    """Manages SQLite database connections with thread-local storage.

    Thread Safety:
        This class is thread-safe. Each thread gets its own SQLite connection
        via thread-local storage. The database uses WAL mode for concurrent
        access. All connections are tracked and properly closed on exit.

    WAL Mode:
        The database uses WAL (Write-Ahead Logging) mode for better concurrency.
        WAL mode allows concurrent reads while writes are in progress.

    Usage:
        manager = DatabaseConnectionManager(db_path)
        with manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
    """

    def __init__(self, db_path: Path):
        """Initialize database connection manager.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()

    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection.

        Creates a new connection for the current thread if one doesn't exist.
        Each connection is configured with WAL mode and tracked for cleanup.

        Returns:
            Thread-local SQLite connection
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._connections_lock:
                self._connections.append(conn)
            logger.debug(
                "db_connection_created",
                thread_id=threading.get_ident(),
                total_connections=len(self._connections),
                db_path=str(self._db_path),
            )
        return self._local.conn  # type: ignore[no-any-return]

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection]:
        """Execute operations within a transaction with automatic rollback on error.

        This context manager provides ACID guarantees for database operations:
        - Auto-commits on successful exit
        - Auto-rolls back on any exception
        - Re-raises the original exception after rollback

        Usage:
            with manager.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO ...")
                cursor.execute("UPDATE ...")
            # Commits automatically if no exception

        Yields:
            SQLite connection for the current thread
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute_query(
        self, query: str, params: tuple = (), operation: str = "query"
    ) -> sqlite3.Cursor:
        """Execute a database query with logging.

        Args:
            query: SQL query string
            params: Query parameters
            operation: Operation name for logging

        Returns:
            Cursor with results
        """
        import time

        start_time = time.time()
        try:
            conn = self.get_connection()
            cursor = conn.execute(query, params)
            duration = time.time() - start_time

            # Log slow queries (>100ms)
            if duration > 0.1:
                logger.warning(
                    "db_slow_query",
                    operation=operation,
                    duration=round(duration, 3),
                    params_count=len(params),
                    query_preview=query[:100],
                )
            else:
                logger.debug(
                    "db_query",
                    operation=operation,
                    duration=round(duration, 4),
                    params_count=len(params),
                )

            return cursor
        except sqlite3.Error as e:
            duration = time.time() - start_time
            logger.error(
                "db_query_error",
                operation=operation,
                duration=round(duration, 3),
                error=str(e),
                error_type=type(e).__name__,
                query_preview=query[:100],
            )
            raise

    def close(self) -> None:
        """Close all thread-local database connections.

        This method properly closes all connections that were created across
        different threads. It's safe to call multiple times.
        """
        with self._connections_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(
                        "error_closing_connection",
                        error=str(e),
                    )
            self._connections.clear()

        # Clear thread-local connection for current thread
        if hasattr(self._local, "conn"):
            self._local.conn = None

        logger.debug(
            "closed_all_connections",
            thread_id=threading.get_ident(),
        )

