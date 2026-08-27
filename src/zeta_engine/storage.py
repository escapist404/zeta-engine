import json
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing
from pathlib import Path


class Storage:
    def __init__(
        self,
        *,
        document_db: str | Path | None = None,
        queue_db: str | Path | None = None,
        index_db: str | Path | None = None,
    ):
        if document_db is None and queue_db is None and index_db is None:
            raise ValueError("至少需要提供一个数据库")

        self._document_path = Path(document_db) if document_db is not None else None
        self._queue_path = Path(queue_db) if queue_db is not None else None
        self._index_path = Path(index_db) if index_db is not None else None
        self._stack: ExitStack | None = None

        self.documents: _Document | None = None
        self.queue: _Queue | None = None
        self.index: _Index | None = None

    def __enter__(self) -> "Storage":
        if self._stack is not None:
            raise RuntimeError("Storage 已经打开")

        self._stack = ExitStack()

        try:
            if self._document_path is not None:
                self.documents = _Document(
                    self._open(self._document_path)
                )
                self.documents.initialize()

            if self._queue_path is not None:
                self.queue = _Queue(
                    self._open(self._queue_path)
                )
                self.queue.initialize()

            if self._index_path is not None:
                self.index = _Index(
                    self._open(self._index_path)
                )
                self.index.initialize()

            return self
        except Exception:
            self._stack.close()
            self._stack = None
            self.documents = None
            self.queue = None
            self.index = None
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stack is None:
            return

        stack = self._stack
        self._stack = None
        self.documents = None
        self.queue = None
        self.index = None
        stack.__exit__(exc_type, exc, tb)

    def _open(self, path: Path) -> sqlite3.Connection:
        if self._stack is None:
            raise RuntimeError("Storage 尚未打开")

        path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(path)
        connection.execute("PRAGMA foreign_keys = ON")

        return self._stack.enter_context(closing(connection))


class _Document:
    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def initialize(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def save(
        self,
        *,
        url: str,
        title: str,
        text: str,
        fetched_at: str,
    ) -> int:
        row = self._connection.execute(
            """
            INSERT INTO documents (url, title, text, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                text = excluded.text,
                fetched_at = excluded.fetched_at
            RETURNING id
            """,
            (
                url,
                title,
                text,
                fetched_at,
            ),
        ).fetchone()

        self._connection.commit()
        return row[0]

    def get(self, document_id: int) -> tuple[str, str, str, str] | None:
        row = self._connection.execute(
            """
            SELECT url, title, text, fetched_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,)
        ).fetchone()
        return row

    def iter_all(self) -> Iterator[tuple[int, str, str, str, str]]:
        return iter(self._connection.execute(
            """
            SELECT id, url, title, text, fetched_at
            FROM documents
            ORDER BY id
            """
        ))

    def count_by_host(self, host: str) -> int:
        return self._connection.execute(
            """
            SELECT COUNT(*)
            FROM documents
            WHERE url LIKE ? OR url LIKE ?
            """,
            (f"http://{host}/%", f"https://{host}/%"),
        ).fetchone()[0]


class _Queue:
    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def initialize(self):
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS crawl_tasks (
                url TEXT PRIMARY KEY,
                state TEXT NOT NULL DEFAULT 'pending'
                    CHECK (state IN ('pending', 'processing', 'done', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );

            CREATE INDEX IF NOT EXISTS crawl_tasks_state_idx
            ON crawl_tasks(state);
            """
        )
        self._connection.commit()

    def enqueue(self, urls: list[str] | tuple[str, ...], ) -> list[str]:
        added = []
        for url in urls:
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO crawl_tasks (url) VALUES (?)",
                (url,),
            )
            if cursor.rowcount:
                added.append(url)

        self._connection.commit()
        return added

    def recover(self) -> int:
        cursor = self._connection.execute(
            """
            UPDATE crawl_tasks
            SET state = 'pending'
            WHERE state = 'processing'
            """
        )
        self._connection.commit()
        return cursor.rowcount

    def claim_pending(self, limit: int) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT url
            FROM crawl_tasks
            WHERE state = 'pending'
            ORDER BY rowid
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        urls = [row[0] for row in rows]

        self._connection.executemany(
            """
            UPDATE crawl_tasks
            SET state = 'processing', attempts = attempts + 1, last_error = NULL
            WHERE url = ? AND state = 'pending'
            """,
            ((url,) for url in urls),
        )
        self._connection.commit()
        return urls

    def claim(self, url: str) -> bool:
        cursor = self._connection.execute(
            """
            UPDATE crawl_tasks
            SET state = 'processing', attempts = attempts + 1, last_error = NULL
            WHERE url = ? AND state = 'pending'
            """,
            (url,),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def complete(self, url: str):
        self._connection.execute(
            """
            UPDATE crawl_tasks
            SET state = 'done', last_error = NULL
            WHERE url = ?
            """,
            (url,),
        )
        self._connection.commit()

    def fail(self, url: str, error: str, max_attempts: int) -> str:
        self._connection.execute(
            """
            UPDATE crawl_tasks
            SET state = CASE
                    WHEN attempts < ? THEN 'pending'
                    ELSE 'failed'
                END,
                last_error = ?
            WHERE url = ?
            """,
            (max_attempts, error, url),
        )
        state = self._connection.execute(
            "SELECT state FROM crawl_tasks WHERE url = ?",
            (url,),
        ).fetchone()[0]
        self._connection.commit()
        return state

    def count_by_state(self) -> dict[str, int]:
        return dict(
            self._connection.execute(
                "SELECT state, COUNT(*) FROM crawl_tasks GROUP BY state"
            ).fetchall()
        )


class _Index:
    TITLE, TEXT = 0, 1

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS indexed_documents (
                document_id INTEGER PRIMARY KEY,
                title_norm TEXT NOT NULL,
                text_norm TEXT NOT NULL,
                title_len INTEGER NOT NULL,
                text_len INTEGER NOT NULL,
                fetched_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL UNIQUE,
                document_frequency INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS postings (
                term_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                field INTEGER NOT NULL CHECK (field IN (0, 1)),
                positions TEXT NOT NULL,
                PRIMARY KEY (term_id, document_id, field)
            ) WITHOUT ROWID;

            CREATE INDEX IF NOT EXISTS postings_document_idx
            ON postings(document_id);
            """
        )
        self._connection.commit()

    def replace_document(
        self,
        *,
        document_id: int,
        fetched_at: str,
        title_norm: str,
        text_norm: str,
        postings: dict[tuple[str, int], list[int]],
    ) -> None:
        with self._connection:
            affected_term_ids = {
                row[0]
                for row in self._connection.execute(
                    """
                    SELECT DISTINCT term_id
                    FROM postings
                    WHERE document_id = ?
                    """,
                    (document_id,),
                )
            }

            self._connection.execute(
                "DELETE FROM postings WHERE document_id = ?",
                (document_id,),
            )

            title_len = sum(
                len(positions)
                for (_, field), positions in postings.items()
                if field == self.TITLE
            )

            text_len = sum(
                len(positions)
                for (_, field), positions in postings.items()
                if field == self.TEXT
            )

            self._connection.execute(
                """
                INSERT INTO indexed_documents (
                    document_id,
                    title_norm,
                    text_norm,
                    title_len,
                    text_len,
                    fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title_norm = excluded.title_norm,
                    text_norm = excluded.text_norm,
                    title_len = excluded.title_len,
                    text_len = excluded.text_len,
                    fetched_at = excluded.fetched_at
                """,
                (
                    document_id,
                    title_norm,
                    text_norm,
                    title_len,
                    text_len,
                    fetched_at,
                ),
            )

            for (term, field), positions in postings.items():
                self._connection.execute(
                    "INSERT OR IGNORE INTO terms (term) VALUES (?)",
                    (term,),
                )
                term_id = self._connection.execute(
                    "SELECT id FROM terms WHERE term = ?",
                    (term,),
                ).fetchone()[0]

                affected_term_ids.add(term_id)

                self._connection.execute(
                    """
                    INSERT INTO postings (
                        term_id,
                        document_id,
                        field,
                        positions
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        term_id,
                        document_id,
                        field,
                        json.dumps(positions),
                    ),
                )

            self._connection.executemany(
                """
                UPDATE terms
                SET document_frequency = (
                    SELECT COUNT(DISTINCT document_id)
                    FROM postings
                    WHERE term_id = ?
                )
                WHERE id = ?
                """,
                (
                    (term_id, term_id)
                    for term_id in affected_term_ids
                ),
            )

    def lookup_posting(
        self,
        term: str,
        field: int,
    ) -> dict[int, list[int]]:
        rows = self._connection.execute(
            """
            SELECT document_id, positions
            FROM postings
            WHERE term_id = (
                SELECT id FROM terms
                WHERE term = ?
            )
            AND field = ?
            ORDER BY document_id
            """,
            (term, field),
        ).fetchall()

        return {
            document_id: json.loads(positions)
            for document_id, positions in rows
        }

    def get_document_by_index(
        self,
        document_id: int,
    ) -> tuple[str, str, str] | None:
        return self._connection.execute(
            """
            SELECT fetched_at, title_norm, text_norm
            FROM indexed_documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

    def count_terms(self) -> int:
        return self._connection.execute(
            "SELECT COUNT(*) FROM terms WHERE document_frequency > 0"
        ).fetchone()[0]

    def get_total_len_by_field(self, field: int) -> int:
        column = {
            self.TITLE: "title_len",
            self.TEXT: "text_len",
        }[field]

        return self._connection.execute(
            f"SELECT COALESCE(SUM({column}), 0) FROM indexed_documents"
        ).fetchone()[0]

    def get_metadata(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM index_metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0] if row is not None else None

    def set_metadata(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO index_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self._connection.commit()
