import json
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from pathlib import Path


class Storage:
    """按需打开并统一管理文档、爬取队列和索引数据库。"""

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

    def __enter__(self) -> "Storage":  # noqa: PYI034
        """打开已配置的数据库并初始化相应的数据表。"""

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
        """关闭当前 Storage 管理的全部数据库连接。"""

        if self._stack is None:
            return

        stack = self._stack
        self._stack = None
        self.documents = None
        self.queue = None
        self.index = None
        stack.__exit__(exc_type, exc, tb)

    def _open(self, path: Path) -> sqlite3.Connection:
        """打开 SQLite 数据库，并把连接交给当前上下文统一关闭。"""

        if self._stack is None:
            raise RuntimeError("Storage 尚未打开")

        path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(path)
        connection.execute("PRAGMA foreign_keys = ON")

        return self._stack.enter_context(closing(connection))


class _Document:
    """管理原始网页文档；documents 每行对应一个 URL 的最新抓取结果。"""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def initialize(self) -> None:
        """创建文档表。"""

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                content_html TEXT NOT NULL DEFAULT ''
            )
            """
        )
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(documents)")
        }
        if "content_html" not in columns:
            self._connection.execute(
                "ALTER TABLE documents "
                "ADD COLUMN content_html TEXT NOT NULL DEFAULT ''"
            )
        self._connection.commit()

    def save(
        self,
        *,
        url: str,
        title: str,
        text: str,
        fetched_at: str,
        content_html: str = "",
    ) -> int:
        """新增或更新 URL 对应的文档，并返回稳定的文档 ID。"""

        row = self._connection.execute(
            """
            INSERT INTO documents (url, title, text, fetched_at, content_html)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                text = excluded.text,
                fetched_at = excluded.fetched_at,
                content_html = excluded.content_html
            RETURNING id
            """,
            (
                url,
                title,
                text,
                fetched_at,
                content_html,
            ),
        ).fetchone()

        self._connection.commit()
        return row[0]

    def get(self, document_id: int) -> tuple[str, str, str, str] | None:
        """按文档 ID 返回 URL、标题、正文和抓取时间。"""

        row = self._connection.execute(
            """
            SELECT url, title, text, fetched_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,)
        ).fetchone()
        return row

    def get_content_html(self, document_id: int) -> str:
        """返回文档的安全结构化正文；旧数据没有该字段内容时返回空字符串。"""

        row = self._connection.execute(
            "SELECT content_html FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        return row[0] if row is not None else ""

    def iter_all(self) -> Iterator[tuple[int, str, str, str, str]]:
        """按文档 ID 顺序遍历所有原始文档。"""

        return iter(self._connection.execute(
            """
            SELECT id, url, title, text, fetched_at
            FROM documents
            ORDER BY id
            """
        ))

    def iter_all_with_content_html(
        self,
    ) -> Iterator[tuple[int, str, str, str, str, str]]:
        """按文档 ID 顺序遍历文档及其结构化正文。"""

        return iter(self._connection.execute(
            """
            SELECT id, url, title, text, fetched_at, content_html
            FROM documents
            ORDER BY id
            """
        ))

    def count_by_host(self, host: str) -> int:
        """统计指定主机下已保存的文档数量。"""

        return self._connection.execute(
            """
            SELECT COUNT(*)
            FROM documents
            WHERE url LIKE ? OR url LIKE ?
            """,
            (f"http://{host}/%", f"https://{host}/%"),
        ).fetchone()[0]


class _Queue:
    """管理爬取任务的状态、尝试次数、最近错误和完成时间。"""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def initialize(self):
        """创建爬取任务表及其状态索引。"""

        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS crawl_tasks (
                url TEXT PRIMARY KEY,
                state TEXT NOT NULL DEFAULT 'pending'
                    CHECK (state IN ('pending', 'processing', 'done', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                finished_at TEXT
            );

            CREATE INDEX IF NOT EXISTS crawl_tasks_state_idx
            ON crawl_tasks(state);
            """
        )
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(crawl_tasks)")
        }
        if "finished_at" not in columns:
            self._connection.execute(
                "ALTER TABLE crawl_tasks ADD COLUMN finished_at TEXT"
            )
        self._connection.commit()

    def enqueue(self, urls: list[str] | tuple[str, ...], ) -> list[str]:
        """加入尚不存在的 URL，并返回实际新增的 URL。"""

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
        """将上次中断时仍在处理的任务恢复为待处理，并返回恢复数量。"""

        cursor = self._connection.execute(
            """
            UPDATE crawl_tasks
            SET state = 'pending', finished_at = NULL
            WHERE state = 'processing'
            """
        )
        self._connection.commit()
        return cursor.rowcount

    def claim_pending(self, limit: int) -> list[str]:
        """按入队顺序领取最多 limit 个待处理任务。"""

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
            SET state = 'processing', attempts = attempts + 1,
                last_error = NULL, finished_at = NULL
            WHERE url = ? AND state = 'pending'
            """,
            ((url,) for url in urls),
        )
        self._connection.commit()
        return urls

    def claim(self, url: str) -> bool:
        """尝试领取指定 URL；仅当它原本处于待处理状态时成功。"""

        cursor = self._connection.execute(
            """
            UPDATE crawl_tasks
            SET state = 'processing', attempts = attempts + 1,
                last_error = NULL, finished_at = NULL
            WHERE url = ? AND state = 'pending'
            """,
            (url,),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def complete(self, url: str):
        """将指定任务标记为完成并清除最近错误。"""

        self._connection.execute(
            """
            UPDATE crawl_tasks
            SET state = 'done', last_error = NULL, finished_at = ?
            WHERE url = ?
            """,
            (datetime.now(timezone.utc).isoformat(), url),
        )
        self._connection.commit()

    def fail(self, url: str, error: str, max_attempts: int) -> str:
        """记录失败；未达尝试上限则重试，否则永久失败，并返回新状态。"""

        finished_at = datetime.now(timezone.utc).isoformat()
        self._connection.execute(
            """
            UPDATE crawl_tasks
            SET state = CASE
                    WHEN attempts < ? THEN 'pending'
                    ELSE 'failed'
                END,
                last_error = ?,
                finished_at = CASE
                    WHEN attempts < ? THEN NULL
                    ELSE ?
                END
            WHERE url = ?
            """,
            (max_attempts, error, max_attempts, finished_at, url),
        )
        state = self._connection.execute(
            "SELECT state FROM crawl_tasks WHERE url = ?",
            (url,),
        ).fetchone()[0]
        self._connection.commit()
        return state

    def requeue(
        self,
        *,
        done_before: str | None = None,
        failed: bool = False,
    ) -> dict[str, int]:
        """重新排入过期的完成任务和可选的历史失败任务。"""

        counts = {"done": 0, "failed": 0}
        with self._connection:
            if done_before is not None:
                cursor = self._connection.execute(
                    """
                    UPDATE crawl_tasks
                    SET state = 'pending', attempts = 0, finished_at = NULL
                    WHERE state = 'done'
                      AND (finished_at IS NULL OR finished_at <= ?)
                    """,
                    (done_before,),
                )
                counts["done"] = cursor.rowcount
            if failed:
                cursor = self._connection.execute(
                    """
                    UPDATE crawl_tasks
                    SET state = 'pending', attempts = 0, finished_at = NULL
                    WHERE state = 'failed'
                    """
                )
                counts["failed"] = cursor.rowcount
        return counts

    def count_by_state(self) -> dict[str, int]:
        """返回各任务状态对应的任务数量。"""

        return dict(
            self._connection.execute(
                "SELECT state, COUNT(*) FROM crawl_tasks GROUP BY state"
            ).fetchall()
        )


class _Index:
    """管理用于检索的文档快照、索引元数据、词典和倒排记录。

    indexed_documents 保存每篇文档的归一化字段、token 数和抓取时间；
    index_metadata 保存分词模式等索引级配置；terms 保存词项及文档频率；
    postings 保存词项在每篇文档、每个字段中的出现位置。
    """

    TITLE, TEXT = 0, 1

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def initialize(self) -> None:
        """创建索引所需的数据表和辅助索引。"""

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
        """在同一事务中替换一篇文档的快照、倒排记录和词项频率。"""

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
        """返回词项在指定字段中的文档 ID 到出现位置列表的映射。"""

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
        """返回已索引文档的抓取时间、归一化标题和归一化正文。"""

        return self._connection.execute(
            """
            SELECT fetched_at, title_norm, text_norm
            FROM indexed_documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

    def get_total_len(self) -> tuple[int, int]:
        """返回所有已索引文档的标题和正文 token 总数。"""

        return self._connection.execute(
            """
            SELECT
                COALESCE(SUM(title_len), 0),
                COALESCE(SUM(text_len), 0)
            FROM indexed_documents
            """
        ).fetchone()

    def get_document_lengths(
        self,
        document_ids: set[int],
    ) -> dict[int, tuple[int, int]]:
        """返回指定文档的标题和正文 token 数。"""

        if not document_ids:
            return {}

        placeholders = ", ".join("?" for _ in document_ids)
        rows = self._connection.execute(
            f"""
            SELECT document_id, title_len, text_len
            FROM indexed_documents
            WHERE document_id IN ({placeholders})
            """,
            tuple(document_ids),
        ).fetchall()

        return {
            document_id: (title_len, text_len)
            for document_id, title_len, text_len in rows
        }

    def count_documents(self) -> int:
        """返回当前文档数量。"""

        return self._connection.execute(
            "SELECT COUNT(*) FROM indexed_documents"
        ).fetchone()[0]

    def count_terms(self) -> int:
        """返回当前至少出现在一篇文档中的词项数量。"""

        return self._connection.execute(
            "SELECT COUNT(*) FROM terms WHERE document_frequency > 0"
        ).fetchone()[0]

    def get_metadata(self, key: str) -> str | None:
        """读取索引级元数据；键不存在时返回 None。"""

        row = self._connection.execute(
            "SELECT value FROM index_metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0] if row is not None else None

    def set_metadata(self, key: str, value: str) -> None:
        """新增或覆盖一项索引级元数据。"""

        self._connection.execute(
            """
            INSERT INTO index_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self._connection.commit()
