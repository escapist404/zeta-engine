import sqlite3


def connect_queue(database_path: str) -> sqlite3.Connection:
    return sqlite3.connect(database_path)


def create_queue_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
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
    connection.commit()


def enqueue_tasks(
    connection: sqlite3.Connection,
    urls: list[str] | tuple[str, ...],
) -> list[str]:
    added = []
    for url in urls:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO crawl_tasks (url) VALUES (?)",
            (url,),
        )
        if cursor.rowcount:
            added.append(url)

    connection.commit()
    return added


def recover_tasks(connection: sqlite3.Connection) -> int:
    cursor = connection.execute(
        """
        UPDATE crawl_tasks
        SET state = 'pending'
        WHERE state = 'processing'
        """
    )
    connection.commit()
    return cursor.rowcount


def skip_template_tasks(connection: sqlite3.Connection) -> int:
    cursor = connection.execute(
        """
        UPDATE crawl_tasks
        SET state = 'done', last_error = 'template URL skipped'
        WHERE state IN ('pending', 'processing')
          AND (
              url LIKE '%{{%'
              OR url LIKE '%}}%'
              OR lower(url) LIKE '%7b%7b%'
              OR lower(url) LIKE '%7d%7d%'
          )
        """
    )
    connection.commit()
    return cursor.rowcount


def claim_pending_tasks(
    connection: sqlite3.Connection,
    limit: int,
) -> list[str]:
    rows = connection.execute(
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

    connection.executemany(
        """
        UPDATE crawl_tasks
        SET state = 'processing', attempts = attempts + 1, last_error = NULL
        WHERE url = ? AND state = 'pending'
        """,
        ((url,) for url in urls),
    )
    connection.commit()
    return urls


def claim_task(connection: sqlite3.Connection, url: str) -> bool:
    cursor = connection.execute(
        """
        UPDATE crawl_tasks
        SET state = 'processing', attempts = attempts + 1, last_error = NULL
        WHERE url = ? AND state = 'pending'
        """,
        (url,),
    )
    connection.commit()
    return cursor.rowcount == 1


def complete_task(connection: sqlite3.Connection, url: str) -> None:
    connection.execute(
        """
        UPDATE crawl_tasks
        SET state = 'done', last_error = NULL
        WHERE url = ?
        """,
        (url,),
    )
    connection.commit()


def fail_task(
    connection: sqlite3.Connection,
    url: str,
    error: str,
    max_attempts: int,
) -> str:
    connection.execute(
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
    state = connection.execute(
        "SELECT state FROM crawl_tasks WHERE url = ?",
        (url,),
    ).fetchone()[0]
    connection.commit()
    return state


def task_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return dict(
        connection.execute(
            "SELECT state, COUNT(*) FROM crawl_tasks GROUP BY state"
        ).fetchall()
    )
