from dataclasses import dataclass
import sqlite3


@dataclass
class Document:
    url: str
    title: str
    text: str
    fetched_at: str
    doc_len: int = 0


def connect(database_path: str) -> sqlite3.Connection:
    return sqlite3.connect(database_path)

def create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            url TEXT NOT NULL UNIQUE, 
            title TEXT NOT NULL, 
            text TEXT NOT NULL, 
            fetched_at TEXT NOT NULL, 
            doc_len INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.commit()

def save_document(
    connection: sqlite3.Connection,
    document: Document,
) -> int:
    row = connection.execute(
        """
        INSERT INTO documents (url, title, text, fetched_at, doc_len)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title = excluded.title,
            text = excluded.text,
            fetched_at = excluded.fetched_at,
            doc_len = excluded.doc_len
        RETURNING id
        """,
        (
            document.url,
            document.title,
            document.text,
            document.fetched_at,
            document.doc_len,
        ),
    ).fetchone()

    connection.commit()
    return row[0]

def get_document(
    connection: sqlite3.Connection, 
    document_id: int
) -> Document | None:
    row = connection.execute(
        """
        SELECT url, title, text, fetched_at, doc_len
        FROM documents
        WHERE id = ?
        """,
        (document_id, )
    ).fetchone()
    return Document(*row) if not row is None else None

def list_documents(connection: sqlite3.Connection) -> list[Document]:
    rows = connection.execute(
        """
        SELECT url, title, text, fetched_at, doc_len
        FROM documents
        """
    ).fetchall()
    return [Document(*row) for row in rows]
