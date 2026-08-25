from dataclasses import dataclass


@dataclass
class Document:
    url: str
    title: str
    text: str
    fetched_at: str


def connect(database_path: str):
    ...

def create_tables(connection: any) -> None:
    ...

def save_document(connection: any, document: Document) -> int: 
    ...

def get_document(connection, document_id: int) -> Document | None:
    ...

def list_document(connection) -> list[tuple]:
    ...
