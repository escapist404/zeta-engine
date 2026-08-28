"""原检索（MRR@20）评测接口。"""

from pathlib import Path

from zeta_engine.search import search_bm25f
from zeta_engine.storage import Storage


def evaluate(
        query: str, 
        *, 
        limit: int = 20, 
        document_db: str | Path | None = "/Users/escapist404/BE Course/Conprehensive Programming/zeta-engine/data/zeta.db", 
        index_db: str | Path | None = "/Users/escapist404/BE Course/Conprehensive Programming/zeta-engine/data/index.db", 
) -> list[str]:
    """检索一条查询，并返回按相关性从高到低排列的 URL。

    建议返回 20 条；没有结果时返回空列表。可以在本文件中增加任意辅助
    函数或读取自己的索引，但必须保留本函数的名称和返回格式。
    """
    with Storage(document_db=document_db, index_db=index_db) as storage:
        ids = search_bm25f(storage, query)
        urls = [storage.documents.get(id)[0] for id in ids]
        return urls[:limit]
