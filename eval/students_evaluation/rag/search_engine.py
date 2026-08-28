"""独立的学生 RAG 检索与问答接口。"""

from typing import TypedDict

from call_model import call_model


class SearchResult(TypedDict, total=False):
    """一条检索结果。url 必填，其余字段可按自己的搜索引擎能力提供。"""

    url: str
    title: str
    snippet: str
    content: str


def search(query: str, top_k: int = 20) -> list[SearchResult]:
    """检索接口：请替换为自己的搜索引擎。

    返回值按相关性从高到低排列。RAG 推荐至少提供 ``snippet``；如果已经
    保存了网页正文，也可以提供 ``content``。

    示例：
        return [{
            "url": "https://example.com/a",
            "title": "页面标题",
            "snippet": "与查询有关的网页摘要",
            "content": "可选的网页正文",
        }]
    """
    # TODO: 在这里调用你自己的搜索引擎。
    del query, top_k
    return []


def snippet_merge(
    results: list[SearchResult],
    max_chars: int = 12_000,
) -> str:
    """snippet 整合接口：清洗、去重并组织搜索摘要。

    TODO: 读取 ``title``、``snippet``，去重后组织为上下文字符串。
    当前空实现便于同学逐步补充，不会影响未完成代码的调试启动。
    """
    del results, max_chars
    return ""


def full_merge(
    results: list[SearchResult],
    max_chars: int = 12_000,
) -> str:
    """full 整合接口：读取网页或本地文档正文后组织上下文。

    TODO: 自行实现正文读取、HTML 清洗、分块、截断和异常处理。
    """
    del results, max_chars
    return ""


def custom_integrator(
    results: list[SearchResult],
    query: str = "",
    max_chars: int = 12_000,
) -> str:
    """custom 整合接口：按问题压缩、抽取、去重或重排。

    TODO: 可以接入规则、关键词筛选或使用其他方法。
    """
    del results, query, max_chars
    return ""


def integrate_information(
    results: list[SearchResult],
    strategy: str = "snippet",
    query: str = "",
    max_chars: int = 12_000,
) -> str:
    """只负责按 strategy 分发到三种信息整合接口。"""
    if strategy == "snippet":
        return snippet_merge(results, max_chars=max_chars)
    if strategy == "full":
        return full_merge(results, max_chars=max_chars)
    if strategy == "custom":
        return custom_integrator(
            results,
            query=query,
            max_chars=max_chars,
        )
    raise ValueError("strategy 必须是 snippet、full 或 custom")


def rag_evaluate(
    query: str,
    top_k: int = 5,
    strategy: str = "snippet",
) -> str:
    """新增 RAG 评测接口：对一条查询只返回一个字符串答案。

    同学可以修改内部提示词、上下文合并方式或检索数量。评测客户端仍会
    以 ``rag_evaluate(query)`` 调用，因此默认参数必须可以直接运行。
    """
    results = search(query, top_k=top_k)
    context = integrate_information(results, strategy=strategy, query=query)
    if not context:
        return "未检索到足够信息"

    prompt = f"""请仅依据下面的检索材料回答问题。

要求：
1. 回答事实本身，不要输出分析过程；
2. 涉及计数、实体或日期时给出明确结果；
3. 材料不足时回答“材料不足”，不要使用材料之外的知识猜测。

问题：{query}

检索材料：
{context}

最终答案："""
    return call_model(
        user_prompt=prompt,
        system_prompt="你是一个基于检索材料进行事实问答的 RAG 助手。",
        timeout=60.0,
    ).strip()
