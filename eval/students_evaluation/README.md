# RAG 学生评测说明

本目录包含课程的进阶 RAG 评测代码。

## 目录结构

```text
students_evaluation/
└── rag/                      # RAG 检索与问答
    ├── client.py
    ├── call_model.py
    ├── search_engine.py
    └── rag.ipynb              # RAG 教学 notebook
```

## RAG 评测

RAG 代码位于 `students_evaluation/rag/`，主要流程是：

```text
问题 -> search() -> Top-K 结果 -> 信息整合 -> 大模型 -> 答案
```

同学可以参考 `rag/rag.ipynb`，按需要实现以下接口：

```python
search(query, top_k)
snippet_merge(results)
full_merge(results)
custom_integrator(results, query)
```

其中：

- `search()`：接入自己的倒排索引、向量检索、混合检索或本地知识库；
- `snippet_merge()`：将搜索摘要清洗、去重并组织成上下文；
- `full_merge()`：读取网页或本地文档正文后组织上下文；
- `custom_integrator()`：根据问题进行压缩、抽取、重排或其他处理。

RAG 使用 OpenAI 兼容接口。运行前请在 `rag/call_model.py` 中填写课程允许使用的大模型配置；搜索引擎部分请使用自己实现的接口。

RAG 评测客户端的运行方式：

```bash
cd students_evaluation/rag
python client.py
```

空密码进入 debug 模式。debug 提交会显示每道题的裁判分数和评分理由，
便于调整检索及回答方法。RAG 客户端还会统计
每次 `rag_evaluate(query)` 的端到端耗时，**耗时超过60s的题目视作超时计为0分**。

正式评测只在开放时段进行，每位同学需在助教监督下使用提供的密码评测。
