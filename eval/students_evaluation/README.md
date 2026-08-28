# 学生评测说明

本目录包含课程的学生端评测代码。RAG 部分为进阶内容，在完成基础检索任务后继续进行。

## 目录结构

```text
students_evaluation/
├── main/                     # 基础检索评测
│   ├── client.py             # 基础检索评测客户端
│   └── search_engine.py      # 基础检索接口
└── rag/                      # 进阶：RAG 检索与问答
    ├── client.py
    ├── call_model.py
    ├── search_engine.py
    └── rag.ipynb              # RAG 教学 notebook
```

## 基础检索评测

基础任务使用 `students_evaluation/main/` 目录中的 `client.py` 和
`search_engine.py`。按照课程要求完成自己的搜索引擎后运行：

```bash
cd students_evaluation/main
python client.py
```

客户端会返回每次 `evaluate(query)` 的 MRR 得分，并统计每次
`evaluate(query)` 的端到端耗时。评测完成后会显示所有查询的平均
MRR 得分和平均响应时延；MRR 得分是自动化评测部分的唯一依据。

空密码进入 debug 模式。正式评测只在最后一天上午开放，每位同学需在助教的
监督下利用提供的密码仅评测一次，得到最终得分。

## RAG 进阶部分

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

**RAG正式评测同基础检索评测的要求。**
