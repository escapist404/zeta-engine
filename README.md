# ζ-engine

ζ-engine是一个面向中国人民大学相关站点的中文搜索与 RAG 问答引擎。它能够抓取多个学院站点，保存清理后的正文与语义 HTML，并提供 BM25F、Dense、Hybrid、CrossEncoder 重排和多轮 RAG。

## 主要能力

- 带持久化队列、并发下载、失败重试和定期刷新的站点爬虫。
- 按站点和页面类型选择正文容器，清理导航、侧栏、面包屑和页脚噪声。
- 文档级 BM25F 检索和连续位置的精确短语检索。
- 共享 passage 边界的 BM25 与 BGE Dense 索引，支持混合召回。
- 使用 BGE CrossEncoder 对候选 passage 重排并按文档聚合。
- 多轮 RAG：跟踪问题要求、补充检索、恢复语义结构、验证引用与答案完整性。
- 针对名单、年份、分组计数和跨组交集问题的结构化集合处理。
- 同时提供 CLI、Web 界面和 JSON API。

## 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- 数 GB 本地空间用于 embedding 和 reranker 模型
- CPU，或支持 PyTorch 的 CUDA/MPS 设备

## 安装

```bash
git clone https://github.com/escapist404/zeta-engine.git
cd zeta-engine
uv sync
```

查看可用命令：

```bash
uv run zeta-engine --help
```

## 快速开始

### 1. 抓取站点

```bash
uv run zeta-engine crawl
```

默认从 `src/zeta_engine/constants.py` 中的 `SEED_URLS` 开始，只跟踪 `ALLOWED_DOMAINS` 范围内的链接。文档、爬取队列和日志分别写入 `data/zeta.db`、`data/crawl_queue.db` 和 `logs/zeta-engine.log`。

调整爬取规模和并发：

```bash
uv run zeta-engine crawl \
  --max-pages 1000 \
  --workers 8 \
  --delay 0.5
```

重新抓取 24 小时前完成的页面，并重试历史失败任务：

```bash
uv run zeta-engine crawl --refresh-after-hours 24 --retry-failed
```

`--refresh-after-hours 0` 会刷新所有已完成页面。正文提取的通用选择器位于 `constants.py`，站点专用规则位于 `extraction_rules.py`。

### 2. 构建文档索引

```bash
uv run zeta-engine index --mode search
```

该命令使用 jieba 构建文档级倒排索引，默认写入 `data/index.db`。索引是增量的：文档内容和分词配置未变时会跳过已索引项。

### 3. 下载模型并构建 passage 索引

```bash
uv run hf download BAAI/bge-small-zh-v1.5 \
  --local-dir models/bge-small-zh-v1.5 \
  --exclude "pytorch_model.bin"

uv run hf download BAAI/bge-reranker-base \
  --local-dir models/bge-reranker-base

uv run zeta-engine dense-index --device cpu
```

`dense-index` 将文档切分为带重叠的 passage，一次性生成 Dense 向量、passage 元数据和 passage 级 BM25 索引，默认写入 `data/dense/`。

Apple Silicon 可使用 `--device mps`，NVIDIA GPU 可使用相应 CUDA 设备。

### 4. 搜索

```bash
uv run zeta-engine search "中国人民大学"
uv run zeta-engine search "经济困难学生如何获得帮助" --ranking rerank --device cpu
uv run zeta-engine search "大学生创新创业" --phrase --limit 20
```

`--ranking` 支持：

| 模式 | 说明 | 需要的数据 |
|---|---|---|
| `bm25f` | 文档级 BM25F，标题权重高于正文 | `zeta.db` + `index.db` |
| `dense` | BGE passage 语义检索 | `zeta.db` + `data/dense/` |
| `hybrid` | passage BM25 与 Dense 分数归一化后融合 | `zeta.db` + `data/dense/` |
| `rerank` | Hybrid 召回后使用 CrossEncoder 重排 | Hybrid 数据 + reranker |

Hybrid 默认 `--alpha 0.23`；`0` 偏向稀疏检索，`1` 偏向 Dense 检索。`rerank` 默认重排 50 个候选，批大小为 16，可通过 `--rerank-candidates`、`--reranker-batch-size` 和 `--reranker-model` 调整。

### 5. RAG 问答

RAG 使用 DeepSeek API，当前模型配置为 `deepseek-v4-flash`。可以把 Key 放在
`.env` 中，再导入当前 shell：

```bash
set -a
source .env
set +a
uv run zeta-engine rag "经济困难学生如何申请资助？" --device cpu
```

默认端点是 `https://api.deepseek.com`。如需切换其他 OpenAI-compatible 模型，
可在 `.env` 中同时设置 `ZETA_LLM_BASE_URL` 和 `ZETA_LLM_MODEL`；无需修改 RAG 代码。

RAG 默认最多运行 5 轮。每轮使用 Hybrid passage 召回与 CrossEncoder 重排，再尝试从已保存的语义 HTML 中恢复相邻正文或完整列表、表格和 section。Agent 会跟踪问题中未完成的要求，为其生成补充查询，并检查最终答案是否受现有证据支持。

这里只有一套 Agentic RAG 闭环。`service.answer_question` 是应用入口，所有问题都进入
`rag_engine` 的同一个 `Agent → 工具 → Verifier` 循环。Agent 可在循环中选择普通检索、
完整集合扫描或确定性计算；`collection_rag` 只是完整名单/表格工具，不拥有第二套 system
prompt，也不在闭环外预先路由问题。

```bash
uv run zeta-engine rag "问题" \
  --max-cycles 6 \
  --top-k 8 \
  --rerank-candidates 50 \
  --alpha 0.23 \
  --debug
```

`--debug` 会显示各轮查询、要求状态、证据规模和模型动作。`--max-cycles` 允许 1–8，`--top-k` 控制最终供回答使用的检索结果数。

例如，完整表格中的比例题会留下类似
`collection → deterministic_table_ratio` 的 trace：Agent 决定需要完整集合，工具扫描整张表并
返回机器可读的分子、分母和上下界，闭环直接生成答案，不再让模型手工数 Top-K。

## Web 界面与 API

构建索引后启动服务：

```bash
uv run zeta-engine serve
```

默认监听 <http://127.0.0.1:8000>，并托管根目录的 `index.html`。可通过 `--host`、`--port` 和 `--frontend` 修改。

搜索 API：

```text
GET /api/search?q=关键词&ranking=hybrid&limit=20&alpha=0.23
```

RAG API：

```text
GET /api/search?q=问题&ranking=rag&limit=5&max_cycles=5&debug=1
```

`ranking` 可为 `bm25f`、`dense`、`hybrid`、`rerank` 或 `rag`。`limit` 范围为 1–100；RAG 模式默认为 5，其他模式默认为 20。普通搜索返回 `query`、`count` 和 `results`；RAG 还会返回 `answer`、`status`、`complete`、`requirements`、`claims` 和 `sources`。

## 默认路径

| 用途 | 默认路径 |
|---|---|
| 爬取队列 | `data/crawl_queue.db` |
| 文档数据库 | `data/zeta.db` |
| 文档倒排索引 | `data/index.db` |
| passage BM25、Dense 向量与元数据 | `data/dense/` |
| embedding 模型 | `models/bge-small-zh-v1.5/` |
| reranker 模型 | `models/bge-reranker-base/` |
| 运行日志 | `logs/zeta-engine.log` |
| Web 前端 | `index.html` |

大多数命令都允许使用 `--document-db`、`--index-db`、`--dense-index` 或模型路径参数覆盖默认值。

## 统计与评测

查看各站点文档数和索引词项数：

```bash
uv run zeta-engine stats
```

使用配套评测服务运行搜索 MRR@20 评测：

```bash
uv run zeta-engine eval --mode search --ranking hybrid --device cpu
```

运行 RAG 回答评测：

```bash
uv run zeta-engine eval --mode rag --top-k 8 --device cpu
```

默认评测服务地址为 `http://10.47.253.18:8080/`，可使用 `--base-url` 覆盖。搜索模式调用 `/login` 和 `/mrr`，RAG 模式调用 `/rag/login` 和 `/rag/score`。

## 开发与测试

```bash
uv run python -m unittest discover -s tests
```

核心模块：

| 模块 | 职责 |
|---|---|
| `crawler.py` / `extraction_rules.py` | 抓取、链接发现与正文提取 |
| `storage.py` | SQLite 文档、队列和索引存储 |
| `index.py` / `tokenizer.py` | 文档倒排索引与中文分词 |
| `dense.py` / `search.py` | passage 索引、混合检索与重排 |
| `rag.py` / `rag_engine.py` / `rag_*.py` | 兼容入口、唯一 Agentic 闭环、证据、协议、Prompt 与确定性推理 |
| `collection_ops.py` / `collection_rag.py` | 被闭环调用的结构化集合工具与确定性运算 |
| `service.py` / `web.py` / `cli.py` | 服务编排、HTTP API 与命令行入口 |

更改站点范围时，修改 `src/zeta_engine/constants.py` 中的 `SEED_URLS` 和 `ALLOWED_DOMAINS`；更改某站正文容器时，修改 `src/zeta_engine/extraction_rules.py`。
