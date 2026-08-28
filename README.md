# ζ-engine `(zeta-engine)`

ζ-engine 是一个面向中国人民大学相关站点的搜索引擎。

## 环境要求

* python 3.13+
* [uv](https://docs.astral.sh/uv/)

## 安装

```bash
git clone https://github.com/escapist404/zeta-engine.git
cd zeta-engine
uv sync
```

## 使用

### 运行爬虫

```bash
uv run zeta-engine crawl
```

常用参数：

```bash
uv run zeta-engine crawl \
  --max-pages 1000 \
  --workers 8 \
  --delay 0.5
```

查看全部参数：

```bash
uv run zeta-engine crawl --help
```

默认使用原有的 BeautifulSoup 抽取器。要改用 Resiliparse：

```bash
uv run zeta-engine crawl --extractor resiliparse
```

### 构造索引

```bash
uv run zeta-engine index --mode default
uv run zeta-engine index --mode search --log-file logs/zeta-engine.log
```

索引开始、每处理 100 篇文档以及索引完成时都会输出日志。

构造 BGE Dense 向量索引：

```bash
uv run hf download BAAI/bge-small-zh-v1.5 \
  --local-dir models/bge-small-zh-v1.5 \
  --exclude "pytorch_model.bin"
uv run zeta-engine dense-index --device mps
```

MPS 不可用时将 `--device mps` 改为 `--device cpu`。Dense 索引默认
写入 `data/dense/`。

如需使用 CrossEncoder 重排，另行下载本地 reranker 模型：

```bash
uv run hf download BAAI/bge-reranker-base \
  --local-dir models/bge-reranker-base
```

### 查询

```bash
uv run zeta-engine search "中国人民大学"
uv run zeta-engine search "中国人民大学" --phrase --limit 20
uv run zeta-engine search "中国人民大学" --ranking bm25f
uv run zeta-engine search "经济困难学生如何获得帮助" --ranking dense --device mps
uv run zeta-engine search "经济困难学生如何获得帮助" --ranking hybrid --alpha 0.5 --device mps
uv run zeta-engine search "经济困难学生如何获得帮助" --ranking rerank --device mps
```

`--ranking` 可选 `bm25f`、`dense`、`hybrid` 或 `rerank`，默认为 `hybrid`。
Hybrid 将两路分数分别做
min-max 归一化后线性融合；
`--alpha 0` 等于 BM25F，`--alpha 1` 等于 Dense，默认为 `0.5`。
`rerank` 默认将 Hybrid 的前 50 个候选切成重叠 passage，按 query 的连续字符
匹配动态选出每篇文档最相关的 1–2 个 passage，再由 CrossEncoder 打分并以 MaxP
聚合；可用
`--rerank-candidates`、`--reranker-batch-size` 和 `--reranker-model` 调整。
`--phrase` 要求词和位置连续匹配。

### RAG 问答

设置 OpenAI 兼容服务的 API Key 后，使用检索结果生成回答：

```bash
export ZETA_LLM_API_KEY="你的 API Key"
uv run zeta-engine rag "经济困难学生如何申请资助？" --device mps
```

RAG 最多运行两轮：首轮检索原问题，证据不足时由 Agent 生成最多 3 个
补充查询，第二轮必须回答。CLI 默认使用 Dense，可调整 Top-K 和底层排名方式：

```bash
uv run zeta-engine rag "问题" --top-k 8 --ranking hybrid --alpha 0.5
```

### Web 前端

先构造索引，再启动同时托管前端和搜索 API 的服务：

```bash
uv run zeta-engine index --mode search
uv run zeta-engine serve
```

打开 <http://127.0.0.1:8000>。搜索接口为
`GET /api/search?q=关键词&limit=20&ranking=dense`；`ranking` 默认为 `hybrid`。
Hybrid 接口示例为
`GET /api/search?q=关键词&ranking=hybrid&alpha=0.5`。
CrossEncoder 接口使用 `GET /api/search?q=关键词&ranking=rerank`。
RAG 可在页面下拉菜单中选择，也可使用
`GET /api/search?q=问题&ranking=rag`；Agent 使用 Hybrid 检索，返回模型回答和两轮中实际使用的来源。

### 统计

```bash
uv run zeta-engine stats
```

### 评测

构造索引后运行搜索 MRR@20 评测（默认 `--mode search`）：

```bash
uv run zeta-engine eval
```

运行 RAG 回答评测：

```bash
uv run zeta-engine eval --mode rag --top-k 5
```

RAG 模式使用 `/rag/login`、`/rag/score` 接口；单题异常或耗时超过 60 秒时
提交空答案，debug 模式会显示逐题裁判分数与理由。

如需使用其他评测服务地址：

```bash
uv run zeta-engine eval --base-url http://localhost:8080
```

搜索模式下空密码进入 debug，评测服务会返回每道查询的 reciprocal rank：

```bash
uv run zeta-engine eval --ranking dense --device mps
uv run zeta-engine eval --ranking hybrid --alpha 0.5 --device mps
uv run zeta-engine eval --ranking rerank --device mps
```

### 指定数据库

```bash
uv run zeta-engine stats \
  --document-db data/zeta.db \
  --index-db data/index.db
```

## 配置

爬虫站点配置位于：

```text
src/zeta_engine/constants.py
```

主要配置：

- `SEED_URLS`：开始爬取的入口页面
- `ALLOWED_DOMAINS`：允许继续抓取的站点范围
- `HEADERS`：HTTP 请求头
- `TIMEOUT`：请求超时时间

## 测试

```bash
uv run python -m unittest discover -s tests
```
