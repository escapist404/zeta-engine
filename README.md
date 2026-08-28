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

### 构造索引

```bash
uv run zeta-engine index --mode default
uv run zeta-engine index --mode search --log-file logs/zeta-engine.log
```

索引开始、每处理 100 篇文档以及索引完成时都会输出日志。

### 查询

```bash
uv run zeta-engine search "中国人民大学"
uv run zeta-engine search "中国人民大学" --phrase --limit 20
uv run zeta-engine search "中国人民大学" --ranking tf-idf
uv run zeta-engine search "中国人民大学" --ranking bm25f
```

普通查询会分词并要求所有词都命中；`--ranking` 可选 `simple`、`tf-idf` 或 `bm25f`，默认为 `simple`。`--phrase` 要求词和位置连续匹配。

### Web 前端

先构造索引，再启动同时托管前端和搜索 API 的服务：

```bash
uv run zeta-engine index --mode search
uv run zeta-engine serve
```

打开 <http://127.0.0.1:8000>。搜索接口为 `GET /api/search?q=关键词&limit=10`。

### 统计

```bash
uv run zeta-engine stats
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
