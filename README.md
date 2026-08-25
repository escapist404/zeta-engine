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

### 查看抓取统计

```bash
uv run zeta-engine stats
```

### 指定数据库

```bash
uv run zeta-engine stats \
  --document-db data/zeta.db
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
