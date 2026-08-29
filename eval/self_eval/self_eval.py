#!/usr/bin/env python3
"""Generate, label, and run a frozen three-layer search self-evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FROZEN_SRC = HERE / "frozen_src"
sys.path.insert(0, str(FROZEN_SRC if FROZEN_SRC.is_dir() else ROOT / "src"))

from zeta_engine.dense import warmup_dense  # noqa: E402
from zeta_engine.eval import evaluate  # noqa: E402
from zeta_engine.search import search_bm25f  # noqa: E402
from zeta_engine.storage import Storage  # noqa: E402
from zeta_engine.tokenizer import text_normalize  # noqa: E402


DEFAULT_FREEZE = HERE / "frozen.json"
DEFAULT_QUESTIONS = HERE / "questions.jsonl"
DEFAULT_SELECTIONS = HERE / "selections.jsonl"
DEFAULT_CASES = HERE / "cases.jsonl"
DEFAULT_REPORT = HERE / "report.json"
DEFAULT_DOCUMENT_DB = ROOT / "data/zeta.db"
DEFAULT_INDEX_DB = ROOT / "data/index.db"
DEFAULT_DENSE_INDEX = ROOT / "data/dense"

_DATE_SUFFIX_RE = re.compile(
    r"\s*(?:日期|发布时间|更新时间)\s*[:：]?\s*"
    r"20\d{2}[-年/.]\d{1,2}[-月/.]\d{1,2}日?\s*$"
)
_SITE_SUFFIX_RE = re.compile(r"\s*[-_|]\s*中国人民大学[^-_|]{0,30}$")
_ARTICLE_SUFFIXES = {".htm", ".html", ".shtml"}


def _jsonl_read(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _jsonl_write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted((FROZEN_SRC / "zeta_engine").glob("*.py"))
    paths.extend([args.document_db, args.index_db])
    metadata_path = args.dense_index / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paths.extend([
        metadata_path,
        args.dense_index / str(metadata["vector_file"]),
        args.dense_index / str(metadata["chunk_file"]),
    ])
    model_path = ROOT / str(metadata["model_path"])
    paths.extend(sorted(path for path in model_path.rglob("*") if path.is_file()))
    return paths


def _fingerprint(paths: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.resolve().relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def freeze(args: argparse.Namespace) -> None:
    snapshot = FROZEN_SRC / "zeta_engine"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "src/zeta_engine").glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    files = _fingerprint(_frozen_paths(args))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen {len(files)} files at {commit[:12]} -> {args.output}")


def verify_freeze(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = []
    for item in payload["files"]:
        target = ROOT / item["path"]
        if (
            not target.is_file()
            or target.stat().st_size != item["size"]
            or _sha256(target) != item["sha256"]
        ):
            changed.append(item["path"])
    if changed:
        raise RuntimeError("冻结基线已漂移: " + ", ".join(changed))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_title(title: str) -> str:
    title = text_normalize(title)
    title = _DATE_SUFFIX_RE.sub("", title)
    title = _SITE_SUFFIX_RE.sub("", title)
    return title.strip(" -_|—_")


def _eligible(document: dict[str, object]) -> bool:
    title = canonical_title(str(document["title"]))
    path = Path(urlparse(str(document["url"])).path)
    return (
        10 <= len(title) <= 80
        and len(str(document["text"])) >= 500
        and path.suffix.lower() in _ARTICLE_SUFFIXES
        and not path.stem.lower().startswith("index")
    )


def _load_documents(path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT id, url, title, text FROM documents ORDER BY id"
        ).fetchall()
    return [
        {"id": row[0], "url": row[1], "title": row[2], "text": row[3]}
        for row in rows
    ]


def _stratified_sample(
    documents: list[dict[str, object]], count: int, seed: int
) -> list[dict[str, object]]:
    by_host: dict[str, list[dict[str, object]]] = defaultdict(list)
    for document in documents:
        by_host[urlparse(str(document["url"])).netloc].append(document)
    rng = random.Random(seed)
    for rows in by_host.values():
        rng.shuffle(rows)
    hosts = sorted(by_host, key=lambda host: (-len(by_host[host]), host))
    selected = []
    while hosts and len(selected) < count:
        next_hosts = []
        for host in hosts:
            if by_host[host] and len(selected) < count:
                selected.append(by_host[host].pop())
            if by_host[host]:
                next_hosts.append(host)
        hosts = next_hosts
    return selected


def natural_question(title: str) -> tuple[str, str] | None:
    title = canonical_title(title)
    if any(word in title for word in ("报名", "招生", "推免", "考核")):
        return f"{title}有哪些报名或考核安排？", "admission"
    if any(word in title for word in ("名单", "成绩", "公示", "拟录取")):
        return f"在哪里可以查询{title}？", "roster"
    if any(word in title for word in ("讲座", "论坛", "研讨会", "报告会")):
        return f"{title}的活动信息是什么？", "event"
    if any(word in title for word in ("研究成果", "研究进展", "发表")):
        return f"{title}介绍了什么研究成果？", "research"
    if any(word in title for word in ("获奖", "入选", "当选", "荣获")):
        return f"{title}涉及谁获得了什么荣誉？", "honor"
    if any(word in title for word in ("招聘", "招聘启事")):
        return f"{title}提供了哪些招聘信息？", "recruitment"
    if any(word in title for word in ("成立", "揭牌", "签署")):
        return f"{title}报道了什么事件？", "news"
    return None


def challenge_mutations(query: str) -> list[tuple[str, str]]:
    variants = []
    shorthand = query
    for old, new in (
        ("中国人民大学", "人大"),
        ("推荐免试", "推免"),
        ("研究生", "研招"),
        ("综合考核", "复试"),
    ):
        shorthand = shorthand.replace(old, new)
    if shorthand != query:
        variants.append(("shorthand", shorthand))

    keywords = query
    for phrase in (
        "在哪里可以查询",
        "有哪些报名或考核安排",
        "的活动信息是什么",
        "介绍了什么研究成果",
        "涉及谁获得了什么荣誉",
        "提供了哪些招聘信息",
        "报道了什么事件",
    ):
        keywords = keywords.replace(phrase, "")
    keywords = keywords.strip("？?，,。 ")
    if keywords and keywords != query:
        variants.append(("keyword", keywords))

    compact = re.sub(r"[\s，。；：、“”‘’！？?—-]+", "", query)
    if compact and compact not in {query, *(item[1] for item in variants)}:
        variants.append(("noisy", compact))
    return variants[:2]


def generate(args: argparse.Namespace) -> None:
    verify_freeze(args.freeze)
    all_documents = _load_documents(args.document_db)
    documents = [item for item in all_documents if _eligible(item)]
    title_urls: dict[str, list[str]] = defaultdict(list)
    eligible_title_urls: dict[str, list[str]] = defaultdict(list)
    for document in all_documents:
        title_urls[canonical_title(str(document["title"]))].append(
            str(document["url"])
        )
    for document in documents:
        eligible_title_urls[canonical_title(str(document["title"]))].append(
            str(document["url"])
        )

    layer1_candidates = [
        item for item in documents
        if len(title_urls[canonical_title(str(item["title"]))]) <= 3
    ]
    layer1_docs = _stratified_sample(layer1_candidates, args.layer1, args.seed)
    layer1 = [
        {
            "id": f"l1-{number:04d}",
            "layer": 1,
            "query": canonical_title(str(document["title"])),
            "source_id": document["id"],
            "source_title": document["title"],
            "auto_relevant_urls": sorted(set(title_urls[canonical_title(
                str(document["title"])
            )])),
            "tags": ["title", urlparse(str(document["url"])).netloc],
        }
        for number, document in enumerate(layer1_docs, start=1)
    ]

    synthetic_docs = [
        item for item in documents
        if len(eligible_title_urls[canonical_title(str(item["title"]))]) <= 3
        and natural_question(str(item["title"]))
    ]
    pool = _stratified_sample(synthetic_docs, args.layer2_pool, args.seed + 1)
    layer2 = []
    with Storage(document_db=args.document_db, index_db=args.index_db) as storage:
        assert storage.documents is not None
        for number, document in enumerate(pool, start=1):
            question, tag = natural_question(str(document["title"])) or ("", "")
            ranked_ids = search_bm25f(storage, question)[:args.candidates]
            candidate_ids = list(dict.fromkeys([int(document["id"]), *ranked_ids]))
            candidates = []
            for document_id in candidate_ids:
                found = storage.documents.get(document_id)
                if found is not None:
                    url, title, _text, _fetched_at = found
                    candidates.append({"url": url, "title": title})
            layer2.append({
                "id": f"l2-{number:04d}",
                "layer": 2,
                "query": question,
                "source_id": document["id"],
                "source_title": document["title"],
                "candidate_answers": candidates,
                "tags": [tag, urlparse(str(document["url"])).netloc],
            })

    _jsonl_write(args.output, [*layer1, *layer2])
    print(
        f"Generated {len(layer1)} automatic title cases and "
        f"{len(layer2)} synthetic questions for answer selection -> {args.output}"
    )


def build(args: argparse.Namespace) -> None:
    verify_freeze(args.freeze)
    questions = _jsonl_read(args.questions)
    selections = {
        str(item["id"]): item for item in _jsonl_read(args.selections)
    }
    cases = []
    for item in questions:
        if item["layer"] == 1:
            relevant = item["auto_relevant_urls"]
        else:
            selection = selections.get(str(item["id"]))
            if selection is None or not selection.get("relevant_urls"):
                continue
            relevant = selection["relevant_urls"]
            candidates = {
                candidate["url"] for candidate in item["candidate_answers"]
            }
            unknown = set(relevant) - candidates
            if unknown:
                raise ValueError(
                    f"{item['id']} 选择了候选列表之外的 URL: "
                    + ", ".join(sorted(unknown))
                )
        cases.append({
            "id": item["id"],
            "layer": item["layer"],
            "query": item["query"],
            "relevant_urls": relevant,
            "tags": item["tags"],
        })
        if item["layer"] == 2:
            for index, (kind, query) in enumerate(
                challenge_mutations(str(item["query"])), start=1
            ):
                cases.append({
                    "id": f"l3-{str(item['id'])[3:]}-{index}",
                    "layer": 3,
                    "query": query,
                    "relevant_urls": relevant,
                    "parent_id": item["id"],
                    "tags": [*item["tags"], kind],
                })
    _jsonl_write(args.output, cases)
    counts = {layer: sum(item["layer"] == layer for item in cases) for layer in (1, 2, 3)}
    print(f"Built cases {counts} -> {args.output}")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    ranks = [item["rank"] for item in rows]
    latencies = [float(item["latency_seconds"]) for item in rows]
    total = len(rows)
    return {
        "cases": total,
        "recall@1": sum(rank is not None and rank <= 1 for rank in ranks) / total,
        "recall@5": sum(rank is not None and rank <= 5 for rank in ranks) / total,
        "recall@20": sum(rank is not None and rank <= 20 for rank in ranks) / total,
        "mrr@20": sum(1 / rank for rank in ranks if rank is not None and rank <= 20) / total,
        "latency_mean_seconds": statistics.fmean(latencies),
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": _percentile(latencies, .95),
    }


def _evaluate_ranking(
    cases: list[dict[str, object]], args: argparse.Namespace, ranking: str
) -> dict[str, object]:
    if ranking in {"dense", "hybrid", "rerank"}:
        warmup_dense(args.dense_index, device=args.device)
    rows = []
    with Storage(
        document_db=args.document_db,
        index_db=args.index_db if ranking != "dense" else None,
    ) as storage:
        for number, case in enumerate(cases, start=1):
            started = time.monotonic()
            urls = evaluate(
                storage,
                str(case["query"]),
                ranking=ranking,
                dense_index=args.dense_index,
                device=args.device,
                alpha=args.alpha,
            )
            latency = time.monotonic() - started
            relevant = set(case["relevant_urls"])
            rank = next(
                (index for index, url in enumerate(urls, start=1) if url in relevant),
                None,
            )
            rows.append({
                "id": case["id"],
                "layer": case["layer"],
                "rank": rank,
                "latency_seconds": round(latency, 6),
                "top_urls": urls[:5],
            })
            if number % 25 == 0 or number == len(cases):
                print(f"{ranking}: {number}/{len(cases)}")

    by_layer = {
        str(layer): _metrics([item for item in rows if item["layer"] == layer])
        for layer in (1, 2, 3)
        if any(item["layer"] == layer for item in rows)
    }
    row_by_id = {str(item["id"]): item for item in rows}
    parent_by_id = {str(item["id"]): item.get("parent_id") for item in cases}
    comparable = []
    for item in rows:
        parent_id = parent_by_id.get(str(item["id"]))
        parent = row_by_id.get(str(parent_id))
        if parent and parent["rank"] is not None and parent["rank"] <= 20:
            comparable.append(item)
    robustness = (
        sum(item["rank"] is not None and item["rank"] <= 20 for item in comparable)
        / len(comparable)
        if comparable else 0.0
    )
    return {
        "overall": _metrics(rows),
        "by_layer": by_layer,
        "challenge_recall@20_given_parent_hit": robustness,
        "details": rows,
    }


def run_evaluation(args: argparse.Namespace) -> None:
    freeze_sha256 = verify_freeze(args.freeze)
    cases = _jsonl_read(args.cases)
    if not cases or any(not item.get("relevant_urls") for item in cases):
        raise ValueError("评测集为空或存在未标注题目")
    rankings = [item.strip() for item in args.rankings.split(",") if item.strip()]
    invalid = set(rankings) - {"bm25f", "dense", "hybrid", "rerank"}
    if invalid:
        raise ValueError("不支持的 ranking: " + ", ".join(sorted(invalid)))
    results = {
        ranking: _evaluate_ranking(cases, args, ranking) for ranking in rankings
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_sha256,
        "case_file_sha256": _sha256(args.cases),
        "settings": {
            "rankings": rankings,
            "alpha": args.alpha,
            "device": args.device,
        },
        "results": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for ranking, result in results.items():
        overall = result["overall"]
        print(
            f"{ranking}: MRR@20={overall['mrr@20']:.3f}, "
            f"R@1={overall['recall@1']:.3f}, R@20={overall['recall@20']:.3f}, "
            f"p95={overall['latency_p95_seconds']:.3f}s"
        )
    print(f"Report -> {args.output}")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--document-db", type=Path, default=DEFAULT_DOCUMENT_DB)
    parser.add_argument("--index-db", type=Path, default=DEFAULT_INDEX_DB)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    command = commands.add_parser("freeze", help="记录当前引擎与索引哈希")
    _common(command)
    command.add_argument("--output", type=Path, default=DEFAULT_FREEZE)
    command.set_defaults(handler=freeze)

    command = commands.add_parser("generate", help="生成一、二层题目及候选答案")
    _common(command)
    command.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    command.add_argument("--output", type=Path, default=DEFAULT_QUESTIONS)
    command.add_argument("--layer1", type=int, default=200)
    command.add_argument("--layer2-pool", type=int, default=60)
    command.add_argument("--candidates", type=int, default=8)
    command.add_argument("--seed", type=int, default=20260829)
    command.set_defaults(handler=generate)

    command = commands.add_parser("build", help="合并答案并派生第三层挑战题")
    command.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    command.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    command.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    command.add_argument("--output", type=Path, default=DEFAULT_CASES)
    command.set_defaults(handler=build)

    command = commands.add_parser("evaluate", help="运行本地检索评测")
    _common(command)
    command.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    command.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    command.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    command.add_argument("--rankings", default="bm25f,dense,hybrid")
    command.add_argument("--alpha", type=float, default=.5)
    command.add_argument("--device")
    command.set_defaults(handler=run_evaluation)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
