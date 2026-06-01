import json
import hashlib
import datetime
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

# 复用你现有爬虫里的函数
from crawl_audit_playwright import collect_article_links_with_playwright, parse_detail_page
from clean_audit_data import (
    clean_title,
    clean_date,
    clean_source,
    clean_content,
    parse_question_info,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

CLEAN_JSON_PATH = DATA_DIR / "audit_knowledge_clean.json"
CLEAN_CSV_PATH = DATA_DIR / "audit_knowledge_clean.csv"
INCREMENTAL_LOG_PATH = LOG_DIR / "incremental_update.log"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "audit_knowledge"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh"

TOP_LIST_PAGES = 10
BATCH_SIZE = 16


def write_log(message: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {message}"
    print(line)

    with open(INCREMENTAL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_existing_data() -> list[dict[str, Any]]:
    if not CLEAN_JSON_PATH.exists():
        write_log(f"未发现旧数据文件：{CLEAN_JSON_PATH}")
        return []

    with open(CLEAN_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("audit_knowledge_clean.json 格式错误，应该是 list")

    return data


def save_clean_data(data: list[dict[str, Any]]):
    # 按 question_id 倒序排列，问题94、93、92...
    data = sorted(
        data,
        key=lambda x: x.get("question_id") or 0,
        reverse=True,
    )

    with open(CLEAN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(data)
    df.to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8-sig")

    write_log(f"已保存清洗数据：{CLEAN_JSON_PATH}")
    write_log(f"已保存 CSV 数据：{CLEAN_CSV_PATH}")


def clean_one_item(raw_item: dict[str, Any]) -> dict[str, Any]:
    title = clean_title(raw_item.get("title", ""))
    publish_date = clean_date(raw_item.get("publish_date", ""))
    source = clean_source(raw_item.get("source", ""))
    url = raw_item.get("url", "").strip()
    content = clean_content(raw_item.get("content", ""), title)

    question_id, question = parse_question_info(title)

    clean_item = {
        "question_id": question_id,
        "title": title,
        "question": question,
        "url": url,
        "publish_date": publish_date,
        "source": source,
        "category": "审计知识",
        "doc_type": "qa",
        "content": content,
        "embedding_text": title + "\n" + content,
        "content_hash": md5_text(content),
        "char_count": len(content),
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return clean_item


def get_point_id(item: dict[str, Any]) -> int:
    """
    Qdrant point id。
    优先使用 question_id。
    如果没有 question_id，就用 url hash 转成整数。
    """
    question_id = item.get("question_id")
    if question_id is not None:
        return int(question_id)

    url = item.get("url", "")
    return int(hashlib.md5(url.encode("utf-8")).hexdigest()[:12], 16)


def upsert_to_qdrant(items: list[dict[str, Any]]):
    if not items:
        write_log("没有需要写入 Qdrant 的数据")
        return

    write_log("加载 embedding 模型...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = QdrantClient(url=QDRANT_URL)

    points = []

    for item in items:
        vector = model.encode(item["embedding_text"]).tolist()

        payload = {
            "question_id": item.get("question_id"),
            "title": item.get("title"),
            "question": item.get("question"),
            "url": item.get("url"),
            "publish_date": item.get("publish_date"),
            "source": item.get("source"),
            "category": item.get("category"),
            "doc_type": item.get("doc_type"),
            "content": item.get("content"),
            "content_hash": item.get("content_hash"),
            "char_count": item.get("char_count"),
            "updated_at": item.get("updated_at"),
        }

        points.append(
            PointStruct(
                id=get_point_id(item),
                vector=vector,
                payload=payload,
            )
        )

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )

        write_log(f"已写入 Qdrant：{i + len(batch)}/{len(points)}")

    write_log("Qdrant 增量写入完成")


def main():
    write_log("========== 开始增量更新审计知识库 ==========")

    existing_data = load_existing_data()
    existing_by_url = {
        item.get("url"): item
        for item in existing_data
        if item.get("url")
    }

    write_log(f"当前本地已有数据：{len(existing_data)} 条")

    write_log("开始抓取列表页链接...")
    current_articles = collect_article_links_with_playwright(max_pages=TOP_LIST_PAGES)
    write_log(f"当前官网列表共发现链接：{len(current_articles)} 条")

    new_articles = []

    for article in current_articles:
        url = article.get("url")
        if not url:
            continue

        if url not in existing_by_url:
            new_articles.append(article)

    write_log(f"发现新增文章：{len(new_articles)} 条")

    if not new_articles:
        write_log("没有新增文章，本次无需更新详情页和向量库")
        write_log("========== 增量更新结束 ==========")
        return

    new_clean_items = []

    for idx, article in enumerate(new_articles, start=1):
        url = article["url"]
        write_log(f"正在爬取新增文章 {idx}/{len(new_articles)}：{url}")

        try:
            raw_detail = parse_detail_page(url)

            if not raw_detail.get("title"):
                raw_detail["title"] = article.get("title", "")

            clean_item = clean_one_item(raw_detail)

            if not clean_item["content"]:
                write_log(f"新增文章正文为空，跳过：{url}")
                continue

            new_clean_items.append(clean_item)
            write_log(f"新增文章清洗完成：{clean_item.get('title')}")

        except Exception as e:
            write_log(f"新增文章爬取失败：{url}")
            write_log(f"错误原因：{e}")

    if not new_clean_items:
        write_log("没有成功清洗的新文章，不执行入库")
        write_log("========== 增量更新结束 ==========")
        return

    merged_by_url = existing_by_url.copy()

    for item in new_clean_items:
        merged_by_url[item["url"]] = item

    merged_data = list(merged_by_url.values())

    save_clean_data(merged_data)
    upsert_to_qdrant(new_clean_items)

    write_log(f"本次新增入库：{len(new_clean_items)} 条")
    write_log(f"当前总数据量：{len(merged_data)} 条")
    write_log("========== 增量更新成功 ==========")


if __name__ == "__main__":
    main()
