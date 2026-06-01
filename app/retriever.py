from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import settings


client = QdrantClient(url=settings.QDRANT_URL)
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)


def search_audit_knowledge(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    if top_k is None:
        top_k = settings.TOP_K

    query_vector = embedding_model.encode(question).tolist()

    results = client.query_points(
    collection_name=settings.QDRANT_COLLECTION,
    query=query_vector,
    limit=top_k,
    with_payload=True,
    ).points

    docs = []

    for item in results:
        payload = item.payload or {}

        docs.append(
            {
                "score": item.score,
                "question_id": payload.get("question_id"),
                "title": payload.get("title"),
                "question": payload.get("question"),
                "content": payload.get("content"),
                "url": payload.get("url"),
                "publish_date": payload.get("publish_date"),
                "source": payload.get("source"),
                "category": payload.get("category"),
                "doc_type": payload.get("doc_type"),
            }
        )

    return docs


def format_docs_as_context(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "未检索到相关资料。"

    blocks = []

    for idx, doc in enumerate(docs, start=1):
        block = f"""[资料{idx}]
匹配分数：{doc.get("score")}
标题：{doc.get("title")}
发布日期：{doc.get("publish_date")}
来源：{doc.get("source")}
原文链接：{doc.get("url")}
正文：
{doc.get("content")}
"""
        blocks.append(block)

    return "\n\n".join(blocks)
