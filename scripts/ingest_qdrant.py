import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# ================== 配置 ==================
DATA_FILE = Path("data/audit_knowledge_clean.json")  # 清洗好的数据
COLLECTION_NAME = "audit_knowledge"
VECTOR_DIM = 512  # 根据你选的 embedding 模型决定
QDRANT_URL = "http://localhost:6333"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh"  # 你本地可以下载或使用 huggingface 模型
# =========================================

# 初始化 Qdrant 客户端
client = QdrantClient(url=QDRANT_URL)

# 检查 collection 是否存在
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    print(f"创建 collection: {COLLECTION_NAME}")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
else:
    print(f"collection {COLLECTION_NAME} 已存在")

# 初始化 embedding 模型
print("加载 embedding 模型...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 读取数据
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

points = []
for item in data:
    # 文本向量化
    vector = model.encode(item["embedding_text"]).tolist()

    # metadata
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
    }

    points.append(PointStruct(id=item["question_id"], vector=vector, payload=payload))

# 批量写入
BATCH_SIZE = 16
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )
    print(f"已写入 {i+len(batch)}/{len(points)} 条数据")

print("所有数据写入 Qdrant 完成！")
