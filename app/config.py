import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "audit_knowledge")

    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.colleryu.xyz/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh")
    TOP_K: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
