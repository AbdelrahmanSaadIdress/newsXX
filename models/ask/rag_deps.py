from __future__ import annotations
import chromadb
from helpers.Config import get_settings
from stores.llm.LLM_Factory import LLMFactory


class RAGDeps:
    """
    Initialised once at application startup and shared across all requests.
    Wraps the project's own OpenAIProvider (via LLMFactory) and ChromaDB.
    """

    def __init__(self, top_k: int = 6):
        settings   = get_settings()
        self.top_k = top_k

        # ── provider (your own OpenAIProvider via LLMFactory) ────────────────
        self.embed_provider = LLMFactory.create(
            provider=settings.PROVIDERS,
            config={
                "api_key": settings.OPENAI_API_KEY,
                "api_url": settings.OPENAI_API_URL,
            },
        )
        
        self.embed_provider.set_embedding_model(
            model_id=settings.OPENAI_EMBEDDING_MODEL_ID,
            embedding_size=1536,
        )

        self.generation_provider = LLMFactory.create(
            provider=settings.PROVIDERS,
            config={
                "api_key": settings.OPENAI_API_KEY,
                "api_url": settings.OPENAI_API_URL,
            },
        )
        
        self.generation_provider.set_generation_model(
            model_id=settings.OPENAI_GENERATION_MODEL_ID,
        )

        # ── ChromaDB ─────────────────────────────────────────────────────────
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection    = self.chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLL,
            metadata={"hnsw:space": "cosine"},
        )