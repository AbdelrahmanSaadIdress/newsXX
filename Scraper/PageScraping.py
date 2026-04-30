from pyquery import PyQuery as pq
from helpers.Config import get_settings, Settings

from pymongo import MongoClient
import chromadb
import time

from stores.llm.LLM_Factory import LLMFactory
from models.analyzingANDtranslating.analyze_Trans_deps import A_TDeps

from sentence_transformers import SentenceTransformer


class PageScaraper:
    def __init__(self, main_page_url: str = "https://www.ajnet.me", a_tDeps: A_TDeps = None):
        self.main_page_url = main_page_url
        self.a_tDeps = a_tDeps
        app_settings = get_settings()

        self.mongo_url        = app_settings.MONGO_URL
        self.mongo_db         = app_settings.MONGO_DB
        self.mongo_collection = app_settings.MONGO_COL_1

        self.chroma_path       = app_settings.CHROMA_PATH
        self.chroma_collection = app_settings.CHROMA_COLL

        # ── MongoDB ──────────────────────────────────────────────────────────
        self.mongo_client     = MongoClient(self.mongo_url)
        self.collection       = self.mongo_client[self.mongo_db][self.mongo_collection]
        self.analysis_col     = self.mongo_client[self.mongo_db][app_settings.MONGO_COL_ANALYSIS]
        self.statistics_col   = self.mongo_client[self.mongo_db][app_settings.MONGO_COL_STATS]

        # Ensure url is a unique index on the analysis collection
        self.analysis_col.create_index("url", unique=True)

        # ── LLM (OpenAI for embeddings) ───────────────────────────────────────
        self.llm_provider = LLMFactory.create(
            provider="openai",
            config={
                "api_key": app_settings.OPENAI_API_KEY,
                "api_url": app_settings.OPENAI_API_URL,
            },
        )
        self.llm_provider.set_embedding_model(
            model_id       = app_settings.OPENAI_EMBEDDING_MODEL_ID,
            embedding_size = 1536 ,
        )

        # self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


        # ── ChromaDB ─────────────────────────────────────────────────────────
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.chroma_col    = self.chroma_client.get_or_create_collection(
            name     = self.chroma_collection,
            metadata = {"hnsw:space": "cosine"},
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def chunk_content(self, content: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
        chunks, current = [], ""

        for paragraph in paragraphs:
            if len(current) + len(paragraph) > chunk_size and current:
                chunks.append(current.strip())
                current = current[-overlap:] + " " + paragraph
            else:
                current = current + "\n" + paragraph if current else paragraph

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def store_chunks_in_chroma(self, mongo_id: str, chunks: list[str], metadata: dict):
        if not chunks:
            return

        ids        = []
        embeddings = []
        metadatas  = []
        documents  = []

        for i, chunk in enumerate(chunks):
            embedding = self.llm_provider.embed_text(chunk)
            # embedding = self.embedding_model.encode(chunk).tolist()
            if embedding is None:
                continue

            ids.append(f"{mongo_id}_chunk_{i}")
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append({
                "mongo_id":     str(mongo_id),
                "title":        str(metadata.get("title") or ""),
                "url":          str(metadata.get("url") or ""),
                "current_day":  str(metadata.get("current_day") or ""),
                "chunk_index":  int(i),
                "total_chunks": int(len(chunks)),
            })
            time.sleep(2)

        if ids:
            self.chroma_col.add(
                ids        = ids,
                embeddings = embeddings,
                documents  = documents,
                metadatas  = metadatas,
            )

    def get_content(self, content_container):
        content_parts = []
        for element in content_container.children().items():
            text = element.text().strip()
            if text:
                content_parts.append(text)
        return "\n".join(content_parts)

    # ── analysis & storage ────────────────────────────────────────────────────

    def _analyse_and_store(
        self,
        content: str,
        title: str,
        description: str,
        url: str,
        current_day: str,
    ) -> dict | None:
        """
        Run A_TDeps analysis on content, persist results, and return a
        summary dict for the daily-post accumulator.

        Returns None if analysis fails or a_tDeps is not available.
        """
        if self.a_tDeps is None:
            return None

        news_details = self.a_tDeps.generate_analysis(content)
        if news_details is None:
            return None

        # ── persist to analysis collection (url is unique index) ─────────────
        analysis_doc = {
            "url":             url,
            "title":           title,
            "description":     description,
            "current_day":     current_day,
            "story_title":     news_details.story_title,
            "story_keywords":  news_details.story_keywords,
            "story_summary":   news_details.story_summary,
            "story_category":  news_details.story_category,
            "story_entities":  [e.model_dump() for e in news_details.story_entities],
        }
        try:
            self.analysis_col.insert_one(analysis_doc)
        except Exception as e:
            # Duplicate URL or other write error — log and continue
            print(f"[PageScraper] analysis insert skipped for {url}: {e}")

        # ── persist category to statistics collection ─────────────────────────
        self.statistics_col.insert_one({
            "url":            url,
            "current_day":    current_day,
            "story_category": news_details.story_category,
        })

        # ── return lightweight summary for the daily post ─────────────────────
        return {
            "title":         title,
            "description":   description,
            "url":           url,
            "story_summary": news_details.story_summary,   # list[str]
        }

    # ── main entry point ──────────────────────────────────────────────────────

    def scrape(
        self,
        href,
        title=None,
        description=None,
        current_day: str = None,
        Day: bool = True,
    ):
        page_full_url = self.main_page_url + href
        page          = pq(url=page_full_url)

        if title is None:
            title = page("header h1").text()

        content_container = page(".wysiwyg--all-content")
        content           = self.get_content(content_container)

        if Day:
            # ── Store article in Mongo ────────────────────────────────────────
            result   = self.collection.insert_one({
                "title":       title,
                "description": description,
                "content":     content,
                "url":         page_full_url,
                "current_day": current_day,
            })
            mongo_id = str(result.inserted_id)

            # ── Chunk + Embed + Store in Chroma ───────────────────────────────
            chunks = self.chunk_content(content)
            self.store_chunks_in_chroma(
                mongo_id = mongo_id,
                chunks   = chunks,
                metadata = {
                    "title":       title,
                    "url":         page_full_url,
                    "current_day": current_day,
                },
            )

            # ── Analyse + store analysis & statistics ─────────────────────────
            summary = self._analyse_and_store(
                content     = content,
                title       = title,
                description = description,
                url         = page_full_url,
                current_day = current_day,
            )

            # Return summary so BulkScraper can accumulate it
            return summary   # None if analysis unavailable

        time.sleep(0.5)
        return {
            "title":       title,
            "description": description,
            "content":     content,
            "url":         page_full_url,
            "current_day": current_day,
        }