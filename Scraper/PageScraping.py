from pyquery import PyQuery as pq
from helpers.Config import get_settings, Settings

from pymongo import MongoClient
import chromadb
import time

from stores.llm.LLM_Factory import LLMFactory

class PageScaraper:
    def __init__(self, main_page_url:str="https://www.ajnet.me"):
        self.main_page_url = main_page_url
        app_settings = get_settings()
        
        self.mongo_url = app_settings.MONGO_URL
        self.mongo_db = app_settings.MONGO_DB
        self.mongo_collection = app_settings.MONGO_COL_1

        self.chroma_path = app_settings.CHROMA_PATH
        self.chroma_collection = app_settings.CHROMA_COLL
        
        # ── MongoDB ──────────────────────────────────────────────────────────
        self.mongo_client = MongoClient(self.mongo_url)
        self.collection   = self.mongo_client[self.mongo_db][self.mongo_collection]
        # ── LLM (OpenAI for embeddings) ───────────────────────────────────────
        self.llm_provider = LLMFactory.create(
            provider="openai",
            config={
                "api_key": app_settings.OPENAI_API_KEY,   # add to your Settings
                "api_url": app_settings.OPENAI_API_URL, # optional, if using a custom base
            }
        )
        self.llm_provider.set_embedding_model(
            model_id       = app_settings.OPENAI_EMBEDDING_MODEL_ID,  # or text-embedding-ada-002
            embedding_size = 1536
        )
        # ── ChromaDB ─────────────────────────────────────────────────────────
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.chroma_col    = self.chroma_client.get_or_create_collection(
            name     = self.chroma_collection,
            metadata = {"hnsw:space": "cosine"},
        )

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
            time.sleep(0.5)
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

        content = "\n".join(content_parts)
        return content

    def scrape(self, href, title=None, description=None, current_day:str=None, Day:bool=True):
        page_full_url = self.main_page_url + href
        page = pq(url=page_full_url)
        
        if title is None:
            title = page("header h1").text()
        
        content_container = page(".wysiwyg--all-content")
        content = self.get_content(content_container)

        if Day:
            # Store in Mongo
            result = self.collection.insert_one({
                "title":title,
                "description":description,
                "content":content,
                "url":page_full_url,
                "current_day": current_day
            })
            mongo_id = str(result.inserted_id)
            # Chunk + Embed + Store in Chroma
            chunks = self.chunk_content(content)
            self.store_chunks_in_chroma(
                mongo_id = mongo_id,
                chunks   = chunks,
                metadata = {
                    "title":       title,
                    "url":         page_full_url,
                    "current_day": current_day,
                }
            )
            return None
        
        time.sleep(0.5)
        return {
            "title":title,
            "description":description,
            "content":content,
            "url":page_full_url,
            "current_day": current_day
        }




