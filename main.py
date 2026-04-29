# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger


from routers import base_router
from routers import scraping_router, lifespan, daily_scrape_job
from routers import session_router
from routers import rag_router

from dotenv import load_dotenv
load_dotenv()

from models.ask.session_store import MongoSessionStore
from models.ask.rag_deps import RAGDeps
from helpers import get_settings, Settings

settings = get_settings()

# ── shared singletons ────────────────────────────────────────

scheduler = AsyncIOScheduler()
@asynccontextmanager
async def lifespan(app: FastAPI):

    app.rag_deps     = RAGDeps(top_k=4)
    app.session_store = MongoSessionStore(
        mongo_url       = settings.MONGO_URL,
        db_name         = settings.MONGO_DB,
        collection_name = "chat_sessions",
        ttl_days        = 30,
    )

    # startup
    await app.session_store.ensure_indexes()       # create MongoDB indexes once
    scheduler.add_job(
            daily_scrape_job,
            trigger=CronTrigger(hour=23, minute=0),  # every day at 12:00 pm 
            id="daily_scrape",
            replace_existing=True,
        )
    scheduler.start()  
    print("Scheduler started.")
    
    yield

    
    scheduler.shutdown()
    print("Scheduler stopped.")
    # shutdown
    await app.session_store.close()


app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(scraping_router)
app.include_router(session_router)
app.include_router(rag_router)

