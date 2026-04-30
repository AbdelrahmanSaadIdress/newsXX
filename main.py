from fastapi import FastAPI
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from dotenv import load_dotenv
load_dotenv()

from routers import base_router
from routers import scraping_router, daily_scrape_job  
from routers import session_router
from routers import rag_router
from routers import article_router

from models.ask.session_store import MongoSessionStore
from models.ask.rag_deps import RAGDeps
from models.analyzingANDtranslating.analyze_Trans_deps import A_TDeps

from helpers import get_settings

settings  = get_settings()
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ───────────────────────────────────────────────────────────────
    app.rag_deps      = RAGDeps(top_k=4)
    app.A_T_deps      = A_TDeps(top_k=6)
    app.session_store = MongoSessionStore(
        mongo_url       = settings.MONGO_URL,
        db_name         = settings.MONGO_DB,
        collection_name = "chat_sessions",
        ttl_days        = 30,
    )

    await app.session_store.ensure_indexes()

    scheduler.add_job(
        daily_scrape_job,
        trigger         = CronTrigger(hour=23, minute=0),
        kwargs          = {"a_tDeps": app.A_T_deps},
        id              = "daily_scrape",
        replace_existing= True,
    )
    scheduler.start()
    print("Scheduler started.")

    yield

    # ── shutdown ──────────────────────────────────────────────────────────────
    scheduler.shutdown()
    print("Scheduler stopped.")
    await app.session_store.close()


app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(scraping_router)
app.include_router(session_router)
app.include_router(rag_router)
app.include_router(article_router)
