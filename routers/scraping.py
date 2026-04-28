# routers/scraping.py

from fastapi import APIRouter, BackgroundTasks, Query, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from Scraper.BulkScraping import BulkScraper
from Scraper.PageScraping import PageScaraper

# ── Constants ─────────────────────────────────────────────────────────────────
MAIN_PAGE_URL = "https://www.ajnet.me/news"
NEWS_FEED_URL = f"{MAIN_PAGE_URL}"

# ── Schemas ───────────────────────────────────────────────────────────────────
class ScrapeUrlRequest(BaseModel):
    url: str
    save_to_db: bool = True

class ScrapeResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

# ── Core task ─────────────────────────────────────────────────────────────────
async def _run_bulk(num_of_samples: int, day_only: bool):
    scraper = BulkScraper(headless=False)
    try:
        await run_in_threadpool(
            scraper.scrape_bulk,
            url=NEWS_FEED_URL,
            Day=day_only,
            num_of_samples=num_of_samples
        )
    finally:
        scraper.driver.quit()


# ── Scheduled job (runs daily) ────────────────────────────────────────────────
async def daily_scrape_job():
    print("Scheduler: starting daily scrape...")
    await _run_bulk(num_of_samples=500, day_only=True)
    print("Scheduler: daily scrape finished.")

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app):
    # runs at startup
    scheduler.add_job(
        daily_scrape_job,
        trigger=CronTrigger(hour=23, minute=0),  # every day at 12:00 pm 
        id="daily_scrape",
        replace_existing=True,
    )
    scheduler.start()
    print("Scheduler started.")

    yield  # server is running here

    # runs at shutdown
    scheduler.shutdown()
    print("Scheduler stopped.")

# ── Router ────────────────────────────────────────────────────────────────────
scraping_router = APIRouter(
    prefix="/api/v1/scrape",
    tags=["Scraping"],
)

@scraping_router.post("/bulk-finetune", response_model=ScrapeResponse)
async def scrape_bulk_for_finetune(
    background_tasks: BackgroundTasks,
    num_of_samples: int = Query(default=50, ge=1, le=500),
):
    background_tasks.add_task(_run_bulk, num_of_samples=num_of_samples, day_only=False)
    return ScrapeResponse(
        status="accepted",
        message=f"Scraping {num_of_samples} articles in the background. Results → assets/news_sample.jsonl",
    )

@scraping_router.post("/daily", response_model=ScrapeResponse)
async def scrape_daily(
    background_tasks: BackgroundTasks,
    num_of_samples: int = Query(default=5, ge=1, le=500),
):
    background_tasks.add_task(_run_bulk, num_of_samples=num_of_samples, day_only=True)
    return ScrapeResponse(
        status="accepted",
        message="Scraping today's articles in the background. Data will be saved to MongoDB and ChromaDB.",
    )

@scraping_router.post("/url", response_model=ScrapeResponse)
async def scrape_specific_url(body: ScrapeUrlRequest):
    page_scraper = PageScaraper(main_page_url=MAIN_PAGE_URL)

    href = body.url
    if href.startswith(MAIN_PAGE_URL):
        href = href[len(MAIN_PAGE_URL):]

    try:
        result = await run_in_threadpool(
            page_scraper.scrape,
            href=href,
            title=None,
            description=None,
            current_day=None,
            Day=body.save_to_db,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if body.save_to_db:
        return ScrapeResponse(status="success", message="Article scraped and saved to MongoDB + ChromaDB.")

    return ScrapeResponse(status="success", message="Article scraped successfully.", data=result)