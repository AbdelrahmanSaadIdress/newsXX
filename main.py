# main.py
from fastapi import FastAPI
from routers import base_router
from routers import scraping_router, lifespan

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(scraping_router)