from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME:str
    APP_VERSION:str

    MONGO_URL:str
    MONGO_DB:str
    MONGO_COL_1:str

    CHROMA_PATH:str
    CHROMA_COLL:str

    OPENAI_API_KEY:str
    OPENAI_API_URL:str
    OPENAI_GENERATION_MODEL_ID:str
    OPENAI_EMBEDDING_MODEL_ID:str


    PROVIDERS:str

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()