from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    app_env: str = "development"

    # ---------------------------------------------------------
    # PostgreSQL
    # ---------------------------------------------------------

    database_url: str

    # ---------------------------------------------------------
    # Redis
    # ---------------------------------------------------------

    redis_url: str

    # ---------------------------------------------------------
    # Qdrant
    # ---------------------------------------------------------

    qdrant_url: str
    qdrant_collection_name: str = "knowledge_chunks"

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    embedding_provider: str = "gemini"

    # ---------------------------------------------------------
    # Gemini
    # ---------------------------------------------------------

    gemini_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()