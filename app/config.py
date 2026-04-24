"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for DocMind API."""

    database_url: str = "postgresql+asyncpg://postgres:root@localhost:5432/docmind"
    ollama_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "qwen2.5:14b"
    chroma_persist_dir: str = "./chroma_store"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 20
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 4
    app_env: str = "development"
    app_version: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
