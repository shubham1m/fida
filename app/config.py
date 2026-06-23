"""
Centralised application configuration.

All runtime configuration is sourced from environment variables (via .env)
and validated through a single Pydantic Settings object. Using one
Settings class (rather than scattering os.getenv calls across the codebase)
gives us a single source of truth, type validation at startup, and easy
mocking in tests.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment variables.

    Each attribute maps 1:1 to an entry in .env.example. Pydantic validates
    types and raises at import time if a required variable is missing,
    so misconfiguration fails fast instead of surfacing as a runtime error
    deep inside a request handler.
    """

    # --- Azure OpenAI ---
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # --- App behaviour ---
    max_upload_size_mb: int = 50
    default_top_k: int = 5
    similarity_threshold: float = 0.75
    faiss_index_path: str = "./data/faiss_index"
    log_level: str = "INFO"

    # --- Frontend ---
    backend_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def max_upload_size_bytes(self) -> int:
        """Convert the configured MB limit into bytes for stream-size checks."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    lru_cache ensures the .env file is parsed only once per process; every
    caller (routers, services) shares the same validated configuration
    object instead of re-reading the environment repeatedly.
    """
    return Settings()
