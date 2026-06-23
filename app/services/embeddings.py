"""
Thin wrapper around the Azure OpenAI embedding client.

Wrapping LangChain's AzureOpenAIEmbeddings in our own EmbeddingService class
means the rest of the app depends on a stable interface (embed_documents /
embed_query) rather than on LangChain's class directly, and retry/backoff
policy lives in one place.
"""
from typing import List

from langchain_openai import AzureOpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings


class EmbeddingService:
    """Generates vector embeddings for text using Azure OpenAI.

    Retries transient failures (e.g. rate limiting) up to 3 times with
    exponential backoff, per the error-handling requirement in section 5.1.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_embedding_deployment,
        )

    @property
    def model_name(self) -> str:
        """Expose the configured deployment name for response/audit metadata."""
        return self._settings.azure_openai_embedding_deployment

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunk texts at ingestion time."""
        return self._client.embed_documents(texts)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def embed_query(self, text: str) -> List[float]:
        """Embed a single user question at query time."""
        return self._client.embed_query(text)
