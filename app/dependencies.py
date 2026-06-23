"""
Dependency wiring for FastAPI's dependency-injection system.

ServiceContainer instantiates every service once per process (the vector
store especially must be a singleton, since it owns the in-memory FAISS
index). Routers request services via get_container() rather than building
them directly, which keeps routers thin and makes it trivial to swap in
mocks during testing.
"""
from functools import lru_cache

from app.config import Settings, get_settings
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from app.services.llm_chain import LLMChain
from app.services.retriever import Retriever
from app.services.vector_store import VectorStoreManager

EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension


class ServiceContainer:
    """Holds one instance of every service, wired together with shared config."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService(settings)
        self.vector_store = VectorStoreManager(
            index_path=settings.faiss_index_path, embedding_dim=EMBEDDING_DIM
        )
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            similarity_threshold=settings.similarity_threshold,
        )
        self.llm_chain = LLMChain(settings=settings, retriever=self.retriever)


@lru_cache
def get_container() -> ServiceContainer:
    """Return the process-wide singleton ServiceContainer, built lazily on first use."""
    return ServiceContainer(get_settings())
