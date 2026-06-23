"""
Pydantic response schemas for the FastAPI routers.

These classes define the exact JSON contracts described in the
requirements doc (sections 5.1-5.3), so frontend and tests can rely on a
stable shape regardless of internal refactors.
"""
from enum import Enum
from typing import List

from pydantic import BaseModel


class ConfidenceLevel(str, Enum):
    """Discrete confidence buckets derived from retrieval similarity scores."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IngestResponse(BaseModel):
    """Returned by POST /ingest after a document has been chunked and indexed."""

    status: str
    filename: str
    pages_processed: int
    chunks_created: int
    embedding_model: str
    index_size: int


class Citation(BaseModel):
    """A single grounded source excerpt backing part of an answer."""

    source_file: str
    page_number: int
    excerpt: str
    similarity_score: float


class QueryResponse(BaseModel):
    """Returned by POST /query: the grounded answer plus its citations."""

    question: str
    answer: str
    citations: List[Citation]
    model: str
    tokens_used: int
    confidence: ConfidenceLevel


class DocumentSummary(BaseModel):
    """A single entry in the GET /docs listing."""

    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Returned by GET /docs."""

    documents: List[DocumentSummary]
    total_documents: int
    total_chunks: int


class DeleteResponse(BaseModel):
    """Returned by DELETE /docs and DELETE /docs/{filename}."""

    status: str
    detail: str


class HealthResponse(BaseModel):
    """Returned by GET /health."""

    status: str
    index_size: int
