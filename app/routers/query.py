"""
POST /query plus the document-management endpoints (GET/DELETE /docs,
GET /health). Grouped here since they all operate on the same indexed
corpus rather than the ingestion pipeline.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import ServiceContainer, get_container
from app.models.request_models import QueryRequest
from app.models.response_models import (
    DeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    HealthResponse,
    QueryResponse,
)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest, container: ServiceContainer = Depends(get_container)
) -> QueryResponse:
    """Answer a natural-language question grounded in the indexed documents."""
    if container.vector_store.size == 0:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet.")

    return container.llm_chain.answer(
        question=request.question,
        top_k=request.top_k,
        temperature=request.temperature,
        source_filter=request.source_filter,
    )


@router.get("/docs", response_model=DocumentListResponse)
async def list_documents(
    container: ServiceContainer = Depends(get_container),
) -> DocumentListResponse:
    """List every indexed document along with its chunk count."""
    counts = container.vector_store.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentSummary(filename=name, chunk_count=count) for name, count in counts.items()
        ],
        total_documents=len(counts),
        total_chunks=container.vector_store.size,
    )


@router.delete("/docs/{filename}", response_model=DeleteResponse)
async def delete_document(
    filename: str, container: ServiceContainer = Depends(get_container)
) -> DeleteResponse:
    """Remove a single document (all of its chunks) from the index."""
    removed = container.vector_store.delete_by_source(filename)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in index.")
    return DeleteResponse(status="success", detail=f"Removed {removed} chunks for '{filename}'.")


@router.delete("/docs", response_model=DeleteResponse)
async def clear_documents(
    container: ServiceContainer = Depends(get_container),
) -> DeleteResponse:
    """Clear the entire index, removing all documents."""
    container.vector_store.clear()
    return DeleteResponse(status="success", detail="Index cleared.")


@router.get("/health", response_model=HealthResponse)
async def health_check(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
    """Lightweight liveness/readiness probe used by Docker healthchecks."""
    return HealthResponse(status="ok", index_size=container.vector_store.size)
