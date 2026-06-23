"""
POST /ingest: accepts a PDF upload, extracts/chunks/embeds it, and stores
the result in the FAISS index.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.dependencies import ServiceContainer, get_container
from app.models.response_models import IngestResponse
from app.services.document_processor import EmptyDocumentError
from app.services.vector_store import IndexedChunk

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile, container: ServiceContainer = Depends(get_container)
) -> IngestResponse:
    """Ingest a single PDF: extract text, chunk it, embed it, and index it.

    Validates file type and size before doing any expensive work, then
    delegates extraction/chunking to DocumentProcessor, embedding to
    EmbeddingService, and persistence to VectorStoreManager.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    max_bytes = container.settings.max_upload_size_bytes
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {container.settings.max_upload_size_mb}MB upload limit.",
        )

    try:
        chunks = container.document_processor.chunk_document(pdf_bytes, source_file=file.filename)
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from this PDF.")

    embeddings = container.embedding_service.embed_documents([c.text for c in chunks])
    indexed_chunks = [
        IndexedChunk(
            text=c.text,
            source_file=c.source_file,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
        )
        for c in chunks
    ]
    container.vector_store.upsert(embeddings, indexed_chunks)

    pages_processed = max((c.page_number for c in chunks), default=0)
    return IngestResponse(
        status="success",
        filename=file.filename,
        pages_processed=pages_processed,
        chunks_created=len(chunks),
        embedding_model=container.embedding_service.model_name,
        index_size=container.vector_store.size,
    )
