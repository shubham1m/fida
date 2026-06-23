"""
Tests for similarity threshold filtering, top_k limiting, and source
filtering in the Retriever.
"""
from app.services.retriever import Retriever
from app.services.vector_store import IndexedChunk


class FakeEmbeddingService:
    """Stand-in for EmbeddingService that returns a fixed query vector."""

    def embed_query(self, text: str):
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_retrieve_filters_below_threshold(vector_store, fake_embedding):
    vector_store.upsert(
        [fake_embedding],
        [IndexedChunk(text="match", source_file="a.pdf", page_number=1, chunk_index=0)],
    )
    retriever = Retriever(vector_store, FakeEmbeddingService(), similarity_threshold=0.99)
    results = retriever.retrieve("question", top_k=5)
    assert len(results) == 1  # identical vectors -> similarity 1.0, passes threshold


def test_retrieve_respects_top_k(vector_store, fake_embedding):
    chunks = [
        IndexedChunk(text=f"chunk{i}", source_file="a.pdf", page_number=1, chunk_index=i)
        for i in range(5)
    ]
    vector_store.upsert([fake_embedding] * 5, chunks)
    retriever = Retriever(vector_store, FakeEmbeddingService(), similarity_threshold=0.0)
    results = retriever.retrieve("question", top_k=2)
    assert len(results) == 2


def test_retrieve_applies_source_filter(vector_store, fake_embedding):
    vector_store.upsert(
        [fake_embedding, fake_embedding],
        [
            IndexedChunk(text="a-chunk", source_file="a.pdf", page_number=1, chunk_index=0),
            IndexedChunk(text="b-chunk", source_file="b.pdf", page_number=1, chunk_index=0),
        ],
    )
    retriever = Retriever(vector_store, FakeEmbeddingService(), similarity_threshold=0.0)
    results = retriever.retrieve("question", top_k=5, source_filter="b.pdf")
    assert len(results) == 1
    assert results[0]["source_file"] == "b.pdf"
