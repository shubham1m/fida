"""
Tests for FAISS upsert, retrieval, persistence, and deletion.
"""
from app.services.vector_store import IndexedChunk, VectorStoreManager


def _make_chunk(text="hello", source="a.pdf", page=1, idx=0) -> IndexedChunk:
    return IndexedChunk(text=text, source_file=source, page_number=page, chunk_index=idx)


def test_upsert_increases_size(vector_store, fake_embedding):
    vector_store.upsert([fake_embedding], [_make_chunk()])
    assert vector_store.size == 1


def test_search_returns_matching_chunk(vector_store, fake_embedding):
    vector_store.upsert([fake_embedding], [_make_chunk(text="needle")])
    results = vector_store.search(fake_embedding, top_k=1)
    assert len(results) == 1
    chunk, score = results[0]
    assert chunk.text == "needle"
    assert score > 0.9


def test_persistence_survives_reload(temp_index_path, fake_embedding):
    store = VectorStoreManager(index_path=temp_index_path, embedding_dim=8)
    store.upsert([fake_embedding], [_make_chunk(text="persisted")])

    reloaded = VectorStoreManager(index_path=temp_index_path, embedding_dim=8)
    assert reloaded.size == 1
    results = reloaded.search(fake_embedding, top_k=1)
    assert results[0][0].text == "persisted"


def test_delete_by_source_removes_only_matching_chunks(vector_store, fake_embedding):
    vector_store.upsert(
        [fake_embedding, fake_embedding],
        [_make_chunk(source="a.pdf"), _make_chunk(source="b.pdf")],
    )
    removed = vector_store.delete_by_source("a.pdf")
    assert removed == 1
    assert vector_store.size == 1
    assert vector_store.list_documents() == {"b.pdf": 1}


def test_clear_empties_index(vector_store, fake_embedding):
    vector_store.upsert([fake_embedding], [_make_chunk()])
    vector_store.clear()
    assert vector_store.size == 0


def test_list_documents_counts_chunks_per_source(vector_store, fake_embedding):
    vector_store.upsert(
        [fake_embedding, fake_embedding, fake_embedding],
        [
            _make_chunk(source="a.pdf", idx=0),
            _make_chunk(source="a.pdf", idx=1),
            _make_chunk(source="b.pdf", idx=0),
        ],
    )
    assert vector_store.list_documents() == {"a.pdf": 2, "b.pdf": 1}
