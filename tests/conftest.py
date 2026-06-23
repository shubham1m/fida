"""
Shared pytest fixtures: a synthetic PDF, a temp FAISS index directory,
and mocked Azure OpenAI services so the test suite never makes real API
calls (and never costs money in CI).
"""
import io
from typing import List

import fitz  # PyMuPDF
import pytest

from app.config import Settings
from app.services.vector_store import VectorStoreManager

EMBEDDING_DIM = 8  # small dimension keeps test vectors cheap and readable


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Build a 3-page in-memory PDF with known, distinct text per page."""
    doc = fitz.open()
    page_texts = [
        "Apple reported total net sales of $391.0 billion in fiscal 2024.",
        "Net income for fiscal 2024 was $93.7 billion, up from prior year.",
        "Forward-looking statements involve risks and uncertainties.",
    ]
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """Build a single blank page PDF with no text content."""
    doc = fitz.open()
    doc.new_page()
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()


@pytest.fixture
def temp_index_path(tmp_path):
    """Provide an isolated FAISS index directory, cleaned up automatically by pytest."""
    return str(tmp_path / "faiss_index")


@pytest.fixture
def vector_store(temp_index_path) -> VectorStoreManager:
    """A fresh VectorStoreManager backed by a temp directory and small embedding dim."""
    return VectorStoreManager(index_path=temp_index_path, embedding_dim=EMBEDDING_DIM)


@pytest.fixture
def fake_embedding() -> List[float]:
    """A deterministic unit-ish vector standing in for a real embedding."""
    return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def test_settings(monkeypatch) -> Settings:
    """Settings instance with dummy Azure credentials, safe for use in tests."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    return Settings()
