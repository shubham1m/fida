"""
Integration tests for all FastAPI endpoints: status codes, response
schemas, and error cases. The ServiceContainer dependency is overridden
with a lightweight fake so no real Azure OpenAI calls are made.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_container
from app.main import app
from app.models.response_models import ConfidenceLevel, QueryResponse


@pytest.fixture
def fake_container(tmp_path):
    """A MagicMock standing in for ServiceContainer, with just enough real
    behaviour wired up to exercise router logic without external calls."""
    container = MagicMock()
    container.settings = Settings(
        azure_openai_api_key="test",
        azure_openai_endpoint="https://test.openai.azure.com/",
        max_upload_size_mb=1,
    )
    container.vector_store.size = 0
    container.vector_store.list_documents.return_value = {}
    return container


@pytest.fixture
def client(fake_container):
    app.dependency_overrides[get_container] = lambda: fake_container
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_check(client, fake_container):
    fake_container.vector_store.size = 3
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "index_size": 3}


def test_ingest_rejects_non_pdf(client):
    response = client.post(
        "/ingest", files={"file": ("test.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_ingest_rejects_oversized_file(client, fake_container):
    big_content = b"x" * (2 * 1024 * 1024)  # 2MB, limit is 1MB
    response = client.post(
        "/ingest", files={"file": ("big.pdf", big_content, "application/pdf")}
    )
    assert response.status_code == 413


def test_query_returns_400_when_index_empty(client, fake_container):
    fake_container.vector_store.size = 0
    response = client.post("/query", json={"question": "What is the revenue?"})
    assert response.status_code == 400


def test_query_returns_answer_when_index_populated(client, fake_container):
    fake_container.vector_store.size = 5
    fake_container.llm_chain.answer.return_value = QueryResponse(
        question="What is the revenue?",
        answer="Revenue was $391B.",
        citations=[],
        model="gpt-4o",
        tokens_used=10,
        confidence=ConfidenceLevel.HIGH,
    )
    response = client.post("/query", json={"question": "What is the revenue?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Revenue was $391B."


def test_list_documents(client, fake_container):
    fake_container.vector_store.list_documents.return_value = {"a.pdf": 3, "b.pdf": 2}
    fake_container.vector_store.size = 5
    response = client.get("/docs")
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] == 2
    assert data["total_chunks"] == 5


def test_delete_document_not_found(client, fake_container):
    fake_container.vector_store.delete_by_source.return_value = 0
    response = client.delete("/docs/missing.pdf")
    assert response.status_code == 404


def test_delete_document_success(client, fake_container):
    fake_container.vector_store.delete_by_source.return_value = 3
    response = client.delete("/docs/a.pdf")
    assert response.status_code == 200


def test_clear_documents(client, fake_container):
    response = client.delete("/docs")
    assert response.status_code == 200
    fake_container.vector_store.clear.assert_called_once()
