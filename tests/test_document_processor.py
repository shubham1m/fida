"""
Tests for PDF extraction, chunking size/overlap, metadata attachment, and
empty-PDF handling.
"""
import pytest

from app.services.document_processor import DocumentProcessor, EmptyDocumentError


def test_extract_pages_returns_text_per_page(sample_pdf_bytes):
    processor = DocumentProcessor()
    pages = processor.extract_pages(sample_pdf_bytes)
    assert len(pages) == 3
    assert "Apple" in pages[0]


def test_extract_pages_raises_on_empty_pdf(empty_pdf_bytes):
    processor = DocumentProcessor()
    with pytest.raises(EmptyDocumentError):
        processor.extract_pages(empty_pdf_bytes)


def test_chunk_document_attaches_metadata(sample_pdf_bytes):
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    chunks = processor.chunk_document(sample_pdf_bytes, source_file="test.pdf")

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.source_file == "test.pdf"
        assert chunk.page_number >= 1
        assert chunk.total_chunks == len(chunks)


def test_chunk_document_respects_page_boundaries(sample_pdf_bytes):
    processor = DocumentProcessor()
    chunks = processor.chunk_document(sample_pdf_bytes, source_file="test.pdf")
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2, 3}


def test_chunk_index_is_sequential(sample_pdf_bytes):
    processor = DocumentProcessor()
    chunks = processor.chunk_document(sample_pdf_bytes, source_file="test.pdf")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
