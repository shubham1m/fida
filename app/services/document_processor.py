"""
PDF text extraction and chunking.

Encapsulated as a DocumentProcessor class (rather than free functions) so
that chunking configuration (size/overlap/separators) is owned by one
object and can be swapped out or subclassed for different document types
without touching call sites.
"""
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter


class EmptyDocumentError(ValueError):
    """Raised when a PDF contains no extractable text."""


@dataclass
class DocumentChunk:
    """A single chunk of extracted text plus the metadata needed to cite it back.

    Using a dataclass instead of a raw dict gives us typed attribute access
    (chunk.text, chunk.page_number, ...) throughout the rest of the pipeline.
    """

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    total_chunks: int = field(default=0)

    def to_metadata(self) -> dict:
        """Serialise the citation-relevant fields for storage alongside the embedding."""
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


class DocumentProcessor:
    """Extracts text from a PDF and splits it into citation-friendly chunks.

    chunk_size/chunk_overlap default to the values mandated by the
    requirements doc (800/150) but are configurable for testing with
    smaller synthetic documents.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def extract_pages(self, pdf_bytes: bytes) -> List[str]:
        """Extract raw text per page from a PDF byte stream using PyMuPDF.

        Returns a list where index i holds the text of page i+1. Raises
        EmptyDocumentError if every page is blank (e.g. a scanned image PDF
        with no OCR layer), since there is nothing to chunk or embed.
        """
        pages: List[str] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                pages.append(page.get_text())

        if not any(page_text.strip() for page_text in pages):
            raise EmptyDocumentError("PDF contains no extractable text.")
        return pages

    def chunk_document(self, pdf_bytes: bytes, source_file: str) -> List[DocumentChunk]:
        """Run the full extract -> split -> annotate pipeline for one PDF.

        Each chunk records its originating page number so that later
        citations can point a user to the exact page, and a running
        chunk_index/total_chunks pair for ordering and stats.
        """
        pages = self.extract_pages(pdf_bytes)

        chunks: List[DocumentChunk] = []
        for page_number, page_text in enumerate(pages, start=1):
            if not page_text.strip():
                continue
            for piece in self._splitter.split_text(page_text):
                chunks.append(
                    DocumentChunk(
                        text=piece,
                        source_file=source_file,
                        page_number=page_number,
                        chunk_index=len(chunks),
                    )
                )

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks
