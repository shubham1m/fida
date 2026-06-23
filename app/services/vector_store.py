"""
FAISS-backed vector store management.

VectorStoreManager owns the on-disk FAISS index and the metadata needed to
turn a similarity-search hit back into a citation. Keeping persistence,
upsert and deletion logic inside one class avoids spreading FAISS-specific
code across routers and services.
"""
import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


@dataclass
class IndexedChunk:
    """A chunk as stored in the index, paired with its embedding-free metadata.

    The raw vector lives inside the FAISS index itself; this object holds
    everything else needed to reconstruct a citation or list documents.
    """

    text: str
    source_file: str
    page_number: int
    chunk_index: int


class VectorStoreManager:
    """Manages a persisted FAISS index plus a parallel metadata store.

    FAISS only stores vectors and returns integer ids; it has no concept of
    metadata or deletion-by-filter. This class maintains a metadata list
    addressed by the same ids FAISS uses, and rebuilds the index on delete
    operations since plain FAISS indices don't support efficient removal.
    """

    METADATA_FILENAME = "metadata.pkl"
    INDEX_FILENAME = "index.faiss"

    def __init__(self, index_path: str, embedding_dim: int = 1536):
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self._metadata: List[IndexedChunk] = []
        self._index: faiss.Index = faiss.IndexFlatIP(embedding_dim)
        os.makedirs(self.index_path, exist_ok=True)
        self._load_if_exists()

    @property
    def size(self) -> int:
        """Total number of chunks currently indexed."""
        return len(self._metadata)

    def _index_file(self) -> str:
        return os.path.join(self.index_path, self.INDEX_FILENAME)

    def _metadata_file(self) -> str:
        return os.path.join(self.index_path, self.METADATA_FILENAME)

    def _load_if_exists(self) -> None:
        """Restore a previously persisted index/metadata pair, if present."""
        if os.path.exists(self._index_file()) and os.path.exists(self._metadata_file()):
            self._index = faiss.read_index(self._index_file())
            with open(self._metadata_file(), "rb") as f:
                self._metadata = pickle.load(f)

    def persist(self) -> None:
        """Write the current index and metadata to disk, surviving restarts."""
        faiss.write_index(self._index, self._index_file())
        with open(self._metadata_file(), "wb") as f:
            pickle.dump(self._metadata, f)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise vectors so inner-product search behaves as cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def upsert(self, embeddings: List[List[float]], chunks: List[IndexedChunk]) -> None:
        """Add new vectors and their metadata to the index, then persist.

        FAISS has no native "upsert by key" for a flat index, so re-ingesting
        the same filename will simply append duplicate chunks; callers that
        want clean replacement should delete the existing document first via
        VectorStoreManager.delete_by_source.
        """
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must be the same length")

        vectors = self._normalize(np.array(embeddings, dtype="float32"))
        self._index.add(vectors)
        self._metadata.extend(chunks)
        self.persist()

    def search(
        self, query_embedding: List[float], top_k: int, source_filter: Optional[str] = None
    ) -> List[Tuple[IndexedChunk, float]]:
        """Return the top_k most similar chunks to a query embedding.

        When source_filter is set, over-fetches from FAISS and filters by
        source_file in Python, since a flat index has no native predicate
        support.
        """
        if self.size == 0:
            return []

        query_vec = self._normalize(np.array([query_embedding], dtype="float32"))
        fetch_k = self.size if source_filter else min(top_k, self.size)
        scores, indices = self._index.search(query_vec, fetch_k)

        results: List[Tuple[IndexedChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            chunk = self._metadata[idx]
            if source_filter and chunk.source_file != source_filter:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    def list_documents(self) -> Dict[str, int]:
        """Return a mapping of source_file -> chunk count for GET /docs."""
        counts: Dict[str, int] = {}
        for chunk in self._metadata:
            counts[chunk.source_file] = counts.get(chunk.source_file, 0) + 1
        return counts

    def delete_by_source(self, source_file: str) -> int:
        """Remove all chunks belonging to a single source file and rebuild the index.

        Returns the number of chunks removed. Rebuilding from the surviving
        metadata is the simplest correct approach for a flat (non-IDMap)
        FAISS index, and is acceptable given expected corpus sizes.
        """
        kept = [c for c in self._metadata if c.source_file != source_file]
        removed = len(self._metadata) - len(kept)
        if removed:
            self._rebuild(kept)
        return removed

    def clear(self) -> None:
        """Remove every chunk from the index and delete persisted files on disk."""
        self._metadata = []
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        if os.path.isdir(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

    def _rebuild(self, surviving_metadata: List[IndexedChunk]) -> None:
        """Recreate the FAISS index from scratch using only surviving chunks' vectors.

        Note: this re-reads vectors from the existing index by their old
        positions before they are dropped, then re-adds them in order.
        """
        if not surviving_metadata:
            self.clear()
            return

        surviving_ids = [i for i, c in enumerate(self._metadata) if c in surviving_metadata]
        vectors = np.array(
            [self._index.reconstruct(i) for i in surviving_ids], dtype="float32"
        )

        new_index = faiss.IndexFlatIP(self.embedding_dim)
        new_index.add(vectors)

        self._index = new_index
        self._metadata = surviving_metadata
        self.persist()
