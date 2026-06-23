"""
Retrieval logic: turns a question into a filtered, scored list of chunks.

Separated from VectorStoreManager so similarity-threshold filtering and
embedding-of-the-query are easy to unit test independently of FAISS
internals.
"""
from typing import Dict, List, Optional

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreManager


class Retriever:
    """Combines query embedding + vector search + relevance filtering.

    Encapsulating this as its own class lets llm_chain.py ask a single
    question ("give me relevant chunks for this query") without knowing
    anything about FAISS or embedding models.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_service: EmbeddingService,
        similarity_threshold: float,
    ):
        self._vector_store = vector_store
        self._embeddings = embedding_service
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self, question: str, top_k: int, source_filter: Optional[str] = None
    ) -> List[Dict]:
        """Embed the question, search the index, and drop low-similarity chunks.

        Returns a list of dicts (text, source_file, page_number,
        similarity_score) ready for prompt_utils.format_context, sorted by
        descending similarity.
        """
        query_embedding = self._embeddings.embed_query(question)
        hits = self._vector_store.search(query_embedding, top_k=top_k, source_filter=source_filter)

        results = [
            {
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "similarity_score": score,
            }
            for chunk, score in hits
            if score >= self.similarity_threshold
        ]
        return sorted(results, key=lambda r: r["similarity_score"], reverse=True)
