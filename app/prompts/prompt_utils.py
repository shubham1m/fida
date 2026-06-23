"""
Helpers that transform retrieved vector-store chunks into the text blocks
the prompt templates expect. Isolating this formatting logic keeps
llm_chain.py focused on orchestration rather than string assembly.
"""
from typing import Dict, List


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a numbered context block for prompt injection.

    Args:
        chunks: list of dicts with keys: text, source_file, page_number, similarity_score

    Returns:
        A "---"-delimited string where each excerpt is labelled with its
        source file, page number and relevance score, so the LLM can both
        ground its answer and produce accurate citations.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} | Source: {chunk['source_file']} | Page {chunk['page_number']} | "
            f"Relevance: {chunk['similarity_score']:.2f}]\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)


def determine_confidence(similarity_scores: List[float], threshold: float) -> str:
    """Derive a coarse confidence label from retrieval similarity scores.

    Rule of thumb used across the app:
    - No scores at all, or best score below threshold -> "low"
    - Best score comfortably above threshold (>= threshold + 0.1) -> "high"
    - Otherwise -> "medium"
    """
    if not similarity_scores:
        return "low"
    best = max(similarity_scores)
    if best < threshold:
        return "low"
    if best >= threshold + 0.1:
        return "high"
    return "medium"
