"""
Orchestrates the RAG chain: retrieve context, build the prompt, call
GPT-4o, and parse the response into the structured QueryResponse shape.

This is the one class that ties prompts + retriever + LLM client together,
so it is the natural place to log prompt versions for traceability
(requirement 6.4).
"""
import re

import structlog
from langchain_openai import AzureChatOpenAI

from app.config import Settings
from app.models.response_models import Citation, ConfidenceLevel, QueryResponse
from app.prompts.prompt_utils import determine_confidence, format_context
from app.prompts.rag_prompt import PROMPT_VERSION as RAG_PROMPT_VERSION
from app.prompts.rag_prompt import RAG_PROMPT_TEMPLATE
from app.prompts.system_prompt import PROMPT_VERSION as SYSTEM_PROMPT_VERSION
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.services.retriever import Retriever

logger = structlog.get_logger(__name__)


class LLMChain:
    """Runs the retrieve -> prompt -> generate -> parse pipeline for one question."""

    def __init__(self, settings: Settings, retriever: Retriever):
        self._settings = settings
        self._retriever = retriever
        self._client = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
        )

    def answer(
        self, question: str, top_k: int, temperature: float, source_filter: str | None
    ) -> QueryResponse:
        """Produce a grounded, cited answer for a single user question."""
        chunks = self._retriever.retrieve(question, top_k=top_k, source_filter=source_filter)

        context = format_context(chunks) if chunks else "No relevant context was found."
        source_label = chunks[0]["source_file"] if chunks else "the uploaded document"
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question, source=source_label)

        logger.info(
            "llm_chain.invoke",
            system_prompt_version=SYSTEM_PROMPT_VERSION,
            rag_prompt_version=RAG_PROMPT_VERSION,
            chunks_retrieved=len(chunks),
            top_k=top_k,
            temperature=temperature,
        )

        self._client.temperature = temperature
        response = self._client.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )

        answer_text = self._strip_confidence_marker(response.content)
        citations = [
            Citation(
                source_file=c["source_file"],
                page_number=c["page_number"],
                excerpt=c["text"][:300],
                similarity_score=c["similarity_score"],
            )
            for c in chunks
        ]
        confidence = determine_confidence(
            [c["similarity_score"] for c in chunks], self._settings.similarity_threshold
        )
        tokens_used = self._extract_token_usage(response)

        return QueryResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            model=self._settings.azure_openai_chat_deployment,
            tokens_used=tokens_used,
            confidence=ConfidenceLevel(confidence),
        )

    @staticmethod
    def _strip_confidence_marker(text: str) -> str:
        """Remove the trailing [Confidence: ...] marker; confidence is reported as its own field."""
        return re.sub(r"\[Confidence:\s*\w+\]\s*$", "", text).strip()

    @staticmethod
    def _extract_token_usage(response) -> int:
        """Pull total token usage out of the LangChain response metadata, defaulting to 0."""
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        return usage.get("total_tokens", 0)
