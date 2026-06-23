"""
Tests for prompt construction, context formatting, and response parsing
in LLMChain. The Azure chat client is mocked to avoid real API calls.
"""
from unittest.mock import MagicMock

import pytest

from app.models.response_models import ConfidenceLevel
from app.prompts.prompt_utils import determine_confidence, format_context
from app.services.llm_chain import LLMChain


def test_format_context_includes_source_and_page():
    chunks = [
        {
            "text": "Revenue grew 5%.",
            "source_file": "a.pdf",
            "page_number": 12,
            "similarity_score": 0.9,
        }
    ]
    context = format_context(chunks)
    assert "a.pdf" in context
    assert "Page 12" in context
    assert "Revenue grew 5%." in context


@pytest.mark.parametrize(
    "scores,threshold,expected",
    [
        ([], 0.75, "low"),
        ([0.5], 0.75, "low"),
        ([0.80], 0.75, "medium"),
        ([0.95], 0.75, "high"),
    ],
)
def test_determine_confidence(scores, threshold, expected):
    assert determine_confidence(scores, threshold) == expected


def test_llm_chain_answer_returns_structured_response(test_settings, monkeypatch):
    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = [
        {
            "text": "Net income was $93.7B.",
            "source_file": "apple.pdf",
            "page_number": 33,
            "similarity_score": 0.95,
        }
    ]

    monkeypatch.setattr(
        "app.services.llm_chain.AzureChatOpenAI",
        lambda **kwargs: MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(
                    content="Net income was $93.7B. [Confidence: High]",
                    response_metadata={"token_usage": {"total_tokens": 42}},
                )
            )
        ),
    )

    chain = LLMChain(settings=test_settings, retriever=fake_retriever)
    response = chain.answer(
        question="What was net income?", top_k=5, temperature=0.0, source_filter=None
    )

    assert response.answer == "Net income was $93.7B."
    assert response.confidence == ConfidenceLevel.HIGH
    assert response.tokens_used == 42
    assert len(response.citations) == 1
    assert response.citations[0].source_file == "apple.pdf"
