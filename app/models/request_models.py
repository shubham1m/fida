"""
Pydantic request schemas for the FastAPI routers.

Keeping request schemas separate from response schemas (see
response_models.py) makes the API contract explicit and lets FastAPI
auto-generate accurate OpenAPI docs for clients.
"""
from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload accepted by POST /query.

    Mirrors the JSON shape defined in the requirements doc section 5.2.
    """

    question: str = Field(..., min_length=1, description="Natural language question to answer.")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of chunks to retrieve.")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM sampling temperature.")
    source_filter: Optional[str] = Field(
        default=None, description="Restrict retrieval to a single source filename."
    )
