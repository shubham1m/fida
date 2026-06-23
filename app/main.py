"""
FastAPI application entrypoint.

Only responsible for app construction and router registration; all
business logic lives in app/services and app/routers so this file stays
a stable, easy-to-read composition root.
"""
import structlog
from fastapi import FastAPI

from app.routers import ingest, query

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))  # INFO level

app = FastAPI(
    title="Financial Document Intelligence Assistant",
    description="RAG-powered Q&A over financial filings with grounded, cited answers.",
    version="1.0.0",
    # The requirements doc reserves "/docs" for document management
    # (GET/DELETE), so FastAPI's interactive Swagger UI is moved to /swagger.
    docs_url="/swagger",
)

app.include_router(ingest.router)
app.include_router(query.router)
