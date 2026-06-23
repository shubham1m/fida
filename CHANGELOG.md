# Changelog

All notable changes to the prompt templates and application are documented here.

## [Unreleased]

### Added
- Initial implementation of the Financial Document Intelligence Assistant: FastAPI backend (ingest/query/docs/health endpoints), Streamlit frontend, FAISS-backed RAG pipeline, Docker/Compose setup, GitHub Actions CI/CD, and pytest suite (92% coverage).
- `system_prompt.py` v1.0 — analyst persona with strict grounding rules, mandatory citation format, and confidence-level reporting.
- `rag_prompt.py` v1.0 — per-query template structuring answers as Direct Answer → Supporting Evidence → Citations.
