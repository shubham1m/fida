# Financial Document Intelligence Assistant

## 1. Project Summary
A containerised GenAI application that lets financial analysts ask natural-language questions over financial filings (10-K/10-Q, annual reports, earnings releases). A RAG pipeline retrieves the most relevant excerpts from an indexed document corpus and an engineered prompt grounds GPT-4o's answer in those excerpts, returning page-level citations and a confidence indicator so every claim can be verified.

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User (Browser)                    │
└────────────────────┬────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────┐
│              Streamlit Frontend (Port 8501)          │
│  - File upload (PDF)                                │
│  - Chat interface                                   │
│  - Source citation display                          │
└────────────────────┬────────────────────────────────┘
                     │ REST API calls
┌────────────────────▼────────────────────────────────┐
│             FastAPI Backend (Port 8000)              │
│  POST /ingest   → Ingest & index document           │
│  POST /query    → RAG query + LLM answer            │
│  GET  /docs     → List indexed documents            │
│  DELETE /docs   → Clear index                       │
└──────┬─────────────────────────┬───────────────────┘
       │                         │
┌──────▼──────┐         ┌────────▼────────────────────┐
│    FAISS    │         │     Azure OpenAI             │
│ Vector Store│         │  - GPT-4o (chat completion) │
│  (persisted │         │  - text-embedding-3-small   │
│   to disk)  │         └─────────────────────────────┘
└─────────────┘
```

### RAG pipeline
PDF upload → text extraction (PyMuPDF) → chunking (800/150 overlap) → embedding (Azure OpenAI) → FAISS index → on query: embed question → similarity search (top_k) → filter by threshold → engineered prompt → GPT-4o → structured, cited answer.

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | Azure OpenAI (GPT-4o) |
| Embeddings | Azure OpenAI (text-embedding-3-small) |
| Vector Store | FAISS (local, persisted) |
| RAG Framework | LangChain |
| API Backend | FastAPI |
| Frontend | Streamlit |
| Containerisation | Docker + Docker Compose |
| Cloud Deploy | Azure Container Apps |
| CI/CD | GitHub Actions |
| Testing | pytest |

## 4. Local Setup

```bash
git clone <repo-url>
cd fida
cp .env.example .env        # fill in your Azure OpenAI credentials
docker compose up --build
```

Open http://localhost:8501 for the chat UI, or http://localhost:8000/swagger for the API docs.

## 5. Usage Guide
1. Upload a PDF filing via the sidebar "Upload a PDF filing" control, then click "Ingest document".
2. Ask a question in the chat box, e.g. "What was total revenue in FY2024?"
3. Expand "Sources" under the answer to see the exact page-level excerpts the answer was grounded in, along with a confidence indicator.

## 6. Prompt Engineering Notes
All prompts live in `app/prompts/` and are version-stamped (`PROMPT_VERSION`) so every LLM call can be traced back to the exact wording used:
- `system_prompt.py` defines the analyst persona and hard grounding rules (no prior knowledge, mandatory citations, explicit "insufficient context" fallback).
- `rag_prompt.py` is the per-query template that injects retrieved context and instructs the model to structure its answer as Direct Answer → Supporting Evidence → Citations.
- `prompt_utils.py` formats retrieved chunks into a numbered, citation-ready context block and derives a coarse confidence label from similarity scores.

## 7. Running Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest --cov=app --cov-report=html
```

Azure OpenAI calls are mocked throughout the suite (`pytest-mock`), so tests run without API costs. Coverage target: ≥80% (currently ~92%).

## 8. Azure Deployment
CI builds and validates the Docker image on every PR (`.github/workflows/ci.yml`). On merge to `main`, `.github/workflows/deploy.yml` builds and pushes both images to Azure Container Registry and deploys them to Azure Container Apps. Configure the following repository secrets: `ACR_LOGIN_SERVER`, `ACR_USERNAME`, `ACR_PASSWORD`, `AZURE_CREDENTIALS`, `AZURE_RESOURCE_GROUP`.

## 9. Sample Questions
1. "What was total revenue and net income for the most recent fiscal year?"
2. "What are the company's main risk factors related to supply chain?"
3. "How did diluted earnings per share change year-over-year?"
4. "What forward-looking statements were made about next year's growth?"
5. "Summarize the liquidity and capital resources section."

Sample filings (publicly available on SEC EDGAR / investor relations pages) can be placed in `data/sample_docs/` for local testing.
