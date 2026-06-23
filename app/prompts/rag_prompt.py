"""
RAG query prompt template injected per-question alongside the system prompt.

Kept separate from system_prompt.py so the two can be versioned and tuned
independently (e.g. tightening citation formatting without touching the
persona rules).
"""

PROMPT_VERSION = "v1.0"

RAG_PROMPT_TEMPLATE = """You are answering a question about financial documents. Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer based solely on the context provided above
- Cite specific pages using [Source: {source}, Page X]
- If the context is insufficient, state that clearly
- For financial figures, be exact - do not round unless the source rounds
- Structure your answer: Direct Answer -> Supporting Evidence -> Citations

ANSWER:"""
