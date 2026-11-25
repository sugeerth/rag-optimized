"""Thin compatibility shim over `engine.py` + the seed documents.

The in-memory store used to be a keyword-overlap toy; it is now the real hybrid
engine (BM25 + dense + rerank) in `engine.py`. This module keeps the old
`add_document / search / count / reset` API and owns the seed corpus that makes
the app answer questions before anything is uploaded.
"""

from engine import ENGINE


def add_document(doc_name: str, text: str) -> int:
    """Chunk + index `text`; re-adding a doc_name replaces it. Returns chunk count."""
    return ENGINE.add_document(doc_name, text)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid retrieval + rerank. Returns [{"text", "metadata", "rerank_score", "scores"}]."""
    cands, detail = ENGINE.hybrid_search([query])
    hits = ENGINE.rerank(query, cands, detail, top_k=top_k)
    return [{"text": h["text"][:200] + ("..." if len(h["text"]) > 200 else ""), "metadata": h["metadata"],
             "rerank_score": h["rerank_score"], "scores": h["scores"]} for h in hits]


def count() -> int:
    return ENGINE.n_alive


def reset() -> None:
    """Clear the store (mainly for tests)."""
    ENGINE.reset()


# --- Seed documents --------------------------------------------------------
# Loaded on import so queries return real citations even before any upload.
# Tweak freely — these are plain markdown with ## headings.

SEED_DOCS: dict[str, str] = {
    "what-is-rag.md": """## What is RAG
Retrieval-Augmented Generation (RAG) enhances a large language model by fetching
relevant context from an external knowledge base before generating an answer.
Instead of relying only on what the model memorized during training, RAG grounds
responses in your own documents, which reduces hallucinations and lets the system
answer questions about private or up-to-date information.

## How This Pipeline Works
The pipeline runs in stages. First, documents are split into chunks using
structure-aware chunking that respects headings and paragraphs. Each chunk is then
embedded into a vector. At query time the question is embedded too, and a vector
search retrieves the most similar chunks. A cross-encoder reranker reorders those
candidates by true relevance. Finally an agentic loop reflects on whether the
retrieved context is sufficient, optionally searches again, and only then writes a
grounded answer with citations.

## Reranking
Reranking is the step that takes the rough list of vector-search hits and reorders
them with a more expensive but more accurate cross-encoder model. Vector search is
fast but approximate; the reranker reads the query and each candidate chunk together
and scores how well they actually match, so the best evidence rises to the top
before the answer is written.
""",
    "architecture.md": """## Architecture and Features
This project is an optimized RAG pipeline with full observability built in. Every
request flows through guardrails, a multi-level cache, query transformation,
parallel vector search, reranking, and an agentic retrieval loop, with a trace
recorded for each span so you can see exactly where time was spent.

## Guardrails
Input guardrails screen incoming queries for prompt injection and PII before any
work is done, blocking unsafe requests up front. Output guardrails scan the
generated answer for leaked PII, missing citations, and refusal patterns, and can
redact sensitive data so nothing unsafe is returned to the user.

## Caching and Tracing
A two-tier cache stores both full user answers and intermediate chunk-retrieval
results keyed by embedding, so repeated or similar queries skip expensive steps.
Tracing wraps every pipeline stage in a timed span, persists each trace, and exposes
aggregate latency metrics through the observability endpoints.

## Demo Mode and LoRA Lab
Demo mode simulates the full pipeline without needing an API key or the heavy ML
stack, so the app stays fully interactive on a free host. The LoRA Lab runs a real
but tiny LoRA fine-tune on text you upload using pure NumPy, showing the loss curve
and before/after samples without any GPU.
""",
    "faq.md": """## Frequently Asked Questions

## Do I need an API key
No. Demo mode runs the entire pipeline experience — guardrails, retrieval, reranking,
tracing, and citations — without an API key. Adding an Anthropic API key and the full
dependencies unlocks live LLM-generated answers.

## Can I upload my own documents
Yes. Upload a .pdf, .txt, or .md file, or POST JSON to the ingest endpoint. The text
is chunked and indexed immediately, and your next query will retrieve and cite the
new content.

## Does it work on the free tier
Yes. On the slim free tier there is no torch, chromadb, or sentence-transformers, so
a lightweight in-memory keyword retriever handles ingestion and search so that upload
and query still work end to end.
""",
}


def seed() -> None:
    """(Re)load the sample documents."""
    for name, text in SEED_DOCS.items():
        add_document(name, text)


seed()
