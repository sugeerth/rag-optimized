"""Pure-Python, in-memory retrieval store for the slim (free-tier) deployment.

When the heavy embedding stack (torch / chromadb / sentence-transformers) isn't
installed, we still want uploading a document and then querying it to work. This
module provides a tiny keyword/token-overlap retriever that depends only on the
stdlib plus the dependency-free `chunking.py`.

Design notes (this codebase values simplicity):
  - Chunks live in a plain module-level list `_CHUNKS`.
  - Retrieval is classic bag-of-words token overlap with a light TF weighting and
    a small bonus when a query token also appears in the chunk's heading.
  - Everything is easy to tweak: see the small constants near `search()`.
"""

import re

from chunking import chunk_by_structure

# --- In-memory store -------------------------------------------------------

# Each entry: {"text": str, "metadata": {"doc_name", "heading", "doc_type", ...}}
_CHUNKS: list[dict] = []

# Tokenizer: lowercase words on \w+ boundaries. Reused for query + chunk text.
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on word boundaries."""
    return _TOKEN_RE.findall(text.lower())


# --- Mutation API ----------------------------------------------------------

def add_document(doc_name: str, text: str) -> int:
    """Chunk `text` and add it to the store, returning the number of chunks added.

    Any chunks previously added under the same `doc_name` are removed first, so
    re-uploading a document updates it in place rather than duplicating.
    """
    # Drop prior chunks from this doc so re-upload replaces instead of duplicates.
    global _CHUNKS
    _CHUNKS = [c for c in _CHUNKS if c["metadata"].get("doc_name") != doc_name]

    chunks = chunk_by_structure(text, doc_name, max_chunk_size=512, overlap=50)
    for ch in chunks:
        _CHUNKS.append({"text": ch.text, "metadata": ch.metadata})
    return len(chunks)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Keyword/token-overlap retrieval over the in-memory store.

    Scoring: for each query token, add (count_in_chunk / (1 + len(chunk_tokens)))
    — a simple TF weighting that rewards matches in short, focused chunks — plus a
    small bonus if the token also appears in the chunk's heading.

    Returns up to `top_k` hits as dicts:
        {"text": <truncated to ~200 chars>, "metadata": ..., "rerank_score": 0..1}

    If nothing overlaps (or the store is empty of matches), falls back to the
    first `top_k` chunks so a loaded store always returns something.
    """
    if not _CHUNKS:
        return []

    q_tokens = _tokenize(query)
    HEADING_BONUS = 0.5  # extra weight when a query token appears in the heading

    scored = []  # (raw_score, chunk)
    for chunk in _CHUNKS:
        chunk_tokens = _tokenize(chunk["text"])
        if not chunk_tokens:
            continue
        heading_tokens = set(_tokenize(chunk["metadata"].get("heading", "")))

        score = 0.0
        for qt in q_tokens:
            count = chunk_tokens.count(qt)
            if count:
                score += count / (1 + len(chunk_tokens))
            if qt in heading_tokens:
                score += HEADING_BONUS / (1 + len(chunk_tokens))
        scored.append((score, chunk))

    # If no query token overlapped anything, fall back to first top_k chunks.
    if not any(s > 0 for s, _ in scored):
        fallback = _CHUNKS[:top_k]
        return [_to_hit(c, 0.0, max_score=1.0) for c in fallback]

    scored.sort(key=lambda sc: sc[0], reverse=True)
    top = scored[:top_k]
    max_score = top[0][0] or 1.0  # normalize against the best hit

    return [_to_hit(chunk, raw, max_score) for raw, chunk in top]


def _to_hit(chunk: dict, raw_score: float, max_score: float) -> dict:
    """Shape a stored chunk into the response dict the frontend expects."""
    text = chunk["text"]
    truncated = text[:200] + ("..." if len(text) > 200 else "")
    norm = round(raw_score / max_score, 3) if max_score else 0.0
    return {"text": truncated, "metadata": chunk["metadata"], "rerank_score": norm}


def count() -> int:
    """Number of chunks currently in the store."""
    return len(_CHUNKS)


def reset() -> None:
    """Clear the store (mainly for tests)."""
    _CHUNKS.clear()


# --- Seed documents --------------------------------------------------------
# Loaded on import so queries return real-looking citations even before any
# upload. Tweak freely — these are plain markdown with ## headings.

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


def _seed() -> None:
    """Populate the store with the sample documents on import."""
    for name, text in SEED_DOCS.items():
        add_document(name, text)


_seed()
