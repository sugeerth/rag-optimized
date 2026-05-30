import os
from functools import lru_cache

import anthropic


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-sonnet-4-6-20250520"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

# Retrieval settings
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Cache settings
USER_CACHE_MAX = 1000
USER_CACHE_TTL = 3600
CHUNK_CACHE_MAX = 5000
CHUNK_CACHE_TTL = 1800

# Query transformation
ENABLE_QUERY_REWRITE = True
ENABLE_HYDE = False
NUM_QUERY_VARIANTS = 3

# Agentic RAG
ENABLE_AGENTIC = True
MAX_AGENT_ITERATIONS = 3


@lru_cache(maxsize=1)
def get_llm_client() -> anthropic.Anthropic:
    """Singleton Anthropic client shared across all modules."""
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def format_chunks(chunks: list[dict]) -> str:
    """Format chunks into a context string for LLM prompts."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        header = f"[Source {i} | {meta.get('doc_name', 'unknown')} | {meta.get('heading', '')}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def parse_json_from_text(text: str) -> dict | None:
    """Safely extract and parse JSON from LLM text output."""
    import json
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
