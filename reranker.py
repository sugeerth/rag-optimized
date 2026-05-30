"""Cross-encoder reranker.

Takes top-K vector search results and re-scores them with a cross-encoder
to filter noise and surface the most relevant chunks.
"""

from functools import lru_cache

# Heavy ML dep is optional so this module stays importable in slim demo-mode
# deploys. get_reranker() raises a clear RuntimeError if it's missing. Run
# `pip install -r requirements-full.txt` for full RAG mode.
try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - exercised only in slim deploys
    CrossEncoder = None

import config


@lru_cache(maxsize=1)
def get_reranker():
    if CrossEncoder is None:
        raise RuntimeError(
            "sentence-transformers not installed — run "
            "pip install -r requirements-full.txt for full RAG mode"
        )
    return CrossEncoder(config.RERANKER_MODEL)


def rerank(query: str, hits: list[dict], top_k: int = config.TOP_K_RERANK) -> list[dict]:
    """Rerank retrieved chunks using a cross-encoder.

    Args:
        query: The user's query text
        hits: List of dicts with 'text', 'metadata', 'distance' keys
        top_k: Number of top results to return after reranking

    Returns:
        Top-k hits sorted by cross-encoder relevance score (descending)
    """
    if not hits:
        return []

    model = get_reranker()
    pairs = [[query, hit["text"]] for hit in hits]
    scores = model.predict(pairs)

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)

    reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
