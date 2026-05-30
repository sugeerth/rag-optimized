"""Query transformation: rewriting, HyDE, and multi-query expansion.

Transforms raw user queries into better retrieval queries before embedding.
"""

import config


def rewrite_query(query: str) -> str:
    """Use LLM to rewrite the query for better retrieval."""
    if not config.ENABLE_QUERY_REWRITE:
        return query

    client = config.get_llm_client()
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Rewrite this search query to be more specific and effective for document retrieval.
Return ONLY the rewritten query, nothing else.

Original query: {query}"""
        }],
    )
    return response.content[0].text.strip()


def generate_hyde_document(query: str) -> str:
    """Generate a hypothetical document that would answer the query (HyDE).

    The embedding of this hypothetical answer often matches relevant docs
    better than the query embedding itself.
    """
    if not config.ENABLE_HYDE:
        return query

    client = config.get_llm_client()
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Write a short paragraph that directly answers this question.
Write as if you are a technical document. Be specific and factual.

Question: {query}"""
        }],
    )
    return response.content[0].text.strip()


def expand_query(query: str, num_variants: int = config.NUM_QUERY_VARIANTS) -> list[str]:
    """Generate multiple query variants for parallel search."""
    client = config.get_llm_client()
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Generate {num_variants} different search queries that would help find information to answer the original question.
Each query should approach the topic from a different angle.
Return one query per line, nothing else.

Original question: {query}"""
        }],
    )

    variants = []
    for line in response.content[0].text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1. " or "1) "
        import re
        cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line)
        if cleaned:
            variants.append(cleaned)
    return [query] + variants[:num_variants]
