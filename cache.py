"""Two-level caching: user query cache and chunk retrieval cache.

User cache: maps query text -> final LLM response (avoids full pipeline re-run).
Chunk cache: maps query embedding hash -> retrieved chunks (avoids vector DB hits).
"""

import hashlib
import json
from cachetools import TTLCache

import config


_user_cache = TTLCache(maxsize=config.USER_CACHE_MAX, ttl=config.USER_CACHE_TTL)
_chunk_cache = TTLCache(maxsize=config.CHUNK_CACHE_MAX, ttl=config.CHUNK_CACHE_TTL)

# Track indexed doc versions for cache invalidation
_doc_versions: dict[str, str] = {}


def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _embedding_hash(embedding: list[float]) -> str:
    raw = json.dumps(embedding[:10])  # First 10 dims is enough for a cache key
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# --- User Cache (query -> final response) ---

def get_user_cache(query: str) -> str | None:
    return _user_cache.get(_hash_key(query))


def set_user_cache(query: str, response: str):
    _user_cache[_hash_key(query)] = response


# --- Chunk Cache (embedding -> retrieved chunks) ---

def get_chunk_cache(embedding: list[float]) -> list[dict] | None:
    return _chunk_cache.get(_embedding_hash(embedding))


def set_chunk_cache(embedding: list[float], chunks: list[dict]):
    _chunk_cache[_embedding_hash(embedding)] = chunks


# --- Cache Invalidation ---

def invalidate_for_doc(doc_name: str):
    """Clear all caches when a document is re-indexed.

    Simple strategy: clear everything. For production, use per-doc tracking.
    """
    _user_cache.clear()
    _chunk_cache.clear()


def update_doc_version(doc_name: str, content_hash: str) -> bool:
    """Returns True if the document is new or changed."""
    old = _doc_versions.get(doc_name)
    _doc_versions[doc_name] = content_hash
    return old != content_hash


def cache_stats() -> dict:
    return {
        "user_cache_size": len(_user_cache),
        "user_cache_max": config.USER_CACHE_MAX,
        "chunk_cache_size": len(_chunk_cache),
        "chunk_cache_max": config.CHUNK_CACHE_MAX,
    }
