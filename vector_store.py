"""Vector store layer using ChromaDB with sentence-transformers embeddings.

Supports parallel searches across multiple query variants.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Heavy ML deps are optional: the app boots in demo mode with only the slim
# requirements.txt. We import them defensively so this module stays importable
# when they're absent; functions that actually use them raise a clear
# RuntimeError instead. Run `pip install -r requirements-full.txt` for full
# RAG mode.
try:
    import chromadb
except ImportError:  # pragma: no cover - exercised only in slim deploys
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

import config
from chunking import Chunk

_executor = ThreadPoolExecutor(max_workers=4)


def _require_heavy_deps():
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers not installed — run "
            "pip install -r requirements-full.txt for full RAG mode"
        )
    if chromadb is None:
        raise RuntimeError(
            "chromadb not installed — run "
            "pip install -r requirements-full.txt for full RAG mode"
        )


@lru_cache(maxsize=1)
def get_embedding_model():
    _require_heavy_deps()
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_chroma_client():
    _require_heavy_deps()
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def index_chunks(chunks: list[Chunk]) -> int:
    """Index a list of chunks into ChromaDB. Returns count indexed."""
    if not chunks:
        return 0

    collection = get_collection()
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    embeddings = embed_texts(texts)

    # Generate unique IDs based on doc name + index
    ids = [f"{c.metadata.get('doc_name', 'doc')}_{i}" for i, c in enumerate(chunks)]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def search(query_embedding: list[float], top_k: int = config.TOP_K_RETRIEVAL) -> list[dict]:
    """Search vector store for similar chunks."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


async def parallel_search(query_embeddings: list[list[float]], top_k: int = config.TOP_K_RETRIEVAL) -> list[dict]:
    """Run multiple vector searches in parallel and merge results (deduplicated)."""
    loop = asyncio.get_running_loop()

    futures = [
        loop.run_in_executor(_executor, search, emb, top_k)
        for emb in query_embeddings
    ]
    all_results = await asyncio.gather(*futures)

    # Merge and deduplicate by ID, keeping the best (lowest) distance
    seen = {}
    for result_set in all_results:
        for hit in result_set:
            hit_id = hit["id"]
            if hit_id not in seen or hit["distance"] < seen[hit_id]["distance"]:
                seen[hit_id] = hit

    merged = sorted(seen.values(), key=lambda x: x["distance"])
    return merged[:top_k]
