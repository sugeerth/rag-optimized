"""FastAPI server for the RAG pipeline with full observability."""

import logging
import re

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import cache
import tracing

# NOTE: `pipeline` and `evaluation` pull in heavy ML deps (torch / chromadb /
# sentence-transformers) transitively. They are imported lazily inside the
# route handlers that actually need them so the app can boot and serve demo
# mode with only the slim dependencies in requirements.txt. To go back to
# eager imports, move these back up here and install requirements-full.txt.

logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized RAG Pipeline", version="2.0.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


class QueryRequest(BaseModel):
    query: str
    evaluate: bool = False


class IngestRequest(BaseModel):
    doc_name: str
    content: str


class EvalRequest(BaseModel):
    test_cases: list[dict]


class LoraRequest(BaseModel):
    text: str
    steps: int = 300
    seed_prompt: str | None = None


# --- Core Endpoints ---

@app.post("/query")
async def handle_query(req: QueryRequest):
    """Run the full RAG pipeline on a user query."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    import pipeline  # lazy: heavy ML deps, see top-of-file note
    return await pipeline.query(req.query, enable_eval=req.evaluate)


def _ingest(doc_name: str, text: str) -> dict:
    """Ingest text, turning the 'heavy deps missing' case into a clear message.

    The live slim demo has no embedding model, so indexing can't run there —
    we return a friendly 503 instead of a generic 500. Full mode just works.
    """
    import pipeline  # lazy: heavy ML deps, see top-of-file note
    try:
        return pipeline.ingest_document(doc_name, text)
    except RuntimeError as exc:
        raise HTTPException(503, f"Indexing needs full mode (pip install -r requirements-full.txt). Detail: {exc}")


@app.post("/ingest")
async def handle_ingest(req: IngestRequest):
    """Ingest a document (JSON body with doc_name and content)."""
    if not req.content.strip():
        raise HTTPException(400, "Content cannot be empty")
    return _ingest(req.doc_name, req.content)


def _extract_text(filename: str, raw: bytes) -> str:
    """Pull plain text out of an uploaded file. Supports .pdf, .txt, .md.

    PDFs are binary, so we extract their text with pypdf instead of decoding
    them as utf-8 (which is what used to crash uploads). Add new file types here.
    """
    is_pdf = filename.lower().endswith(".pdf") or raw[:5] == b"%PDF-"
    if is_pdf:
        try:
            from pypdf import PdfReader  # light dep, in requirements.txt
        except ImportError:
            raise HTTPException(500, "PDF support needs pypdf — run: pip install pypdf")
        import io
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    # Plain text / markdown — tolerate stray bytes instead of 500-ing.
    return raw.decode("utf-8", errors="replace")


@app.post("/upload")
async def handle_upload(file: UploadFile = File(...)):
    """Upload and ingest a .pdf, .txt, or .md file."""
    # Sanitize filename to prevent path traversal
    safe_name = re.sub(r"[^\w\-.]", "_", file.filename or "upload.txt")
    content = await file.read()
    text = _extract_text(safe_name, content)
    if not text.strip():
        raise HTTPException(400, "No readable text found (scanned/image-only PDFs aren't supported).")
    return _ingest(safe_name, text)


# --- LoRA Lab: live fine-tuning on uploaded text (pure NumPy, fits free tier) ---

@app.post("/lora/finetune")
async def handle_lora(req: LoraRequest):
    """Run a real (tiny) LoRA fine-tune on the supplied text and return loss +
    before/after samples. Pure NumPy so it works on free-tier Render."""
    import asyncio
    import lora_live  # light: only needs numpy
    steps = max(20, min(req.steps, 1000))  # clamp so a request can't hog the CPU
    try:
        # Training is CPU-bound; run it off the event loop so the server stays responsive.
        return await asyncio.to_thread(
            lora_live.finetune_on_text, req.text, steps=steps, seed_prompt=req.seed_prompt
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/lora/finetune_file")
async def handle_lora_file(file: UploadFile = File(...), steps: int = 300):
    """Upload a .pdf/.txt/.md and LoRA-fine-tune on its text in one shot."""
    import asyncio
    import lora_live
    safe_name = re.sub(r"[^\w\-.]", "_", file.filename or "upload.txt")
    content = await file.read()
    text = _extract_text(safe_name, content)
    steps = max(20, min(steps, 1000))
    try:
        result = await asyncio.to_thread(lora_live.finetune_on_text, text, steps=steps)
        result["doc_name"] = safe_name
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# --- Evaluation Endpoints ---

@app.post("/evaluate")
async def handle_eval(req: EvalRequest):
    """Run batch evaluation over test cases."""
    import evaluation  # lazy: pulls in pipeline/heavy ML deps, see top-of-file note
    import pipeline
    return evaluation.run_batch_eval(req.test_cases, pipeline.query)


# --- Observability Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "cache": cache.cache_stats()}


@app.get("/traces")
async def get_traces(limit: int = 20):
    """Get recent pipeline traces."""
    return tracing.get_recent_traces(limit)


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Get a specific trace by ID."""
    result = tracing.get_trace(trace_id)
    if not result:
        raise HTTPException(404, "Trace not found")
    return result


@app.get("/metrics")
async def get_metrics():
    """Get aggregate latency metrics."""
    return tracing.get_latency_summary()


# --- Demo Mode (works without API key) ---

import config
import guardrails as _guardrails
import uuid
import time  # noqa: E402
import random  # noqa: E402


@app.get("/config/status")
async def config_status():
    """Check if API key is configured."""
    return {"api_key_set": bool(config.ANTHROPIC_API_KEY)}


@app.post("/demo/query")
async def demo_query(req: QueryRequest):
    """Demo query that simulates the full pipeline without needing an API key."""
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query cannot be empty")

    trace_id = str(uuid.uuid4())[:8]
    start = time.time()

    # Real input guardrails
    input_check = _guardrails.validate_input(query)
    if not input_check["safe"]:
        return {
            "answer": "I can't process this query due to safety concerns.",
            "blocked": True,
            "issues": input_check["issues"],
            "trace": {
                "trace_id": trace_id, "query": query,
                "total_ms": round((time.time() - start) * 1000, 2),
                "spans": [{"name": "input_guardrails", "duration_ms": 1.2, "metadata": input_check}],
            },
        }

    # Simulate realistic pipeline timing
    def _span(name, ms_base, **meta):
        jitter = random.uniform(0.8, 1.3)
        return {"name": name, "duration_ms": round(ms_base * jitter, 2), "metadata": meta}

    # Check if we have real chunks in the vector store
    try:
        from vector_store import get_collection, embed_texts
        collection = get_collection()
        count = collection.count()
    except Exception:
        count = 0

    sources = []
    if count > 0:
        try:
            emb = embed_texts([query])
            results = collection.query(query_embeddings=emb, n_results=min(5, count),
                                       include=["documents", "metadatas", "distances"])
            for i in range(len(results["ids"][0])):
                sources.append({
                    "text": results["documents"][0][i][:200] + ("..." if len(results["documents"][0][i]) > 200 else ""),
                    "metadata": results["metadatas"][0][i],
                    "rerank_score": round(random.uniform(0.6, 0.95), 3),
                })
        except Exception:
            pass

    if not sources:
        sources = [
            {"text": "RAG enhances LLMs by retrieving relevant context from a knowledge base before generating responses...",
             "metadata": {"doc_name": "demo-doc", "heading": "What is RAG?", "doc_type": "general"}, "rerank_score": 0.92},
            {"text": "The pipeline: chunk documents, embed them, store in vector DB, retrieve similar chunks, rerank, generate answer...",
             "metadata": {"doc_name": "demo-doc", "heading": "How it Works", "doc_type": "tutorial"}, "rerank_score": 0.87},
            {"text": "Structure-aware chunking splits by headings and paragraphs. Fixed-size fallback with overlap handles long sections...",
             "metadata": {"doc_name": "demo-doc", "heading": "Chunking", "doc_type": "general"}, "rerank_score": 0.74},
        ]

    source_text = "\n".join(f"[Source {i+1}]: {s['text']}" for i, s in enumerate(sources))
    answer = (
        f"Based on the retrieved documents:\n\n"
        f"{sources[0]['text'][:150]} [Source 1]\n\n"
        f"{sources[1]['text'][:150] if len(sources) > 1 else ''} [Source 2]\n\n"
        f"The system uses structure-aware chunking, cross-encoder reranking, and "
        f"agentic retrieval with self-reflection to ensure high-quality, grounded answers. [Source 3]"
    )

    output_check = _guardrails.validate_output(answer, [{"text": s["text"][:200], "metadata": s.get("metadata", {})} for s in sources])

    spans = [
        _span("input_guardrails", 1.5, safe=True),
        _span("user_cache_check", 0.3, hit=False),
        _span("query_rewrite", 450, original=query, rewritten=f"detailed explanation of: {query}"),
        _span("query_expansion", 380, num_variants=3),
        _span("embedding", 45, num_embeddings=4),
        _span("chunk_cache_check", 0.2, hit=False),
        _span("parallel_vector_search", 28, num_hits=20),
        _span("reranking", 120, before=20, after=len(sources)),
        _span("agentic_retrieval", 1800, iterations=2, grounded=True),
        _span("output_guardrails", 1.0, **output_check),
    ]
    total_ms = sum(s["duration_ms"] for s in spans)

    trace_data = {"trace_id": trace_id, "query": query, "total_ms": round(total_ms, 2), "spans": spans}
    tracing._save_trace(trace_data)

    result = {
        "answer": answer,
        "sources": sources,
        "cached": False,
        "grounded": True,
        "output_safety": output_check,
        "trace": trace_data,
        "reasoning_trace": [
            "Iteration 1: Evaluating retrieved context — found 20 chunks covering the query topic.",
            "Searching: ['RAG pipeline architecture', 'retrieval augmented generation techniques']",
            "Iteration 2: Context is sufficient to provide a comprehensive answer.",
            "Final answer generated (grounded=True)",
        ],
    }

    if req.evaluate:
        result["evaluation"] = {
            "faithfulness": round(random.uniform(0.82, 0.98), 2),
            "relevance": round(random.uniform(0.85, 0.99), 2),
            "completeness": round(random.uniform(0.70, 0.95), 2),
            "detail": "Answer is well-supported by sources with proper citations.",
        }

    return result


# --- UI ---

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def ui():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
