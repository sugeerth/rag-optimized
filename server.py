"""FastAPI server for the RAG pipeline with full observability."""

import logging
import re

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import cache
import demo_store  # seeds the engine with the sample docs on import
import livestats
import tracing
from engine import ENGINE

# NOTE: `pipeline` and `evaluation` pull in heavy ML deps (torch / chromadb /
# sentence-transformers) transitively. They are imported lazily inside the
# route handlers that actually need them so the app can boot and serve demo
# mode with only the slim dependencies in requirements.txt. To go back to
# eager imports, move these back up here and install requirements-full.txt.

logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized RAG Pipeline", version="3.0.0")


class StatsMiddleware:
    """Feed every request into the live stats behind /stats and Mission Control.

    Plain ASGI rather than @app.middleware("http"): the latter wraps each request in
    extra tasks and streams, which costs real throughput on a hot path.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        status = 500

        async def _send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            livestats.record_request(error=status >= 500)


app.add_middleware(StatsMiddleware)


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

def _heavy_stack_available() -> bool:
    """True only when the full embedding stack is installed (full mode).

    On the slim free tier these aren't present, so we route to the demo pipeline
    instead of letting the real pipeline blow up partway through.
    """
    try:
        import vector_store
        vector_store.get_collection()  # raises RuntimeError in slim mode
        return True
    except Exception:
        return False


@app.post("/query")
async def handle_query(req: QueryRequest):
    """Run the full RAG pipeline on a user query.

    On the slim free tier the heavy embedding stack is missing, so the real
    pipeline can't run. We fall back to the demo pipeline (same response shape)
    so /query stays functional instead of 500-ing.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if not _heavy_stack_available():
        return await demo_query(req)
    try:
        import pipeline  # lazy: heavy ML deps, see top-of-file note
        return await pipeline.query(req.query, enable_eval=req.evaluate)
    except (RuntimeError, ImportError):
        return await demo_query(req)


def _ingest(doc_name: str, text: str) -> dict:
    """Ingest text, falling back to the in-memory demo store in slim mode.

    Full mode embeds + indexes via the real pipeline. On the slim free tier the
    embedding stack is missing (RuntimeError) or unimportable (ImportError), so we
    fall back to a pure-Python in-memory store — uploading then querying still works.
    """
    # The NumPy engine always indexes the doc, so demo-mode queries can cite it.
    n = demo_store.add_document(doc_name, text)
    try:
        import pipeline  # lazy: heavy ML deps, see top-of-file note
        return pipeline.ingest_document(doc_name, text)
    except (RuntimeError, ImportError):
        return {"status": "indexed (hybrid engine)", "doc_name": doc_name, "chunks": n,
                "index": ENGINE.stats()}


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
    return {"status": "ok", "cache": cache.cache_stats(), "engine": ENGINE.stats()}


@app.get("/stats")
async def stats(window: int = 60):
    """Live rolling stats: throughput, latency percentiles, cache hit rate, memory, index size."""
    window = max(1, min(window, livestats.WINDOW_SECONDS))
    return {**livestats.snapshot(window=window), "engine": ENGINE.stats()}


class CorpusRequest(BaseModel):
    docs: int = 500


@app.post("/demo/load_corpus")
async def load_corpus(req: CorpusRequest):
    """Index N synthetic service runbooks (with known answers) so there is a real corpus to
    search. Returns a few sample questions, including one with a typo."""
    import asyncio
    import corpus
    n = max(1, min(req.docs, 5000))
    docs, questions = corpus.build(n)

    # Index on the event loop (the engine is single-threaded by design), yielding
    # between batches so queries keep flowing while a big corpus loads.
    for i, d in enumerate(docs):
        ENGINE.add_document(d["doc_name"], d["content"])
        if i % 20 == 19:
            await asyncio.sleep(0)
    samples = [next(q for q in questions if q["kind"] == k) for k in ("plain", "paraphrase", "typo")]
    return {"status": "indexed", "documents": n, "index": ENGINE.stats(), "sample_questions": samples,
            "questions": [{"query": q["query"], "answer": q["answer"]} for q in questions[:400]]}


@app.post("/demo/reset")
async def reset_index():
    """Drop everything and reload just the seed documents."""
    ENGINE.reset()
    demo_store.seed()
    return {"status": "reset", "index": ENGINE.stats()}


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

import config  # noqa: E402


@app.get("/config/status")
async def config_status():
    """Check if API key is configured."""
    return {"api_key_set": bool(config.ANTHROPIC_API_KEY)}


@app.post("/demo/query")
async def demo_query(req: QueryRequest):
    """Answer with the pure-NumPy hybrid engine: no API key, no heavy deps, nothing simulated.

    Every span in the returned trace is a real measured stage (see engine.py).
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    result = ENGINE.query(req.query, evaluate=req.evaluate)
    livestats.record_query(result["trace"]["total_ms"], cached=result.get("cached", False),
                           blocked=result.get("blocked", False), grounded=result.get("grounded", True))
    return result


# --- UI ---

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def ui():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
