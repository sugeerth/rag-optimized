# rag-optimized

An observable, optimized RAG pipeline with a web UI and a no-API-key demo mode.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sugeerth/rag-optimized)

Click the button above to deploy a live demo to Render in one click. No API key required — it boots straight into demo mode.

---

## Run locally in 30 seconds

```
pip install -r requirements.txt
uvicorn server:app --reload
# open http://localhost:8000
```

That's it. The slim dependencies are enough to run the web UI and the full demo experience.

---

## Demo mode vs full mode

**Demo mode (default, zero config).** With just the slim `requirements.txt` and no API key, the app simulates the full RAG pipeline end to end — query transformation, retrieval, reranking, and answer synthesis — with **real guardrails and real span tracing** so you can watch every stage. Perfect for exploring, demoing, or deploying to the cloud for free.

**Full mode (real RAG).** To run real embeddings, vector search, cross-encoder reranking, and LLM-generated answers:

```
pip install -r requirements-full.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn server:app --reload
```

`requirements-full.txt` adds the heavy pieces (ChromaDB + sentence-transformers) on top of the slim set. Set `ANTHROPIC_API_KEY` to get real LLM answers; without it, the app stays in demo mode.

---

## Project map

Everything is plain Python and easy to tweak. Here's where each piece lives:

| File | What it does |
| --- | --- |
| `server.py` | FastAPI routes + the demo endpoint; serves the web UI |
| `pipeline.py` | RAG orchestration — wires all the stages together |
| `vector_store.py` | ChromaDB + embeddings (vector storage and search) |
| `reranker.py` | Cross-encoder reranking of retrieved chunks |
| `query_transform.py` | Query rewrite / expansion |
| `chunking.py` | Structure-aware document chunking |
| `agentic_rag.py` | Agentic retrieval loop (multi-step reasoning) |
| `guardrails.py` | Input/output safety + PII handling |
| `cache.py` | User + chunk caches |
| `tracing.py` | Span tracing across pipeline stages |
| `evaluation.py` | Faithfulness / relevance scoring |
| `config.py` | All tunable settings in one place |
| `static/index.html` | The web UI |

---

## Tweak it

Open **`config.py`** — that's where the knobs live:

- `TOP_K_RETRIEVAL`, `TOP_K_RERANK` — how many chunks to retrieve and keep
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunking granularity
- `ENABLE_QUERY_REWRITE`, `ENABLE_HYDE`, `NUM_QUERY_VARIANTS` — query transformation
- `ENABLE_AGENTIC`, `MAX_AGENT_ITERATIONS` — agentic RAG behavior
- `EMBEDDING_MODEL`, `RERANKER_MODEL`, `LLM_MODEL` — model names

Want a new route? **Add an endpoint by editing `server.py` — it's plain FastAPI.** Define a function, decorate it with `@app.get(...)` or `@app.post(...)`, and you're done.

---

## How to set your API key on Render

Demo mode needs no key. To enable real LLM answers on a deployed instance:

1. Go to the Render **Dashboard**.
2. Open **your service**.
3. Go to **Environment**.
4. Add a variable: `ANTHROPIC_API_KEY` = your key.

Render will redeploy and the app will start serving real answers.
