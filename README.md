# rag-optimized

An observable, optimized RAG pipeline with a web UI. With no API key, no GPU and no vector database it still does **real** retrieval: a pure-NumPy hybrid engine (BM25 + dense vectors, fused, reranked, cached, traced) that answers in about 2 ms and has been soak-tested for an hour straight.

| | |
| --- | --- |
| Retrieval, 2,000 docs / 16,000 chunks | recall@5 **99.5%**, MRR 0.97 (`python bench.py`) |
| Uncached pipeline latency, one core | p50 **2.1 ms**, p99 4.0 ms |
| 1-hour soak | SOAK_HEADLINE |

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

**Demo mode (default, zero config).** With just the slim `requirements.txt` and no API key, the app runs the whole pipeline for real on a pure-NumPy engine (`engine.py`). Nothing is simulated and every trace span is a measured stage. The one thing it does not do is generate text: answers are the best-supported sentences from your documents, each with its citation, and it declines when the evidence is weak.

**Full mode (real RAG).** To run real embeddings, vector search, cross-encoder reranking, and LLM-generated answers:

```
pip install -r requirements-full.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn server:app --reload
```

`requirements-full.txt` adds the heavy pieces (ChromaDB + sentence-transformers) on top of the slim set. Set `ANTHROPIC_API_KEY` to get real LLM answers; without it, the app stays in demo mode.

---

## The engine (`engine.py`)

Every query goes through these stages, each a timed span you can see in the UI:

| Stage | What actually runs |
| --- | --- |
| Input guardrails | Prompt-injection and PII patterns; unsafe queries stop here |
| Cache | Two tiers: exact string, then a canonical key (stemmed, stopwords dropped, order ignored) so "What is RAG?" and "rag what is" share an entry. Any ingest invalidates it; re-ingesting identical content is a no-op |
| Query rewrite | Drops filler words and fixes typos against the index vocabulary (edit distance 1, or 2 for long words) |
| Multi-query expansion | Pseudo-relevance feedback: the most distinctive terms from the top BM25 hits become an extra query variant |
| Hybrid search | BM25 inverted index (heading terms boosted) plus cosine over hashed n-gram embeddings, scored for all variants in one BLAS call and fused with reciprocal rank fusion |
| Rerank | Scores the fused pool on term coverage, BM25, dense similarity, phrase and heading matches, then MMR for diversity |
| Agentic loop | Checks how much of the question the context covers; if key terms are missing it searches again for them, and it refuses to answer below a grounding threshold |
| Answer + evaluation | Extractive, cited sentences. Faithfulness, relevance, completeness and context precision are computed from the text, not sampled |

The dense vectors are feature-hashed word and character n-grams, not a neural model. They make retrieval robust to typos and word forms, which is what BM25 is bad at. They do not understand synonyms: a question whose wording never appears in the documents ("who gets paged" for a doc that says "owned by") finds the right document but often the wrong sentence. Full mode's sentence-transformer embeddings close that gap.

### Measure it

```
python bench.py        # 2,000 docs, 2,000 questions with known answers
```

`corpus.py` generates service runbooks with known facts, so quality is measured rather than eyeballed. A third of the questions carry a typo or a paraphrase.

| Question type | n | recall@1 | recall@5 | MRR | Answer contains the fact |
| --- | --- | --- | --- | --- | --- |
| Worded like the docs | 1,019 | 96.7% | 100% | 0.983 | 90.6% |
| Typo in the service name | 514 | 94.2% | 98.1% | 0.958 | 88.9% |
| Paraphrased | 467 | 93.1% | 100% | 0.966 | 68.5% |
| All | 2,000 | 95.2% | 99.5% | 0.973 | 85.0% |

### Soak it

```
python soak.py --duration 3600     # one hour; spawns its own server
python soak.py --duration 60       # quick version
```

Eight closed-loop clients ask checked questions while another thread re-ingests documents (cache invalidation, tombstones, index compaction) and a third kicks off LoRA fine-tunes. Prompt-injection and PII queries must be blocked and off-topic questions must be declined; the run exits non-zero on any error. It writes `report.html` (charts), `report.md`, `summary.json` and `timeseries.jsonl`.

SOAK_SECTION

### Watch it

The **Mission Control** tab polls `GET /stats` every second: queries per second, p50/p99, cache hit rate, memory, index size. It can load a 500-document corpus and fire 2,000 queries from your browser so you can watch the numbers move.

---

## 🧬 LoRA Lab — live fine-tuning on your upload

Open the **LoRA Lab** tab, upload a `.pdf`/`.txt`/`.md` (or paste text), and hit **Fine-tune with LoRA**. The server runs a **real but tiny LoRA**: a frozen base model plus trainable low-rank adapters (`A @ B`), trained with hand-written gradients in **pure NumPy** — so it fits Render's free tier (512MB / 0.1 CPU) and finishes in well under a second. You get a live loss curve plus **before/after** samples.

Because the base model is intentionally small (char-level), the generated text is rough — the point is to *see LoRA actually train* (loss drops, "after" text picks up your document's style) on a free hosted box. It's all in **`lora_live.py`**, fully commented and easy to tweak (`D`, `R`, `STEPS`, `LR` at the top). Endpoints: `POST /lora/finetune` (JSON text) and `POST /lora/finetune_file` (file upload).

> Want real pretrained-model LoRA (PyTorch + PEFT) instead? See `lora_demo.py` + `requirements-lora.txt` — it fine-tunes Qwen2.5-0.5B locally (Apple Silicon/MPS, CUDA, or CPU). That needs ~2GB RAM, so it runs on your machine, not the free cloud tier.

---

## Project map

Everything is plain Python and easy to tweak. Here's where each piece lives:

| File | What it does |
| --- | --- |
| `server.py` | FastAPI routes; serves the web UI |
| `engine.py` | The pure-NumPy hybrid engine: BM25, dense vectors, fusion, rerank, cache, answers |
| `demo_store.py` | Seed documents + a thin wrapper over the engine |
| `livestats.py` | Rolling per-second stats behind `/stats` and Mission Control |
| `corpus.py` | Synthetic corpus with ground-truth answers |
| `bench.py` | Retrieval quality + latency benchmark |
| `soak.py`, `soak_report.py` | Soak test and its HTML/Markdown report |
| `tests/` | `pytest -q` (install `requirements-dev.txt`) |
| `pipeline.py` | RAG orchestration — wires all the stages together |
| `vector_store.py` | ChromaDB + embeddings (vector storage and search) |
| `reranker.py` | Cross-encoder reranking of retrieved chunks |
| `query_transform.py` | Query rewrite / expansion |
| `chunking.py` | Structure-aware document chunking |
| `agentic_rag.py` | Agentic retrieval loop (multi-step reasoning) |
| `guardrails.py` | Input/output safety + PII handling |
| `cache.py` | User + chunk caches |
| `tracing.py` | Span tracing across pipeline stages (bounded in memory; `TRACE_PERSIST=1` also writes JSON files) |
| `evaluation.py` | Faithfulness / relevance scoring |
| `config.py` | All tunable settings in one place |
| `static/index.html` | The web UI |

---

## Tweak it

The engine's knobs are constants at the top of **`engine.py`** (`DIM`, `TOP_K`, `MMR_LAMBDA`, `GROUNDED_MIN_SCORE`, ...). The full-mode pipeline's live in **`config.py`**:

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
