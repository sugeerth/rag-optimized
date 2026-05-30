"""Main RAG pipeline orchestrator.

Flow: Query → Guardrails → Cache Check → Query Transform → Embed → Parallel Search →
      Rerank → Agentic Loop (optional) → LLM Call → Output Guardrails → Evaluate → Output

Full tracing on every step.
"""

import hashlib

import config
import cache
import vector_store
import reranker
import query_transform
import guardrails
import agentic_rag
import evaluation
from tracing import Trace
from chunking import chunk_by_structure


# --- Document Ingestion ---

def ingest_document(doc_name: str, content: str) -> dict:
    """Ingest a document: chunk, embed, and index."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    if not cache.update_doc_version(doc_name, content_hash):
        return {"status": "unchanged", "doc_name": doc_name, "chunks": 0}

    cache.invalidate_for_doc(doc_name)

    chunks = chunk_by_structure(
        content, doc_name,
        max_chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )
    count = vector_store.index_chunks(chunks)

    return {"status": "indexed", "doc_name": doc_name, "chunks": count}


# --- Query Pipeline ---

async def query(user_query: str, enable_eval: bool = False) -> dict:
    """Full RAG pipeline with tracing, guardrails, agentic retrieval, and evaluation."""
    trace = Trace(user_query)

    # 1. Input guardrails
    with trace.span("input_guardrails"):
        input_check = guardrails.validate_input(user_query)
    trace.finish_span(result=input_check)

    if not input_check["safe"]:
        trace.finish()
        return {
            "answer": "I can't process this query due to safety concerns.",
            "blocked": True,
            "issues": input_check["issues"],
            "trace": trace.finish(),
        }

    # 2. Check user cache
    with trace.span("user_cache_check"):
        cached = cache.get_user_cache(user_query)
    trace.finish_span(hit=cached is not None)

    if cached:
        return {
            "answer": cached,
            "sources": [],
            "cached": True,
            "trace": trace.finish(),
        }

    # 3. Query transformation
    with trace.span("query_rewrite"):
        rewritten = query_transform.rewrite_query(user_query)
    trace.finish_span(original=user_query, rewritten=rewritten)

    # 4. Multi-query expansion
    with trace.span("query_expansion"):
        query_variants = query_transform.expand_query(rewritten)
    trace.finish_span(num_variants=len(query_variants))

    # 5. Embed all query variants
    with trace.span("embedding"):
        all_embeddings = vector_store.embed_texts(query_variants)
    trace.finish_span(num_embeddings=len(all_embeddings))

    # 6. Check chunk cache
    primary_embedding = all_embeddings[0]
    with trace.span("chunk_cache_check"):
        cached_chunks = cache.get_chunk_cache(primary_embedding)
    trace.finish_span(hit=cached_chunks is not None)

    if cached_chunks:
        hits = cached_chunks
    else:
        # 7. Parallel vector search
        with trace.span("parallel_vector_search"):
            hits = await vector_store.parallel_search(all_embeddings, top_k=config.TOP_K_RETRIEVAL)
        trace.finish_span(num_hits=len(hits))
        cache.set_chunk_cache(primary_embedding, hits)

    # 8. Rerank
    with trace.span("reranking"):
        reranked = reranker.rerank(user_query, hits, top_k=config.TOP_K_RERANK)
    trace.finish_span(before=len(hits), after=len(reranked))

    # 9. Agentic retrieval OR direct LLM call
    if config.ENABLE_AGENTIC:
        with trace.span("agentic_retrieval"):
            agent_result = await agentic_rag.agentic_retrieve(
                query=user_query,
                initial_chunks=reranked,
                trace_callback=lambda name, meta: trace.span(f"agent_sub:{name}"),
            )
        trace.finish_span(
            iterations=agent_result["iterations"],
            grounded=agent_result["grounded"],
        )
        answer = agent_result["final_answer"]
        grounded = agent_result["grounded"]
        final_sources = agent_result["all_sources"]
        reasoning_trace = agent_result["reasoning_trace"]
    else:
        with trace.span("llm_call"):
            context = config.format_chunks(reranked)
            answer, grounded = _call_llm(user_query, context)
        trace.finish_span(grounded=grounded)
        final_sources = reranked
        reasoning_trace = []

    # 10. Output guardrails
    sources_for_check = [
        {"text": h["text"][:200], "metadata": h.get("metadata", {})}
        for h in final_sources
    ]
    with trace.span("output_guardrails"):
        output_check = guardrails.validate_output(answer, sources_for_check)
    trace.finish_span(result=output_check)

    if not output_check["safe"]:
        answer = guardrails.redact_pii(answer)

    # 11. Online evaluation (optional, adds latency)
    eval_result = None
    if enable_eval:
        with trace.span("evaluation"):
            source_texts = [h["text"] for h in final_sources]
            eval_result = evaluation.evaluate_answer(user_query, answer, source_texts)
        trace.finish_span(
            faithfulness=eval_result.faithfulness,
            relevance=eval_result.relevance,
            completeness=eval_result.completeness,
        )

    # 12. Cache the response (only if grounded and safe)
    if grounded and output_check["safe"]:
        cache.set_user_cache(user_query, answer)

    # Build response
    sources = [
        {
            "text": h["text"][:200] + "..." if len(h["text"]) > 200 else h["text"],
            "metadata": h.get("metadata", {}),
            "rerank_score": h.get("rerank_score", 0),
        }
        for h in final_sources
    ]

    result = {
        "answer": answer,
        "sources": sources,
        "cached": False,
        "grounded": grounded,
        "output_safety": output_check,
        "trace": trace.finish(),
    }

    if reasoning_trace:
        result["reasoning_trace"] = reasoning_trace
    if eval_result:
        result["evaluation"] = {
            "faithfulness": eval_result.faithfulness,
            "relevance": eval_result.relevance,
            "completeness": eval_result.completeness,
            "detail": eval_result.detail,
        }

    return result


def _call_llm(query: str, context: str) -> tuple[str, bool]:
    client = config.get_llm_client()

    system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Answer ONLY based on the provided context. If the context doesn't contain enough information, say so.
2. Cite your sources using [Source N] notation.
3. Be concise and direct.
4. At the end of your response, add a line: GROUNDED: YES or GROUNDED: NO
   - YES if your answer is fully supported by the provided sources
   - NO if you had to go beyond the sources or the sources were insufficient"""

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"""Context:
{context}

Question: {query}"""
        }],
    )

    full_answer = response.content[0].text.strip()
    grounded = "GROUNDED: NO" not in full_answer.upper()
    answer_lines = full_answer.split("\n")
    clean_lines = [l for l in answer_lines if not l.strip().upper().startswith("GROUNDED:")]
    answer = "\n".join(clean_lines).strip()

    return answer, grounded
