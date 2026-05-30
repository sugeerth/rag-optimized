"""Agentic retrieval loop.

Instead of single-shot retrieve -> generate, the LLM acts as an agent that can:
1. Decide if retrieved context is sufficient
2. Decompose complex questions into sub-questions
3. Request follow-up searches with refined queries
4. Self-reflect on its draft answer before finalizing

Max iterations prevents infinite loops.
"""

import config
import vector_store
import reranker


async def agentic_retrieve(
    query: str,
    initial_chunks: list[dict],
    trace_callback=None,
) -> dict:
    """Run the agentic retrieval loop.

    Args:
        query: Original user query
        initial_chunks: Chunks from the first retrieval pass
        trace_callback: Optional function(step_name, metadata) for tracing

    Returns:
        {
            "final_answer": str,
            "all_sources": list[dict],
            "iterations": int,
            "reasoning_trace": list[str],
            "grounded": bool,
        }
    """
    client = config.get_llm_client()
    all_chunks = list(initial_chunks)
    reasoning_trace = []
    iteration = 0

    while iteration < config.MAX_AGENT_ITERATIONS:
        iteration += 1
        context = config.format_chunks(all_chunks)

        if trace_callback:
            trace_callback(f"agent_iteration_{iteration}", {"num_chunks": len(all_chunks)})

        # Ask the agent to evaluate and decide next action
        decision = _agent_decide(client, query, context, iteration, reasoning_trace)
        reasoning_trace.append(f"Iteration {iteration}: {decision['reasoning']}")

        if decision["action"] == "answer":
            # Agent is satisfied — generate final answer
            answer, grounded = _generate_final_answer(client, query, context)
            reasoning_trace.append(f"Final answer generated (grounded={grounded})")

            return {
                "final_answer": answer,
                "all_sources": all_chunks,
                "iterations": iteration,
                "reasoning_trace": reasoning_trace,
                "grounded": grounded,
            }

        elif decision["action"] == "search":
            # Agent wants more information
            new_queries = decision.get("queries", [])
            reasoning_trace.append(f"Searching: {new_queries}")

            new_chunks = await _follow_up_search(new_queries)
            # Deduplicate against existing chunks
            existing_ids = {c["id"] for c in all_chunks}
            for chunk in new_chunks:
                if chunk["id"] not in existing_ids:
                    all_chunks.append(chunk)
                    existing_ids.add(chunk["id"])

            # Re-rank the expanded set
            all_chunks = reranker.rerank(query, all_chunks, top_k=config.TOP_K_RERANK + 3)

        elif decision["action"] == "decompose":
            # Agent wants to break the question into sub-questions
            sub_questions = decision.get("sub_questions", [])
            reasoning_trace.append(f"Decomposed into: {sub_questions}")

            for sq in sub_questions:
                sq_chunks = await _follow_up_search([sq])
                existing_ids = {c["id"] for c in all_chunks}
                for chunk in sq_chunks:
                    if chunk["id"] not in existing_ids:
                        all_chunks.append(chunk)
                        existing_ids.add(chunk["id"])

            all_chunks = reranker.rerank(query, all_chunks, top_k=config.TOP_K_RERANK + 5)

    # Max iterations reached — generate best-effort answer
    context = config.format_chunks(all_chunks)
    answer, grounded = _generate_final_answer(client, query, context)
    reasoning_trace.append(f"Max iterations reached. Best-effort answer (grounded={grounded})")

    return {
        "final_answer": answer,
        "all_sources": all_chunks,
        "iterations": iteration,
        "reasoning_trace": reasoning_trace,
        "grounded": grounded,
    }


def _agent_decide(
    client,
    query: str,
    context: str,
    iteration: int,
    previous_reasoning: list[str],
) -> dict:
    """Ask the agent to decide: answer, search more, or decompose."""
    history = "\n".join(previous_reasoning) if previous_reasoning else "First iteration."

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""You are a retrieval agent deciding whether you have enough context to answer a question.

Question: {query}

Retrieved context:
{context}

Previous reasoning:
{history}

Iteration: {iteration}/{config.MAX_AGENT_ITERATIONS}

Decide ONE action. Respond in JSON only:

Option 1 - You have enough context to answer well:
{{"action": "answer", "reasoning": "why you can answer now"}}

Option 2 - You need more specific information:
{{"action": "search", "reasoning": "what's missing", "queries": ["refined search query 1", "refined search query 2"]}}

Option 3 - The question is complex and needs to be broken down (use only on iteration 1):
{{"action": "decompose", "reasoning": "why decomposition helps", "sub_questions": ["sub q1", "sub q2", "sub q3"]}}

Be decisive. If context is mostly sufficient, choose "answer". Only search if clearly missing critical info."""
        }],
    )

    text = response.content[0].text.strip()
    parsed = config.parse_json_from_text(text)
    if parsed and "action" in parsed:
        return parsed
    return {"action": "answer", "reasoning": "parse_error_fallback"}


def _generate_final_answer(
    client,
    query: str,
    context: str,
) -> tuple[str, bool]:
    """Generate the final answer with self-reflection."""
    # Step 1: Draft answer
    draft_response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=1024,
        system="""You are a helpful assistant. Answer based ONLY on the provided context.
Cite sources using [Source N]. Be concise and accurate.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }],
    )
    draft = draft_response.content[0].text.strip()

    # Step 2: Self-reflection — check for hallucination
    reflection = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Review this draft answer for a RAG system.

Question: {query}

Sources:
{context}

Draft answer:
{draft}

Check:
1. Is every claim in the answer supported by the sources?
2. Are there any hallucinated facts?
3. Is anything important from the sources missing?

Respond in JSON:
{{"grounded": true/false, "issues": ["list of issues if any"], "revised_answer": "only if changes needed, else null"}}"""
        }],
    )

    text = reflection.content[0].text.strip()
    parsed = config.parse_json_from_text(text)
    if parsed:
        grounded = parsed.get("grounded", True)
        revised = parsed.get("revised_answer")
        if revised and revised is not None and str(revised).lower() != "null":
            return revised, grounded
        return draft, grounded
    return draft, True


async def _follow_up_search(queries: list[str]) -> list[dict]:
    """Run follow-up vector searches for the agent's refined queries."""
    embeddings = vector_store.embed_texts(queries)
    results = await vector_store.parallel_search(embeddings, top_k=config.TOP_K_RETRIEVAL)
    return results
