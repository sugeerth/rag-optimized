"""RAG evaluation pipeline.

Two types of evaluation:
1. Retrieval metrics: recall@k, MRR, hit rate (require ground truth)
2. Answer quality: faithfulness, relevance, completeness (LLM-as-judge, no ground truth needed)

Can run online (per-query) or offline (batch over a test set).
"""

from dataclasses import dataclass

import config


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    hit_rate: float
    k: int


@dataclass
class AnswerMetrics:
    faithfulness: float  # 0-1: is answer supported by sources?
    relevance: float     # 0-1: does answer address the question?
    completeness: float  # 0-1: does answer cover all aspects?
    detail: str          # LLM explanation


# --- Retrieval Metrics (require ground truth) ---

def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute retrieval quality given ground truth relevant doc IDs."""
    retrieved_at_k = retrieved_ids[:k]

    # Recall@K: what fraction of relevant docs were retrieved
    hits = [rid for rid in retrieved_at_k if rid in relevant_ids]
    recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0

    # MRR: reciprocal rank of the first relevant result
    mrr = 0.0
    for i, rid in enumerate(retrieved_at_k):
        if rid in relevant_ids:
            mrr = 1.0 / (i + 1)
            break

    # Hit rate: did we get at least one relevant doc?
    hit_rate = 1.0 if hits else 0.0

    return RetrievalMetrics(
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
        hit_rate=hit_rate,
        k=k,
    )


# --- Answer Quality (LLM-as-Judge) ---

def evaluate_answer(
    query: str,
    answer: str,
    sources: list[str],
) -> AnswerMetrics:
    """Use LLM to evaluate answer quality along 3 dimensions."""
    client = config.get_llm_client()

    sources_text = "\n---\n".join(f"[Source {i+1}]: {s}" for i, s in enumerate(sources))

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""You are an evaluation judge for a RAG system. Score the answer on three dimensions.

Question: {query}

Sources provided to the system:
{sources_text}

Answer generated:
{answer}

Score each dimension from 0.0 to 1.0:

1. FAITHFULNESS: Is the answer factually supported by the sources? (1.0 = fully supported, 0.0 = hallucinated)
2. RELEVANCE: Does the answer address the question asked? (1.0 = perfectly relevant, 0.0 = off-topic)
3. COMPLETENESS: Does the answer cover all aspects of the question using available sources? (1.0 = comprehensive, 0.0 = missing key info)

Respond in this exact JSON format, nothing else:
{{"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0, "detail": "brief explanation"}}"""
        }],
    )

    text = response.content[0].text.strip()
    parsed = config.parse_json_from_text(text)
    if not parsed:
        return AnswerMetrics(faithfulness=0.0, relevance=0.0, completeness=0.0, detail="eval_parse_error")

    return AnswerMetrics(
        faithfulness=min(1.0, max(0.0, float(parsed.get("faithfulness", 0)))),
        relevance=min(1.0, max(0.0, float(parsed.get("relevance", 0)))),
        completeness=min(1.0, max(0.0, float(parsed.get("completeness", 0)))),
        detail=parsed.get("detail", ""),
    )


# --- Batch Evaluation ---

@dataclass
class TestCase:
    query: str
    relevant_doc_ids: list[str] | None = None  # For retrieval metrics
    expected_answer: str | None = None          # Optional reference


def run_batch_eval(test_cases: list[dict], pipeline_fn) -> dict:
    """Run evaluation over a batch of test cases.

    Args:
        test_cases: List of {"query": str, "relevant_doc_ids": list[str] (optional)}
        pipeline_fn: async function that takes a query and returns pipeline result dict

    Returns:
        Aggregate metrics across all test cases
    """
    import asyncio

    retrieval_scores = []
    answer_scores = []

    async def _run():
        for tc in test_cases:
            result = await pipeline_fn(tc["query"])

            # Answer quality (always possible)
            source_texts = [s["text"] for s in result.get("sources", [])]
            if source_texts and result.get("answer"):
                ametrics = evaluate_answer(tc["query"], result["answer"], source_texts)
                answer_scores.append({
                    "query": tc["query"],
                    "faithfulness": ametrics.faithfulness,
                    "relevance": ametrics.relevance,
                    "completeness": ametrics.completeness,
                    "detail": ametrics.detail,
                })

            # Retrieval quality (only if ground truth provided)
            if tc.get("relevant_doc_ids"):
                retrieved_ids = [s["metadata"].get("doc_name", "") for s in result.get("sources", [])]
                rmetrics = compute_retrieval_metrics(retrieved_ids, tc["relevant_doc_ids"])
                retrieval_scores.append({
                    "query": tc["query"],
                    "recall_at_k": rmetrics.recall_at_k,
                    "mrr": rmetrics.mrr,
                    "hit_rate": rmetrics.hit_rate,
                })

    asyncio.run(_run())

    def _avg(scores: list[dict], key: str) -> float:
        vals = [s[key] for s in scores if key in s]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "num_cases": len(test_cases),
        "answer_quality": {
            "avg_faithfulness": _avg(answer_scores, "faithfulness"),
            "avg_relevance": _avg(answer_scores, "relevance"),
            "avg_completeness": _avg(answer_scores, "completeness"),
        },
        "details": answer_scores,
    }

    if retrieval_scores:
        summary["retrieval_quality"] = {
            "avg_recall_at_k": _avg(retrieval_scores, "recall_at_k"),
            "avg_mrr": _avg(retrieval_scores, "mrr"),
            "avg_hit_rate": _avg(retrieval_scores, "hit_rate"),
        }
        summary["retrieval_details"] = retrieval_scores

    return summary
