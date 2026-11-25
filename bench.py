"""Offline retrieval-quality + latency benchmark on the ground-truth corpus.

    python bench.py            # 2,000 docs, 2,000 sampled questions
    python bench.py 500 1000   # n_docs, n_questions
"""

import random
import sys
import time

import corpus
from engine import Engine


def run(n_docs: int = 2000, n_questions: int = 2000, seed: int = 7) -> dict:
    eng = Engine()
    docs, questions = corpus.build(n_docs, seed)
    t0 = time.perf_counter()
    for d in docs:
        eng.add_document(d["doc_name"], d["content"])
    ingest_s = time.perf_counter() - t0
    rng = random.Random(seed)
    sample = rng.sample(questions, min(n_questions, len(questions)))
    by_kind: dict[str, dict] = {}
    lat = []
    for q in sample:
        t = time.perf_counter()
        r = eng.query(q["query"], use_cache=False)
        lat.append((time.perf_counter() - t) * 1000)
        names = [s["metadata"]["doc_name"] for s in r["sources"]]
        k = by_kind.setdefault(q["kind"], {"n": 0, "hit1": 0, "hit5": 0, "rr": 0.0, "answer": 0})
        k["n"] += 1
        if q["doc_name"] in names:
            rank = names.index(q["doc_name"]) + 1
            k["hit5"] += 1
            k["hit1"] += rank == 1
            k["rr"] += 1.0 / rank
        k["answer"] += q["answer"].lower() in r["answer"].lower()
    lat.sort()
    out = {"docs": n_docs, "chunks": eng.n_alive, "vocabulary": len(eng.postings), "questions": len(sample),
           "ingest_docs_per_s": round(n_docs / ingest_s), "index_mb": eng.stats()["index_mb"],
           "latency_ms": {"p50": round(lat[len(lat) // 2], 3), "p95": round(lat[int(len(lat) * .95)], 3),
                          "p99": round(lat[int(len(lat) * .99)], 3)},
           "uncached_qps_single_core": round(1000 / (sum(lat) / len(lat))), "by_kind": {}}
    tot = {"n": 0, "hit1": 0, "hit5": 0, "rr": 0.0, "answer": 0}
    for kind, k in sorted(by_kind.items()):
        for f in tot:
            tot[f] += k[f]
    for kind, k in [*sorted(by_kind.items()), ("ALL", tot)]:
        out["by_kind"][kind] = {"n": k["n"], "recall@1": round(k["hit1"] / k["n"], 3),
                                "recall@5": round(k["hit5"] / k["n"], 3), "mrr": round(k["rr"] / k["n"], 3),
                                "answer_accuracy": round(k["answer"] / k["n"], 3)}
    return out


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    res = run(*args)
    by_kind = res.pop("by_kind")
    for k, v in res.items():
        print(f"{k:28} {v}")
    print(f"\n{'kind':12}{'n':>6}{'recall@1':>10}{'recall@5':>10}{'mrr':>8}{'answer':>9}")
    for kind, m in by_kind.items():
        print(f"{kind:12}{m['n']:>6}{m['recall@1']:>10}{m['recall@5']:>10}{m['mrr']:>8}{m['answer_accuracy']:>9}")
