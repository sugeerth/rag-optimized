"""Real hybrid retrieval engine in pure NumPy — no torch, no vector DB, no API key.

This replaces the old *simulated* demo pipeline. Every stage below actually runs
and is actually timed:

    guardrails -> 2-tier cache -> query rewrite -> multi-query expansion (PRF)
      -> embed -> hybrid search (BM25 + dense cosine, RRF-fused)
      -> feature reranker + MMR diversity -> agentic sufficiency loop
      -> extractive answer with citations -> output guardrails -> evaluation

What each piece is (and is not):
  * Sparse: a true BM25 inverted index (heading terms boosted, BM25F-lite).
  * Dense:  a hashed n-gram embedding (word unigrams/bigrams + char trigrams,
            signed feature hashing into DIM dims, L2-normalised). It is NOT a
            neural model — it buys typo/morphology robustness that BM25 lacks,
            which is exactly why fusing the two beats either alone.
  * Answers are extractive: the best-supported sentences, each cited. No LLM, so
    nothing can be hallucinated. Set ANTHROPIC_API_KEY + full deps for generation.

The whole pipeline runs in about a millisecond on one core. Knobs are the
constants right below.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import OrderedDict

import numpy as np

import guardrails
from chunking import chunk_by_structure
from tracing import Trace

# --- Knobs -------------------------------------------------------------------
DIM = 256                 # dense embedding width
BM25_K1, BM25_B = 1.4, 0.75
HEADING_BOOST = 2         # heading terms are counted this many extra times
CANDIDATES = 24           # per-retriever candidates fed to fusion
RRF_K = 60                # reciprocal-rank-fusion constant
RERANK_POOL = 24          # fused candidates the reranker scores
TOP_K = 5                 # sources returned
MMR_LAMBDA = 0.8          # 1.0 = pure relevance, lower = more diversity
GROUNDED_MIN_SCORE = 0.30 # top rerank score needed to call an answer grounded
GROUNDED_MIN_COVERAGE = 0.5
MAX_AGENT_ITERATIONS = 2
CACHE_MAX = 4096
ANSWER_SENTENCES = 3

STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in into is it
its me my of on or our should so than that the their them then there these they this to was we were what when where which
who whom why will with would you your about tell please explain describe give show list any some all also just""".split())

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_stem_cache: dict[str, str] = {}
_feat_cache: dict[str, tuple[int, float]] = {}


def stem(w: str) -> str:
    """Tiny suffix stripper (plural / -ing / -ed / trailing e). Memoised."""
    s = _stem_cache.get(w)
    if s is not None:
        return s
    s = w
    if len(s) > 4 and not s.isdigit():
        if s.endswith("ies"):
            s = s[:-3] + "y"
        elif s.endswith("ing") and len(s) > 5:
            s = s[:-3]
        elif s.endswith("ed") and len(s) > 4:
            s = s[:-2]
        elif s.endswith("s") and not s.endswith("ss"):
            s = s[:-1]
        if len(s) > 3 and s[-1] == s[-2] and s[-1] not in "ls":
            s = s[:-1]          # runn -> run
        if len(s) > 3 and s.endswith("e"):
            s = s[:-1]          # cache / caching -> cach
    if len(_stem_cache) > 200_000:
        _stem_cache.clear()
    _stem_cache[w] = s
    return s


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def terms(text: str) -> list[str]:
    """Content terms: lowercased, stopwords dropped, stemmed."""
    return [stem(w) for w in words(text) if w not in STOPWORDS]


def _feat(name: str) -> tuple[int, float]:
    """Feature hashing: feature string -> (dimension, +/-1 sign)."""
    f = _feat_cache.get(name)
    if f is None:
        h = zlib.crc32(name.encode())
        f = (h % DIM, 1.0 if (h >> 16) & 1 else -1.0)
        if len(_feat_cache) > 400_000:
            _feat_cache.clear()
        _feat_cache[name] = f
    return f


def _grams(t: str) -> set[str]:
    """Padded character bigrams — the candidate generator for typo correction."""
    padded = "^" + t + "$"
    return {padded[i:i + 2] for i in range(len(padded) - 1)}


def edit_distance(a: str, b: str, limit: int = 2) -> int:
    """Damerau-Levenshtein (adjacent transpositions count as one edit), early exit past `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class _Postings:
    """Postings list for one term; python lists for O(1) append, numpy view cached."""
    __slots__ = ("ids", "tfs", "_np")

    def __init__(self):
        self.ids: list[int] = []
        self.tfs: list[int] = []
        self._np = None

    def add(self, cid: int, tf: int):
        self.ids.append(cid)
        self.tfs.append(tf)
        self._np = None

    def arrays(self):
        if self._np is None:
            self._np = (np.asarray(self.ids, dtype=np.int64), np.asarray(self.tfs, dtype=np.float32))
        return self._np


class Engine:
    """In-memory hybrid index + the full query pipeline."""

    def __init__(self):
        self._reset_index()
        self.doc_hash: dict[str, int] = {}
        self.generation = 0                    # bumped on every mutation -> invalidates caches
        self._exact: OrderedDict = OrderedDict()   # L1: raw query string
        self._canon: OrderedDict = OrderedDict()   # L2: order/inflection-insensitive term set
        self.counters = {"queries": 0, "cache_exact": 0, "cache_canonical": 0, "blocked": 0,
                         "ungrounded": 0, "agent_requeries": 0, "ingests": 0, "compactions": 0}

    # ---- index storage ------------------------------------------------------

    def _reset_index(self):
        self.chunks: list[dict | None] = []    # id -> {"text","metadata","terms","tset","hset"}
        self.postings: dict[str, _Postings] = {}
        self.grams: dict[str, set[str]] = {}   # char bigram -> vocabulary terms (typo correction)
        self.doc_chunks: dict[str, list[int]] = {}
        cap = 1024
        self.E = np.zeros((cap, DIM), dtype=np.float32)
        self.dl = np.zeros(cap, dtype=np.float32)
        self.alive = np.zeros(cap, dtype=bool)
        self.n = 0            # slots used (including tombstones)
        self.n_alive = 0
        self.total_len = 0.0

    def _grow(self):
        cap = self.E.shape[0] * 2
        self.E = np.vstack([self.E, np.zeros_like(self.E)])
        self.dl = np.concatenate([self.dl, np.zeros_like(self.dl)])
        self.alive = np.concatenate([self.alive, np.zeros_like(self.alive)])
        assert self.E.shape[0] == cap

    def _add_chunk(self, text: str, metadata: dict):
        body = terms(text)
        head = terms(metadata.get("heading", ""))
        if not body and not head:
            return
        if self.n == self.E.shape[0]:
            self._grow()
        cid = self.n
        tf: dict[str, int] = {}
        for t in body:
            tf[t] = tf.get(t, 0) + 1
        for t in head:
            tf[t] = tf.get(t, 0) + HEADING_BOOST
        for t, c in tf.items():
            p = self.postings.get(t)
            if p is None:
                p = self.postings[t] = _Postings()
                for g in _grams(t):
                    self.grams.setdefault(g, set()).add(t)
            p.add(cid, c)
        length = len(body) + HEADING_BOOST * len(head)
        self.E[cid] = self.embed(metadata.get("heading", "") + " " + text)
        self.dl[cid] = length
        self.alive[cid] = True
        self.total_len += length
        self.chunks.append({"text": text, "metadata": metadata, "terms": body,
                            "tset": frozenset(body), "hset": frozenset(head)})
        self.doc_chunks.setdefault(metadata.get("doc_name", ""), []).append(cid)
        self.n += 1
        self.n_alive += 1

    def _remove_doc(self, doc_name: str):
        for cid in self.doc_chunks.pop(doc_name, []):
            if self.alive[cid]:
                self.alive[cid] = False
                self.total_len -= float(self.dl[cid])
                self.n_alive -= 1
                self.chunks[cid] = None

    def _compact(self):
        """Rebuild without tombstones (they'd otherwise leak memory under re-ingest churn)."""
        live = [c for c in self.chunks if c is not None]
        self._reset_index()
        for c in live:
            self._add_chunk(c["text"], c["metadata"])
        self.counters["compactions"] += 1

    # ---- public mutation API ------------------------------------------------

    def add_document(self, doc_name: str, text: str) -> int:
        """Chunk + index a document. Re-adding the same name replaces it."""
        digest = zlib.crc32(text.encode())
        if self.doc_hash.get(doc_name) == digest and doc_name in self.doc_chunks:
            return len(self.doc_chunks[doc_name])     # unchanged: keep the index and the caches warm
        self._remove_doc(doc_name)
        self.doc_hash[doc_name] = digest
        chunks = chunk_by_structure(text, doc_name, max_chunk_size=512, overlap=50)
        for ch in chunks:
            self._add_chunk(ch.text, ch.metadata)
        dead = self.n - self.n_alive
        if dead > 256 and dead > 0.3 * self.n:
            self._compact()
        self.generation += 1
        self._exact.clear()
        self._canon.clear()
        self.counters["ingests"] += 1
        return len(chunks)

    def remove_document(self, doc_name: str) -> bool:
        if doc_name not in self.doc_chunks:
            return False
        self._remove_doc(doc_name)
        self.doc_hash.pop(doc_name, None)
        self.generation += 1
        self._exact.clear()
        self._canon.clear()
        return True

    def reset(self):
        self._reset_index()
        self.doc_hash.clear()
        self.generation += 1
        self._exact.clear()
        self._canon.clear()

    def stats(self) -> dict:
        return {"chunks": self.n_alive, "documents": len(self.doc_chunks), "vocabulary": len(self.postings),
                "tombstones": self.n - self.n_alive, "embedding_dim": DIM,
                "index_mb": round((self.E[: self.n].nbytes + self.dl[: self.n].nbytes) / 1e6, 2),
                "cache_entries": len(self._exact) + len(self._canon), **self.counters}

    # ---- scoring primitives -------------------------------------------------

    def idf(self, t: str) -> float:
        p = self.postings.get(t)
        df = len(p.ids) if p else 0
        return math.log(1.0 + (max(self.n_alive, 1) - df + 0.5) / (df + 0.5))

    def qweight(self, t: str) -> float:
        """Weight of a query term when judging coverage. A word the index has never seen
        can't be evidence for or against a chunk, so it counts little instead of (as raw
        IDF would have it) the most."""
        return self.idf(t) if t in self.postings else 0.5

    def embed(self, text: str, query: bool = False) -> np.ndarray:
        """Hashed n-gram embedding. Query side is IDF-weighted so rare terms dominate."""
        v = np.zeros(DIM, dtype=np.float32)
        ws = [w for w in words(text) if w not in STOPWORDS]
        counts: dict[str, float] = {}
        prev = None
        for w in ws:
            s = stem(w)
            wt = self.idf(s) if query else 1.0
            counts["u:" + s] = counts.get("u:" + s, 0.0) + wt
            if prev is not None:
                k = "b:" + prev + "_" + s
                counts[k] = counts.get(k, 0.0) + 0.6 * wt
            prev = s
            padded = "^" + w + "$"
            for i in range(len(padded) - 2):
                k = "c:" + padded[i:i + 3]
                counts[k] = counts.get(k, 0.0) + 0.3 * wt
        for k, c in counts.items():
            idx, sign = _feat(k)
            v[idx] += sign * (1.0 + math.log(c) if c > 1 else c)
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def _bm25(self, q_terms: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        if not self.n_alive:
            return scores
        avgdl = self.total_len / self.n_alive or 1.0
        for t in set(q_terms):
            p = self.postings.get(t)
            if p is None:
                continue
            ids, tfs = p.arrays()
            denom = tfs + BM25_K1 * (1.0 - BM25_B + BM25_B * self.dl[ids] / avgdl)
            scores[ids] += self.idf(t) * tfs * (BM25_K1 + 1.0) / denom
        scores[~self.alive[: self.n]] = 0.0
        return scores

    @staticmethod
    def _top(scores: np.ndarray, k: int, floor: float) -> list[int]:
        k = min(k, scores.shape[0])
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [int(i) for i in idx if scores[i] > floor]

    def hybrid_search(self, variants: list[str], vecs: list[np.ndarray] | None = None) -> tuple[list[int], dict[int, dict]]:
        """BM25 + dense for every query variant, fused with reciprocal rank fusion."""
        rrf: dict[int, float] = {}
        detail: dict[int, dict] = {}
        if vecs is None:
            vecs = [self.embed(q, query=True) for q in variants]
        # One BLAS call scores every variant against every chunk.
        sims = self.E[: self.n] @ np.stack(vecs, axis=1)
        sims[~self.alive[: self.n]] = -1.0
        for i, q in enumerate(variants):
            bm = self._bm25(terms(q))
            de = sims[:, i]
            for name, scores, floor in (("bm25", bm, 0.0), ("dense", de, 0.05)):
                for rank, cid in enumerate(self._top(scores, CANDIDATES, floor)):
                    rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                    d = detail.setdefault(cid, {"bm25": 0.0, "dense": 0.0})
                    d[name] = max(d[name], float(scores[cid]))
        ranked = sorted(rrf, key=rrf.get, reverse=True)[:RERANK_POOL]
        for cid in ranked:
            detail[cid]["rrf"] = rrf[cid]
        return ranked, detail

    def rerank(self, query: str, cands: list[int], detail: dict[int, dict], top_k: int = TOP_K,
               qvec: np.ndarray | None = None) -> list[dict]:
        """Feature reranker (coverage, BM25, dense, phrase, heading) + MMR diversity."""
        q = terms(query)
        if not cands:
            return []
        q_set = set(q)
        q_idf = {t: self.qweight(t) for t in q_set}
        idf_sum = sum(q_idf.values()) or 1.0
        q_bi = {(a, b) for a, b in zip(q, q[1:])}
        max_bm = max((detail[c]["bm25"] for c in cands), default=0.0) or 1.0
        if qvec is None:
            qvec = self.embed(query, query=True)
        scored = []
        for cid in cands:
            ch = self.chunks[cid]
            if ch is None:
                continue
            seen = ch["tset"] | ch["hset"]
            cov = sum(q_idf[t] for t in q_set if t in seen) / idf_sum
            bi = 0.0
            if q_bi:
                ct = ch["terms"]
                cbi = {(a, b) for a, b in zip(ct, ct[1:])}
                bi = len(q_bi & cbi) / len(q_bi)
            head = sum(q_idf[t] for t in q_set if t in ch["hset"]) / idf_sum
            dense = max(0.0, float(self.E[cid] @ qvec))
            score = 0.40 * cov + 0.22 * detail[cid]["bm25"] / max_bm + 0.18 * dense + 0.12 * bi + 0.08 * head
            scored.append((score, cid, cov))
        scored.sort(reverse=True)
        # MMR: greedily pick relevant-but-not-redundant chunks.
        ids = [cid for _, cid, _ in scored]
        sim = (self.E[ids] @ self.E[ids].T).tolist()     # candidate-candidate similarity, once
        redundancy = [0.0] * len(scored)
        left = list(range(len(scored)))
        order: list[int] = []
        while left and len(order) < top_k:
            best = max(left, key=lambda i: MMR_LAMBDA * scored[i][0] - (1.0 - MMR_LAMBDA) * redundancy[i])
            left.remove(best)
            order.append(best)
            row = sim[best]
            for i in left:
                if row[i] > redundancy[i]:
                    redundancy[i] = row[i]
        picked = sorted((scored[i] for i in order), reverse=True)
        return [{"id": cid, "text": self.chunks[cid]["text"], "metadata": self.chunks[cid]["metadata"],
                 "rerank_score": round(s, 3), "coverage": round(cov, 3),
                 "scores": {"bm25": round(detail[cid]["bm25"], 3), "dense": round(detail[cid]["dense"], 3),
                            "rrf": round(detail[cid].get("rrf", 0.0), 5)}}
                for s, cid, cov in picked]

    # ---- query understanding -----------------------------------------------

    def correct(self, term: str) -> str | None:
        """Closest vocabulary term for an out-of-vocabulary one: candidates share character
        bigrams, the winner is within 1 edit (2 for long words), ties go to the commoner term."""
        if len(term) < 4 or term.isdigit():
            return None
        votes: dict[str, int] = {}
        for g in _grams(term):
            for cand in self.grams.get(g, ()):
                votes[cand] = votes.get(cand, 0) + 1
        limit = 2 if len(term) >= 8 else 1
        best = None
        for cand in sorted(votes, key=votes.get, reverse=True)[:40]:
            df = len(self.postings[cand].ids)
            if not df:
                continue
            d = edit_distance(term, cand, limit)
            if d <= limit:
                key = (d, -df, cand)
                if best is None or key < best:
                    best = key
        return best[2] if best else None

    def rewrite(self, query: str) -> tuple[str, dict[str, str]]:
        """Keyword-focused rewrite: drop filler/stopwords and fix typos against the index
        vocabulary. Returns (rewritten query, {typo: correction})."""
        kept, fixes = [], {}
        for w in words(query):
            if w in STOPWORDS:
                continue
            if stem(w) not in self.postings:
                fix = self.correct(stem(w))
                if fix:
                    fixes[w] = fix
                    w = fix
            kept.append(w)
        return " ".join(kept) or query.strip(), fixes

    def expand(self, query: str) -> list[str]:
        """Pseudo-relevance feedback: add the most distinctive terms of the top BM25 hits."""
        q = terms(query)
        bm = self._bm25(q)
        top = self._top(bm, 3, 0.0)
        if not top:
            return []
        weights: dict[str, float] = {}
        q_set = set(q)
        for cid in top:
            for t in self.chunks[cid]["tset"]:
                if t not in q_set and len(t) > 2:
                    weights[t] = weights.get(t, 0.0) + self.idf(t)
        extra = sorted(weights, key=weights.get, reverse=True)[:4]
        return extra

    def _coverage(self, query_terms: list[str], sources: list[dict]) -> tuple[float, list[str]]:
        q_set = set(query_terms)
        if not q_set:
            return 1.0, []
        seen = set()
        for s in sources:
            ch = self.chunks[s["id"]]
            if ch:
                seen |= ch["tset"] | ch["hset"]
        idf = {t: self.qweight(t) for t in q_set}
        missing = sorted(t for t in q_set if t not in seen)
        return sum(idf[t] for t in q_set if t in seen) / (sum(idf.values()) or 1.0), missing

    # ---- answer synthesis + evaluation --------------------------------------

    def synthesize(self, query: str, sources: list[dict]) -> str:
        """Extractive answer: the best-supported sentences, each with its citation."""
        q_set = set(terms(query))
        idf = {t: self.qweight(t) for t in q_set}
        idf_sum = sum(idf.values()) or 1.0
        cands = []
        best_source = sources[0]["rerank_score"] if sources else 0.0
        for si, s in enumerate(sources):
            if s["rerank_score"] < 0.75 * best_source:
                continue    # clearly weaker source (often a look-alike entity): don't quote it
            # A sentence inherits the subject terms of its chunk's heading at a discount,
            # so "It listens on port 8443." still scores for a question naming the service.
            hset = self.chunks[s["id"]]["hset"] if self.chunks[s["id"]] else frozenset()
            for sent in _SENT_RE.split(" ".join(s["text"].split())):
                if len(sent) < 20:
                    continue
                st = set(terms(sent))
                cov = sum(idf[t] * (1.0 if t in st else 0.4 if t in hset else 0.0) for t in q_set) / idf_sum
                own = sum(idf[t] for t in q_set if t in st) / idf_sum
                if own <= 0:
                    continue
                cands.append((0.75 * cov + 0.25 * s["rerank_score"], si, sent, st))
        cands.sort(key=lambda c: -c[0])
        chosen = []
        floor = 0.65 * cands[0][0] if cands else 0.0   # don't pad the answer with weak sentences
        for sc, si, sent, st in cands:
            if sc < floor:
                break
            if any(len(st & o[3]) / max(1, len(st | o[3])) > 0.6 for o in chosen):
                continue
            chosen.append((sc, si, sent, st))
            if len(chosen) == ANSWER_SENTENCES:
                break
        if not chosen:
            return ""
        return " ".join(f"{sent} [Source {si + 1}]" for _, si, sent, _ in chosen)

    def evaluate(self, query: str, answer: str, sources: list[dict]) -> dict:
        """Deterministic, computed metrics (no LLM judge, no random numbers)."""
        q_set = set(terms(query))
        idf = {t: self.qweight(t) for t in q_set}
        idf_sum = sum(idf.values()) or 1.0
        src_terms = [set(terms(s["text"])) for s in sources]
        sent_scores = []
        for part in re.split(r"\s*\[Source (\d+)\]\s*", answer):
            if not part or part.isdigit():
                continue
            st = set(terms(part))
            if st:
                sent_scores.append(max((len(st & src) / len(st) for src in src_terms), default=0.0))
        a_terms = set(terms(re.sub(r"\[Source \d+\]", " ", answer)))
        all_src = set().union(*src_terms) if src_terms else set()
        faith = sum(sent_scores) / len(sent_scores) if sent_scores else 0.0
        rel = sum(idf[t] for t in q_set if t in a_terms) / idf_sum
        comp = sum(idf[t] for t in q_set if t in all_src) / idf_sum
        prec = sum(1 for s in sources if s["coverage"] > 0) / len(sources) if sources else 0.0
        return {"faithfulness": round(faith, 2), "relevance": round(rel, 2), "completeness": round(comp, 2),
                "context_precision": round(prec, 2),
                "detail": "Computed, not sampled: faithfulness = share of answer terms found in the cited "
                          "sources; relevance = IDF-weighted query terms covered by the answer; completeness "
                          "= query terms covered by retrieved context; context precision = "
                          f"{round(prec * 100)}% of sources match the query."}

    # ---- the pipeline -------------------------------------------------------

    def _cache_put(self, store: OrderedDict, key, value):
        store[key] = value
        store.move_to_end(key)
        if len(store) > CACHE_MAX:
            store.popitem(last=False)

    def query(self, query: str, evaluate: bool = False, use_cache: bool = True) -> dict:
        query = query.strip()
        trace = Trace(query)
        self.counters["queries"] += 1

        with trace.span("input_guardrails"):
            check = guardrails.validate_input(query)
            trace.finish_span(**check)
        if not check["safe"]:
            self.counters["blocked"] += 1
            return {"answer": "I can't process this query due to safety concerns.", "blocked": True,
                    "issues": check["issues"], "trace": trace.finish()}

        with trace.span("user_cache_check"):
            q_terms = terms(query)
            exact_key = (query, evaluate)
            canon_key = (tuple(sorted(set(q_terms))), evaluate)
            hit, tier = None, None
            if use_cache:
                hit = self._exact.get(exact_key)
                tier = "exact"
                if hit is None and canon_key[0]:
                    hit = self._canon.get(canon_key)
                    tier = "canonical"
            trace.finish_span(hit=hit is not None, tier=tier if hit is not None else None)
        if hit is not None:
            self.counters["cache_" + tier] += 1
            return {**hit, "cached": True, "cache_tier": tier, "trace": trace.finish()}

        with trace.span("query_rewrite"):
            rewritten, fixes = self.rewrite(query)
            if fixes:  # answer the question the user meant
                q_terms = terms(rewritten)
            trace.finish_span(original=query, rewritten=rewritten, corrections=fixes)
        effective = rewritten if fixes else query

        with trace.span("query_expansion"):
            extra = self.expand(rewritten)
            variants = [effective]
            if rewritten != effective.lower():
                variants.append(rewritten)
            if extra:
                variants.append(rewritten + " " + " ".join(extra))
            trace.finish_span(num_variants=len(variants), feedback_terms=extra)

        with trace.span("embedding"):
            vecs = [self.embed(v, query=True) for v in variants]
            trace.finish_span(num_embeddings=len(variants), dim=DIM)

        with trace.span("parallel_vector_search"):
            cands, detail = self.hybrid_search(variants, vecs)
            trace.finish_span(searched_chunks=self.n_alive, num_hits=len(cands), fusion="rrf(bm25,dense)")

        with trace.span("reranking"):
            sources = self.rerank(effective, cands, detail, qvec=vecs[0])
            trace.finish_span(before=len(cands), after=len(sources),
                              top_score=sources[0]["rerank_score"] if sources else 0.0)

        reasoning = [f"Corrected likely typos: {fixes}"] if fixes else []
        with trace.span("agentic_retrieval"):
            iterations = 1
            coverage, missing = self._coverage(q_terms, sources)
            reasoning.append(f"Iteration 1: searched {self.n_alive} chunks with {len(variants)} query variants, "
                             f"kept top {len(sources)}. Context covers {coverage:.0%} of the question's key terms.")
            while missing and iterations < MAX_AGENT_ITERATIONS:
                known = [t for t in missing if t in self.postings]
                if not known:
                    reasoning.append(f"Terms not in the index at all: {missing} — nothing more to retrieve.")
                    break
                iterations += 1
                self.counters["agent_requeries"] += 1
                reasoning.append(f"Iteration {iterations}: context is missing {known}; searching again for them.")
                c2, d2 = self.hybrid_search([" ".join(known)])
                for cid, d in d2.items():
                    old = detail.get(cid)
                    if old is None:
                        detail[cid] = {"bm25": 0.0, "dense": d["dense"], "rrf": d.get("rrf", 0.0)}
                merged = list(dict.fromkeys(cands + c2))
                # Re-score against the ORIGINAL question so the follow-up can't drift off-topic.
                bm = self._bm25(q_terms)
                for cid in merged:
                    detail[cid]["bm25"] = float(bm[cid])
                sources = self.rerank(effective, merged, detail, qvec=vecs[0])
                new_cov, missing = self._coverage(q_terms, sources)
                reasoning.append(f"After re-search coverage is {new_cov:.0%}" +
                                 (" — improved." if new_cov > coverage + 1e-9 else " — no better; keeping best set."))
                coverage = new_cov
            top_score = sources[0]["rerank_score"] if sources else 0.0
            grounded = bool(sources) and top_score >= GROUNDED_MIN_SCORE and coverage >= GROUNDED_MIN_COVERAGE
            answer = self.synthesize(effective, sources) if grounded else ""
            if not answer:
                grounded = False
                self.counters["ungrounded"] += 1
                answer = ("I couldn't find enough evidence in the indexed documents to answer that. "
                          "Try rephrasing, or ingest a document that covers it.")
                reasoning.append(f"Declining to answer: top score {top_score:.2f}, coverage {coverage:.0%} "
                                 "— below the grounding threshold.")
            else:
                reasoning.append(f"Final answer assembled from cited sentences (grounded=True, top score {top_score:.2f}).")
            trace.finish_span(iterations=iterations, grounded=grounded, coverage=round(coverage, 3))

        with trace.span("output_guardrails"):
            out_check = guardrails.validate_output(answer, sources)
            if not out_check["safe"]:
                answer = guardrails.redact_pii(answer)
            trace.finish_span(**out_check)

        public = [{"text": s["text"][:300] + ("..." if len(s["text"]) > 300 else ""), "metadata": s["metadata"],
                   "rerank_score": s["rerank_score"], "scores": s["scores"]} for s in sources]
        result = {"answer": answer, "sources": public, "cached": False, "grounded": grounded,
                  "output_safety": out_check, "reasoning_trace": reasoning,
                  "retrieval": {"variants": variants, "corrections": fixes, "coverage": round(coverage, 3), "engine": "hybrid-bm25+dense"}}
        if evaluate:
            with trace.span("evaluation"):
                result["evaluation"] = self.evaluate(effective, answer, sources)
                trace.finish_span(**{k: v for k, v in result["evaluation"].items() if k != "detail"})

        if use_cache:
            self._cache_put(self._exact, exact_key, result)
            if canon_key[0]:
                self._cache_put(self._canon, canon_key, result)
        return {**result, "trace": trace.finish()}


ENGINE = Engine()
