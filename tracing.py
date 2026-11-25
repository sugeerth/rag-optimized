"""Pipeline observability and tracing.

Tracks every step of a RAG query: latency, token counts, scores, decisions.
Stores traces in memory and on disk for analysis.
"""

import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


TRACE_DIR = Path(__file__).parent / "traces"

# Traces live in a bounded in-memory ring. Writing one JSON file per query (the old
# behaviour) fills the disk under sustained load — a 1-hour soak makes millions of
# files — so disk persistence is opt-in (TRACE_PERSIST=1) and evicted traces are
# deleted so the directory stays bounded too.
PERSIST = os.getenv("TRACE_PERSIST", "") == "1"
if PERSIST:
    TRACE_DIR.mkdir(exist_ok=True)

MAX_RECENT = 500
_recent_traces: deque[dict] = deque()
_by_id: dict[str, dict] = {}


@dataclass
class Span:
    name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    metadata: dict = field(default_factory=dict)

    def finish(self, **extra_metadata):
        self.end_time = time.perf_counter()
        self.metadata.update(extra_metadata)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 4),
            "metadata": self.metadata,
        }


class Trace:
    def __init__(self, query: str):
        self.trace_id = uuid.uuid4().hex[:12]
        self.query = query
        self.start_time = time.perf_counter()
        self.spans: list[Span] = []
        self._current_span: Span | None = None

    def span(self, name: str) -> "Trace":
        """Start a new span. Use as context manager or call finish_span()."""
        self._current_span = Span(name=name)
        self.spans.append(self._current_span)
        return self

    def finish_span(self, **metadata):
        if self._current_span:
            self._current_span.finish(**metadata)
            self._current_span = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish_span()

    def finish(self) -> dict:
        total_ms = (time.perf_counter() - self.start_time) * 1000
        result = {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_ms": round(total_ms, 4),
            "spans": [s.to_dict() for s in self.spans],
        }
        _save_trace(result)
        return result


def _save_trace(trace: dict):
    _recent_traces.append(trace)
    _by_id[trace["trace_id"]] = trace
    if PERSIST:
        (TRACE_DIR / f"{trace['trace_id']}.json").write_text(json.dumps(trace, indent=2))
    while len(_recent_traces) > MAX_RECENT:
        old = _recent_traces.popleft()
        _by_id.pop(old["trace_id"], None)
        if PERSIST:
            (TRACE_DIR / f"{old['trace_id']}.json").unlink(missing_ok=True)


def get_recent_traces(limit: int = 20) -> list[dict]:
    limit = max(0, min(limit, MAX_RECENT))
    return list(_recent_traces)[-limit:] if limit else []


def get_trace(trace_id: str) -> dict | None:
    return _by_id.get(trace_id)


def get_latency_summary() -> dict:
    """Aggregate latency stats across recent traces."""
    if not _recent_traces:
        return {"count": 0}

    totals = [t["total_ms"] for t in _recent_traces]
    span_totals: dict[str, list[float]] = {}
    for t in _recent_traces:
        for s in t["spans"]:
            span_totals.setdefault(s["name"], []).append(s["duration_ms"])

    def _stats(values: list[float]) -> dict:
        values = sorted(values)
        return {
            "mean": round(sum(values) / len(values), 3),
            "p50": round(values[len(values) // 2], 3),
            "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 3),
            "p99": round(values[min(len(values) - 1, int(len(values) * 0.99))], 3),
            "min": round(values[0], 3),
            "max": round(values[-1], 3),
        }

    return {
        "count": len(totals),
        "total_ms": _stats(totals),
        "by_span": {name: _stats(vals) for name, vals in span_totals.items()},
    }
