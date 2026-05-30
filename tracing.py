"""Pipeline observability and tracing.

Tracks every step of a RAG query: latency, token counts, scores, decisions.
Stores traces in memory and on disk for analysis.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


TRACE_DIR = Path(__file__).parent / "traces"
TRACE_DIR.mkdir(exist_ok=True)

# In-memory store for recent traces
_recent_traces: list[dict] = []
MAX_RECENT = 500


@dataclass
class Span:
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    metadata: dict = field(default_factory=dict)

    def finish(self, **extra_metadata):
        self.end_time = time.time()
        self.metadata.update(extra_metadata)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
        }


class Trace:
    def __init__(self, query: str):
        self.trace_id = str(uuid.uuid4())[:8]
        self.query = query
        self.start_time = time.time()
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
        total_ms = (time.time() - self.start_time) * 1000
        result = {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_ms": round(total_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
        }
        _save_trace(result)
        return result


def _save_trace(trace: dict):
    _recent_traces.append(trace)
    if len(_recent_traces) > MAX_RECENT:
        _recent_traces.pop(0)

    path = TRACE_DIR / f"{trace['trace_id']}.json"
    path.write_text(json.dumps(trace, indent=2))


def get_recent_traces(limit: int = 20) -> list[dict]:
    return _recent_traces[-limit:]


def get_trace(trace_id: str) -> dict | None:
    path = TRACE_DIR / f"{trace_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


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
        values.sort()
        return {
            "mean": round(sum(values) / len(values), 2),
            "p50": round(values[len(values) // 2], 2),
            "p95": round(values[int(len(values) * 0.95)], 2),
            "min": round(values[0], 2),
            "max": round(values[-1], 2),
        }

    return {
        "count": len(totals),
        "total_ms": _stats(totals),
        "by_span": {name: _stats(vals) for name, vals in span_totals.items()},
    }
