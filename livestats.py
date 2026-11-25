"""Live, rolling server stats for the Mission Control tab and the soak test.

One bucket per wall-clock second (kept for an hour). Each bucket holds counts and
a bounded reservoir of latency samples, so memory stays flat no matter how long
the server runs or how hard it is hit.
"""

import os
import random
import time
from collections import deque

WINDOW_SECONDS = 3600
SAMPLES_PER_SECOND = 256

_START = time.time()
_buckets: deque[dict] = deque(maxlen=WINDOW_SECONDS)
_totals = {"requests": 0, "queries": 0, "errors": 0, "cache_hits": 0, "blocked": 0, "ungrounded": 0}
_PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 1e6 if hasattr(os, "sysconf") else 4096 / 1e6


def _bucket() -> dict:
    now = int(time.time())
    if not _buckets or _buckets[-1]["t"] != now:
        _buckets.append({"t": now, "requests": 0, "queries": 0, "errors": 0, "cache_hits": 0,
                         "blocked": 0, "seen": 0, "lat": []})
    return _buckets[-1]


def record_request(error: bool = False) -> None:
    b = _bucket()
    b["requests"] += 1
    _totals["requests"] += 1
    if error:
        b["errors"] += 1
        _totals["errors"] += 1


def record_query(latency_ms: float, cached: bool = False, blocked: bool = False, grounded: bool = True) -> None:
    b = _bucket()
    b["queries"] += 1
    _totals["queries"] += 1
    if cached:
        b["cache_hits"] += 1
        _totals["cache_hits"] += 1
    if blocked:
        b["blocked"] += 1
        _totals["blocked"] += 1
    elif not grounded:
        _totals["ungrounded"] += 1
    # Reservoir sampling keeps an unbiased latency sample per second in O(1) memory.
    b["seen"] += 1
    if len(b["lat"]) < SAMPLES_PER_SECOND:
        b["lat"].append(latency_ms)
    else:
        j = random.randrange(b["seen"])
        if j < SAMPLES_PER_SECOND:
            b["lat"][j] = latency_ms


def rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * _PAGE_MB, 1)
    except OSError:
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:
            return 0.0


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    return round(sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))], 4)


def snapshot(window: int = 60, series: int = 120) -> dict:
    """Totals + the last `window` seconds summarised + a per-second series for charts."""
    now = int(time.time())
    recent = [b for b in _buckets if now - window < b["t"] <= now]
    lat = sorted(x for b in recent for x in b["lat"])
    q = sum(b["queries"] for b in recent)
    span = max(1, min(window, now - int(_START) or 1))
    by_t = {b["t"]: b for b in _buckets if b["t"] > now - series}
    points = []
    for t in range(now - series + 1, now + 1):
        b = by_t.get(t)
        bl = sorted(b["lat"]) if b else []
        points.append({"t": t, "qps": b["queries"] if b else 0, "rps": b["requests"] if b else 0,
                       "p50": _pct(bl, 0.5) if bl else None, "p99": _pct(bl, 0.99) if bl else None,
                       "hit_rate": round(b["cache_hits"] / b["queries"], 3) if b and b["queries"] else None})
    return {
        "uptime_s": round(time.time() - _START, 1),
        "rss_mb": rss_mb(),
        "totals": dict(_totals),
        "window_s": window,
        "window": {
            "queries": q,
            "qps": round(q / span, 1),
            "rps": round(sum(b["requests"] for b in recent) / span, 1),
            "errors": sum(b["errors"] for b in recent),
            "cache_hit_rate": round(sum(b["cache_hits"] for b in recent) / q, 3) if q else 0.0,
            "latency_ms": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "p99": _pct(lat, 0.99),
                           "max": round(lat[-1], 4) if lat else 0.0},
        },
        "series": points,
    }
