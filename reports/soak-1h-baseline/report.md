# Soak test: 1.00 hours

**Passed: 0 errors and 0 failed safety checks across 1,908,763 requests.**

Started 2026-11-25 07:41:56 UTC. 8 closed-loop query clients against one uvicorn worker on a 4-CPU box, over 2,000 documents (16,714 chunks). 1,793,345 answers were checked against ground truth. Alongside: 895 document re-ingests (each one invalidates the cache; 1 index compactions), 79 LoRA fine-tunes, 57,046 prompt-injection or PII queries blocked, and 38,207 off-topic questions declined instead of answered. Latency is measured at the client, including HTTP.

| Metric | Value |
| --- | --- |
| Requests served | 1,908,763 |
| Average throughput | 530.2 req/s |
| Errors | 0 |
| Failed safety checks | 0 |
| Latency p50 / p95 / p99 / p99.9 / max | 13.488 / 29.35 / 39.114 / 61.504 / 2223.8 ms |
| Recall@5 | 99.2% |
| Answer accuracy | 81.9% |
| Cache hit rate | 61.0% |
| Server memory, min to max | 188.3 to 300.7 MB |

## Drift: first 10% of the run vs last 10%

| Metric | First 10% | Last 10% | Change |
| --- | --- | --- | --- |
| Throughput (req/s) | 586.864 | 426.125 | -27.4% |
| p50 latency (ms) | 12.456 | 17.907 | +43.8% |
| p99 latency (ms) | 37.5 | 47.108 | +25.6% |
| Server memory (MB) | 205.086 | 283.356 | +38.2% |
| Recall@5 | 0.995 | 0.993 | -0.2% |

Charts: open `report.html`. Raw data: `timeseries.jsonl`, `summary.json`.
