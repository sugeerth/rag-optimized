# Soak test: 1.00 hours

**Passed: 0 errors and 0 failed safety checks across 2,254,787 requests.**

Started 2026-11-25 13:57:06 UTC. 8 closed-loop query clients against one uvicorn worker on a 4-CPU box, over 2,000 documents (16,714 chunks). 2,118,442 answers were checked against ground truth. Alongside: 895 document re-ingests (each one invalidates the cache; 8 index compactions), 79 LoRA fine-tunes, 67,586 prompt-injection or PII queries blocked, and 45,078 off-topic questions declined instead of answered. Latency is measured at the client, including HTTP.

| Metric | Value |
| --- | --- |
| Requests served | 2,254,787 |
| Average throughput | 626.3 req/s |
| Errors | 0 |
| Failed safety checks | 0 |
| Latency p50 / p95 / p99 / p99.9 / max | 11.43 / 25.168 / 32.744 / 47.319 / 256.01 ms |
| Recall@5 | 99.5% |
| Answer accuracy | 82.3% |
| Cache hit rate | 63.2% |
| Server memory, min to max | 188.0 to 275.6 MB |

## Drift: first 10% of the run vs last 10%

| Metric | First 10% | Last 10% | Change |
| --- | --- | --- | --- |
| Throughput (req/s) | 600.667 | 627.606 | +4.5% |
| p50 latency (ms) | 12.225 | 11.512 | -5.8% |
| p99 latency (ms) | 34.38 | 32.879 | -4.4% |
| Server memory (MB) | 205.461 | 274.186 | +33.4% |
| Recall@5 | 0.995 | 0.994 | -0.1% |

Charts: open `report.html`. Raw data: `timeseries.jsonl`, `summary.json`.
