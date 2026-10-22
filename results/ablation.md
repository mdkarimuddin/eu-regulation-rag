| Variant | chunk (words) | chunks | recall@3 | recall@5 | recall@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| vector | 300 | 402 | 0.874 | 0.926 [0.87, 0.97] | 0.947 | 0.833 | 10.8 | 28.9 |
| bm25 | 300 | 402 | 0.874 | 0.911 [0.85, 0.96] | 0.958 | 0.777 | 0.8 | 1.7 |
| hybrid | 300 | 402 | 0.884 | 0.947 [0.90, 0.98] | 0.989 | 0.854 | 12.5 | 17.4 |
| hybrid+rerank | 300 | 402 | 0.905 | 0.937 [0.89, 0.98] | 0.995 | 0.846 | 467.4 | 641.5 |
| hybrid, chunk 150 | 150 | 735 | 0.916 | 0.947 [0.90, 0.98] | 0.974 | 0.863 | 17.2 | 36.6 |
| hybrid, chunk 600 | 600 | 282 | 0.916 | 0.958 [0.92, 0.99] | 0.974 | 0.840 | 17.1 | 26.6 |

n = 95 answerable questions. Brackets are 95% bootstrap intervals for recall@5.

Skipped: hybrid+rewrite: needs an LLM (set ANTHROPIC_API_KEY and pass --with-rewrite)
