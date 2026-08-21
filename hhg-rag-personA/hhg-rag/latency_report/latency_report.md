# Latency Report

Queries tested: 30

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| total | 970.54 | 1178.66 | 2264.19 |
| stt | 1003.98 | 1206.42 | 2264.15 |
| retrieval | 0.01 | 0.01 | 0.01 |
| generation | 0.03 | 0.04 | 1.56 |

Note: the <200ms target applies to the retrieval leg (chunking + vector search), not STT+LLM generation over a network call.
