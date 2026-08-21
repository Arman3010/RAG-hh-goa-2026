# HH Goa 2026 — Task 2: Voice-Enabled RAG

Voice input → STT → chunking/retrieval (vector DB) → grounded answer generation, with guardrails.

## Team split

| Owner | Folder | Responsibility |
|---|---|---|
| Person A | `stt_harness/` | STT integration (Sarvam/ElevenLabs) + orchestration harness |
| Person B | `retrieval/` | Chunking strategies, vector DB, retrieval |
| Person C | `generation_guardrails/`, `tests/` | Answer generation, guardrails, latency benchmark, deploy |

## The contract

Everyone codes against `shared/interfaces.py`. Don't change those dataclass
shapes without telling the other two — they're what lets the three pieces
snap together on integration day instead of a scramble.

- `transcribe(audio_bytes) -> TranscriptionResult` — Person A
- `retrieve(query, top_k, strategy) -> RetrievalResult` — Person B
- `generate(query, chunks) -> GenerationResult` — Person C

`stt_harness/harness.py` wires all three into `run_pipeline(audio_bytes) -> PipelineResult`,
with retries and graceful degradation per stage.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # TODO: fill as each person adds deps
```

## Dataset

[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) on HuggingFace.

## Latency target note

The brief's <200ms end-to-end target is realistic for the retrieval leg
(chunking + vector search) with FAISS, not for STT+LLM generation over a
network call. Report stage-wise P50/P70/P100 separately and be transparent
about which leg the 200ms applies to.

## Run

```bash
python stt_harness/harness.py          # single smoke test
python tests/latency_benchmark.py      # P50/P70/P100 report
```
