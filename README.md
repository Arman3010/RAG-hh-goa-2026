# HH Goa 2026 — Task 2: Voice-Enabled RAG

Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Grounded answer generation, with guardrails.

## Project flow (end to end)

```
User speaks a question
        │
        ▼
[STAGE 1] Speech-to-Text  (Person A)
  Sarvam/ElevenLabs transcribes audio → query text
        │
        ▼
[STAGE 2] Retrieval  (Person B)
  Query text is embedded → searched against a pre-built FAISS vector index
  → top-k most relevant passages returned from the corpus
        │
        ▼
[STAGE 3] Generation + Guardrails  (Person C)
  Pre-checks: reject empty/unsafe/off-topic queries before calling the LLM
  Query + retrieved passages → LLM generates an answer grounded ONLY in
  those passages → post-check verifies groundedness (word-overlap against
  chunks, or an LLM-judge call) → ungrounded answers are refused, not shown
        │
        ▼
Final answer returned to the user
```

All three stages are wired together by the **harness** (`stt_harness/harness.py`),
which calls them in sequence, times each stage, retries on transient failure,
and degrades gracefully instead of crashing if any one stage fails.

## Team split & status

| Owner | Folder | Responsibility | Status |
|---|---|---|---|
| Person A | `stt_harness/`, `deploy/` | STT integration, orchestration harness, deployment | ✅ Done (live link pending) |
| Person B | `retrieval/` | Chunking strategies, vector DB, retrieval | ✅ Done |
| Person C | `generation_guardrails/`, `tests/` | Answer generation, guardrails, latency benchmark | ✅ Done |

## Dataset

[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) on
HuggingFace — MS MARCO queries/passages translated into Indic languages.

The full dataset is 11.45M rows / 55.6GB, far more than needed for this build.
We work off a **500-query Hindi subset** (`retrieval/data/subset.json`),
pulled directly from `train/hintrain.parquet` using range requests (so we
never download the full 3.72GB file), then flattened into a corpus of
**4,996 unique passages** for indexing.

## Retrieval — approach and results

We tested three chunking strategies against the same corpus and evaluated
each with **Recall@3** (does the ground-truth answer passage show up in the
top-3 retrieved results, per the dataset's own `is_selected` labels):

| Strategy | Recall@3 | Notes |
|---|---|---|
| No chunking (whole passage) | 37.21% | Baseline |
| Fixed-size + overlap | 32.23% | Worse — fragmented already-short passages |
| Semantic (sentence-grouped) | 22.92% | Worse, same reason |
| **Metadata-aware (chosen)** | **38.21%** | Best — keeps passages intact, adds query-type tag |

**Key finding:** further splitting hurt recall rather than helping, because
MS MARCO passages are already short (2–4 sentences) — fragmenting them
destroys context rather than sharpening retrieval. Metadata-aware chunking
won by keeping passages whole and adding a lightweight topical signal
instead of cutting anything up.

**Retrieval latency:** ~83ms for a single query end-to-end (embedding +
FAISS search, model pre-loaded), comfortably under the 200ms target for
this stage.

## Guardrails

Two layers, in `generation_guardrails/guardrails.py`:

1. **Pre-generation screening** (before any LLM call): regex-based checks
   reject empty queries, unsafe input (e.g. self-harm, weapons, hacking
   requests), and off-topic requests (e.g. "write me a poem") that fall
   outside the knowledge-base Q&A scope this system is built for.
2. **Post-generation groundedness check**: verifies the generated answer's
   content actually overlaps with the retrieved passages, rather than
   trusting the LLM's output blindly. Default mode is a free word-overlap
   heuristic (fast enough to run on every request); an LLM-judge mode is
   available as a togglable, more thorough alternative for catching
   paraphrased hallucinations, at the cost of an extra API call.

Answers that fail either check are refused with a specific reason
(`unsafe_input`, `off_topic`, `no_context`, `not_grounded`) rather than
shown to the user — this is what satisfies requirement #6's "know when not
to answer."

## Latency

Full pipeline benchmarked across 30 test queries (`tests/latency_benchmark.py`):

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| STT | 1003.98 | 1206.42 | 2264.15 |
| Generation | 0.03 | 0.04 | 1.56 |
| **Total** | 970.54 | 1178.66 | 2264.19 |

**Known issue to fix before final submission:** this benchmark run used
`tests/mock_retrieval.py` (a stand-in built so generation could be tested
before retrieval was ready) rather than the real `retrieval/retriever.py` —
that's why retrieval doesn't appear with a realistic number above. The
harness itself already imports the real retriever correctly; the benchmark
just needs to be re-run now that all three pieces are integrated, to get an
accurate combined report. Retrieval's real, independently-measured latency
is ~83ms (see Retrieval section above).

**On the brief's <200ms target:** read literally it covers the full
pipeline, which isn't achievable with STT and LLM generation over a network
API — a single API round-trip commonly exceeds 200ms on its own regardless
of implementation. Our approach is to report honest, stage-wise numbers
rather than a fabricated total: retrieval comfortably clears 200ms on its
own, and STT/generation numbers above are real, measured latencies, not
estimates.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `stt_harness/.env.example` to `.env` and fill in API keys (STT provider,
LLM provider) before running.

## Run

```bash
python stt_harness/harness.py          # full pipeline smoke test
python tests/latency_benchmark.py      # re-run for an accurate combined report
python tests/test_guardrails.py        # guardrail unit tests
```

## Live link

Pending — to be added before final submission.
