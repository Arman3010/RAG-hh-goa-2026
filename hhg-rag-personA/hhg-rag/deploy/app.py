"""
Minimal FastAPI app exposing the full pipeline as a live HTTP endpoint --
this is what you deploy (Render/HF Spaces/Railway) to satisfy the
"Live working link" submission requirement.

Run locally:
    uvicorn deploy.app:app --reload --port 8000

Then open http://localhost:8000/docs for an interactive test UI (upload a
.wav file straight from the browser -- handy for your demo video too).
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from stt_harness.harness import run_pipeline

app = FastAPI(title="HH Goa 2026 - Voice RAG Pipeline")

# Allow calls from any origin -- fine for a hackathon demo; tighten this
# if you're worried about someone else hammering your free API credits.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "service": "voice-rag-pipeline"}


@app.post("/query")
async def query(audio: UploadFile = File(...)):
    """
    Upload a .wav/.mp3 file, get back the full PipelineResult as JSON:
    transcription, retrieval, generation, and per-stage latency.
    """
    audio_bytes = await audio.read()
    result = run_pipeline(audio_bytes)

    return {
        "query_text": result.query_text,
        "answer": result.generation.answer,
        "refused": result.generation.refused,
        "refusal_reason": result.generation.refusal_reason,
        "grounded": result.generation.grounded,
        "total_latency_ms": round(result.total_latency_ms, 2),
        "stage_latencies_ms": {k: round(v, 2) for k, v in result.stage_latencies_ms.items()},
        "chunks_used": result.generation.chunks_used,
        "error": result.error,
    }
