"""
Person A owns this file.

The harness is the orchestration layer: it calls transcribe -> retrieve -> generate,
times each stage, retries on transient failure, and never lets one stage's
exception kill the whole request -- it degrades gracefully and records the
error on PipelineResult instead.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shared.interfaces import (
    Timer, PipelineResult, TranscriptionResult, RetrievalResult, GenerationResult,
)
from stt_harness.stt import transcribe          # Person A implements shared/interfaces.transcribe here
from retrieval.retriever import retrieve         # Person B implements it here
from generation_guardrails.generator import generate  # Person C implements it here


def run_pipeline(audio_bytes: bytes, max_retries: int = 2) -> PipelineResult:
    stage_latencies = {}

    # --- Stage 1: Speech-to-text ---
    transcription = None
    for attempt in range(max_retries + 1):
        try:
            with Timer() as t:
                transcription = transcribe(audio_bytes)
            stage_latencies["stt"] = t.elapsed_ms
            break
        except Exception as e:
            if attempt == max_retries:
                return PipelineResult(
                    query_text="",
                    transcription=None,
                    retrieval=RetrievalResult(query="", chunks=[]),
                    generation=GenerationResult(answer="", grounded=False, refused=True,
                                                 refusal_reason="stt_failed"),
                    error=f"STT failed after {max_retries + 1} attempts: {e}",
                    stage_latencies_ms=stage_latencies,
                )

    query_text = transcription.text if transcription else ""

    if not query_text.strip():
        return PipelineResult(
            query_text="",
            transcription=transcription,
            retrieval=RetrievalResult(query="", chunks=[]),
            generation=GenerationResult(answer="", grounded=False, refused=True,
                                         refusal_reason="empty_transcription"),
            stage_latencies_ms=stage_latencies,
        )

    # --- Stage 2: Retrieval ---
    try:
        with Timer() as t:
            retrieval = retrieve(query_text)
        stage_latencies["retrieval"] = t.elapsed_ms
    except Exception as e:
        retrieval = RetrievalResult(query=query_text, chunks=[])
        stage_latencies["retrieval"] = 0.0

    # --- Stage 3: Generation + guardrails ---
    try:
        with Timer() as t:
            generation = generate(query_text, retrieval.chunks)
        stage_latencies["generation"] = t.elapsed_ms
    except Exception as e:
        generation = GenerationResult(answer="", grounded=False, refused=True,
                                       refusal_reason="generation_failed")
        stage_latencies["generation"] = 0.0

    total = sum(stage_latencies.values())

    return PipelineResult(
        query_text=query_text,
        transcription=transcription,
        retrieval=retrieval,
        generation=generation,
        total_latency_ms=total,
        stage_latencies_ms=stage_latencies,
    )


if __name__ == "__main__":
    # Configure stdout to handle UTF-8 printing on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
        
    # quick manual smoke test -- replace with a real audio file path
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_queries/hi_01.wav"
    with open(path, "rb") as f:
        result = run_pipeline(f.read())
    print(result)
