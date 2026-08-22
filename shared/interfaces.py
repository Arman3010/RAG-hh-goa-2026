"""
Shared interfaces for the HH Goa 2026 Voice-RAG pipeline.

Every team member builds against THESE signatures so the three pieces
snap together on integration day without renegotiating formats.

Pipeline shape:
  audio_bytes -> [Person A: transcribe] -> query_text
  query_text  -> [Person B: retrieve]   -> List[RetrievedChunk]
  query_text + chunks -> [Person C: generate] -> GenerationResult
"""

"""
Shared interfaces for the HH Goa 2026 Voice-RAG pipeline.
Everyone codes against these signatures so the three pieces integrate cleanly.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    raw_provider_response: Optional[dict] = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_doc_id: Optional[str] = None
    chunk_strategy: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float = 0.0
    strategy_used: Optional[str] = None


@dataclass
class GenerationResult:
    answer: str
    grounded: bool
    refused: bool = False
    refusal_reason: Optional[str] = None
    latency_ms: float = 0.0
    chunks_used: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    query_text: str
    transcription: Optional[TranscriptionResult]
    retrieval: RetrievalResult
    generation: GenerationResult
    total_latency_ms: float = 0.0
    stage_latencies_ms: dict = field(default_factory=dict)
    error: Optional[str] = None


class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000
