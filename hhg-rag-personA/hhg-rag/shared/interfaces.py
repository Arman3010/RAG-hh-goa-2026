"""
Shared interfaces for the HH Goa 2026 Voice-RAG pipeline.

Every team member builds against THESE signatures so the three pieces
snap together on integration day without renegotiating formats.

Pipeline shape:
  audio_bytes -> [Person A: transcribe] -> query_text
  query_text  -> [Person B: retrieve]   -> List[RetrievedChunk]
  query_text + chunks -> [Person C: generate] -> GenerationResult
"""

from dataclasses import dataclass, field
from typing import Optional
import time


# ---------------------------------------------------------------------------
# Person A owns: transcribe()
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None      # e.g. "hi", "en"
    confidence: Optional[float] = None  # 0-1 if provider gives one
    latency_ms: float = 0.0
    raw_provider_response: Optional[dict] = None


def transcribe(audio_bytes: bytes, provider: str = "sarvam") -> TranscriptionResult:
    """
    Person A implements this.
    Input: raw audio bytes (wav/mp3).
    Output: TranscriptionResult. Must always set .text and .latency_ms,
    even on partial/degraded results (empty string text is fine, must
    not raise on bad audio -- return text="" and let the harness decide).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Person B owns: retrieve()
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float                      # similarity/relevance score
    source_doc_id: Optional[str] = None
    chunk_strategy: Optional[str] = None   # e.g. "fixed_512_overlap64", "semantic", "metadata_aware"
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float = 0.0
    strategy_used: Optional[str] = None


def retrieve(query: str, top_k: int = 5, strategy: str = "hybrid") -> RetrievalResult:
    """
    Person B implements this.
    Input: query text (already transcribed).
    Output: RetrievalResult with ranked chunks, best-scoring first.
    Must return an empty chunks list (not raise) if nothing relevant is found --
    that's a valid signal for the guardrail layer ("no grounding available").
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Person C owns: generate() and the guardrail checks
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    answer: str
    grounded: bool                       # did the answer check out against retrieved context?
    refused: bool = False                # True if guardrails blocked an answer
    refusal_reason: Optional[str] = None # e.g. "off_topic", "no_context", "unsafe_input"
    latency_ms: float = 0.0
    chunks_used: list[str] = field(default_factory=list)  # chunk_ids actually cited


def generate(query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    """
    Person C implements this.
    Input: query + retrieved chunks (may be empty list).
    Output: GenerationResult. Must apply guardrails BEFORE calling the LLM
    where possible (cheap checks first: empty context, off-topic query)
    and groundedness check AFTER (does answer content trace back to chunks).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Full pipeline result -- what the harness assembles and what gets logged
# for the latency report (P50/P70/P100).
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    query_text: str
    transcription: Optional[TranscriptionResult]
    retrieval: RetrievalResult
    generation: GenerationResult
    total_latency_ms: float = 0.0
    stage_latencies_ms: dict = field(default_factory=dict)  # {"stt": .., "retrieval": .., "generation": ..}
    error: Optional[str] = None


class Timer:
    """Small helper so all three of you measure latency the same way."""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000
