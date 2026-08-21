"""
Person B: implement retrieve() here against shared/interfaces.py's contract.

Requirement #2 says chunking must be "vast" -- plan to implement at least:
  1. fixed_size_overlap   -- baseline, e.g. 512 tokens, 64 overlap
  2. semantic             -- split on sentence/paragraph embeddings similarity
  3. metadata_aware        -- chunk boundaries respect doc structure/fields

Build each as its own indexed collection (or tag chunks with chunk_strategy)
so you can compare retrieval quality across strategies in your writeup --
that comparison itself is a good thing to show off in the demo video.

Suggested stack: sentence-transformers for embeddings + FAISS for the index
(both free, both fast enough to comfortably clear the <200ms retrieval leg).
"""

import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.interfaces import RetrievalResult, RetrievedChunk


def chunk_fixed_size(documents: list[dict], chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """TODO: naive fixed-size chunking with overlap. Baseline strategy."""
    raise NotImplementedError


def chunk_semantic(documents: list[dict]) -> list[dict]:
    """TODO: split on semantic/topic shifts rather than fixed token counts."""
    raise NotImplementedError


def chunk_metadata_aware(documents: list[dict]) -> list[dict]:
    """TODO: respect document structure (titles, sections, MSMARCO passage/query fields)."""
    raise NotImplementedError


def build_index(strategy: str = "hybrid"):
    """TODO: embed chunks and build/load the FAISS (or Chroma/Qdrant) index."""
    raise NotImplementedError


def retrieve(query: str, top_k: int = 5, strategy: str = "hybrid") -> RetrievalResult:
    t0 = time.perf_counter()

    # TODO: embed query, search index(es), merge/rank results across
    # strategies if using more than one at query time.
    chunks: list[RetrievedChunk] = []

    latency_ms = (time.perf_counter() - t0) * 1000
    return RetrievalResult(query=query, chunks=chunks, latency_ms=latency_ms, strategy_used=strategy)
