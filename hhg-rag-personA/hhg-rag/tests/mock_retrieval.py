"""
Fake retrieve() so you can build and test generation_guardrails/ end-to-end
before Person B's real retriever.py is ready. Matches the exact
RetrievalResult/RetrievedChunk shape from shared/interfaces.py, so swapping
this out for the real thing later is a one-line change.

Usage in your own test scripts:
    from tests.mock_retrieval import mock_retrieve as retrieve
"""

import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.interfaces import RetrievalResult, RetrievedChunk

# A few fake "documents" loosely themed around general-knowledge queries,
# so overlap-based groundedness checks have something real to match against.
_FAKE_DOCS = {
    "india_capital": "New Delhi is the capital of India. It was officially "
                      "declared the capital in 1911, replacing Kolkata (Calcutta).",
    "ml_basics": "Machine learning is a subset of artificial intelligence where "
                 "systems learn patterns from data rather than following "
                 "explicitly programmed rules.",
    "photosynthesis": "Photosynthesis is the process by which plants convert "
                       "sunlight, water, and carbon dioxide into glucose and oxygen.",
}


def mock_retrieve(query: str, top_k: int = 5, strategy: str = "hybrid") -> RetrievalResult:
    t0 = time.perf_counter()
    q_lower = query.lower()

    chunks = []
    if "capital" in q_lower or "india" in q_lower or "राजधानी" in query:
        chunks.append(RetrievedChunk(chunk_id="mock_001", text=_FAKE_DOCS["india_capital"],
                                      score=0.91, source_doc_id="doc_1", chunk_strategy="mock"))
    if "machine learning" in q_lower or "artificial" in q_lower or "बुद्धिमत्ता" in query:
        chunks.append(RetrievedChunk(chunk_id="mock_002", text=_FAKE_DOCS["ml_basics"],
                                      score=0.88, source_doc_id="doc_2", chunk_strategy="mock"))
    if "photosynthesis" in q_lower or "plant" in q_lower:
        chunks.append(RetrievedChunk(chunk_id="mock_003", text=_FAKE_DOCS["photosynthesis"],
                                      score=0.85, source_doc_id="doc_3", chunk_strategy="mock"))

    # Anything that doesn't match a fake doc returns empty -- this is what
    # exercises your "no_context" guardrail path.
    latency_ms = (time.perf_counter() - t0) * 1000
    return RetrievalResult(query=query, chunks=chunks[:top_k], latency_ms=latency_ms,
                            strategy_used="mock")


if __name__ == "__main__":
    for q in ["What is the capital of India?", "What's the weather today?"]:
        print(q, "->", mock_retrieve(q))
