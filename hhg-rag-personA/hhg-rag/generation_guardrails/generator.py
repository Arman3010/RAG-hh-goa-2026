"""
Person C: implement generate() here against shared/interfaces.py's contract.

Requirement #6 (guardrails) -- apply checks in this order:
  1. Off-topic / unsafe input check on the query BEFORE calling the LLM (cheap, fast)
  2. Empty-context check -- if chunks == [], refuse rather than hallucinate
  3. Call the LLM with retrieved context, instructed to answer ONLY from context
  4. Groundedness check AFTER generation -- does the answer's content actually
     trace back to the retrieved chunks? (simple approach: check answer
     sentences have high overlap/similarity with at least one chunk;
     stronger approach: a second LLM call asking "is this answer supported
     by this context, yes/no")

Also owns: latency_benchmark.py (P50/P70/P100 report) and the live deploy.
"""

import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.interfaces import GenerationResult, RetrievedChunk
from generation_guardrails.guardrails import is_off_topic_or_unsafe, check_groundedness
from generation_guardrails.llm_client import call_groq


def call_llm(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    The actual generation call, context-restricted prompt.
    Delegates to Groq (see llm_client.py). Kept as a thin wrapper so this
    file's shape stays close to the shared contract and the provider can be
    swapped later behind this one function.
    """
    context_blocks = [c.text for c in chunks]
    return call_groq(query, context_blocks)


def generate(query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    t0 = time.perf_counter()

    blocked, reason = is_off_topic_or_unsafe(query)
    if blocked:
        return GenerationResult(answer="", grounded=False, refused=True,
                                 refusal_reason=reason,
                                 latency_ms=(time.perf_counter() - t0) * 1000)

    if not chunks:
        return GenerationResult(answer="", grounded=False, refused=True,
                                 refusal_reason="no_context",
                                 latency_ms=(time.perf_counter() - t0) * 1000)

    try:
        answer = call_llm(query, chunks)
    except Exception:
        # Network/auth/rate-limit failures must degrade gracefully, not
        # crash the harness -- matches Person A's retry/degrade contract.
        return GenerationResult(answer="", grounded=False, refused=True,
                                 refusal_reason="generation_failed",
                                 latency_ms=(time.perf_counter() - t0) * 1000)

    grounded = check_groundedness(answer, chunks, query=query)

    latency_ms = (time.perf_counter() - t0) * 1000
    return GenerationResult(
        answer=answer,
        grounded=grounded,
        refused=not grounded,
        refusal_reason=None if grounded else "not_grounded",
        latency_ms=latency_ms,
        chunks_used=[c.chunk_id for c in chunks],
    )
