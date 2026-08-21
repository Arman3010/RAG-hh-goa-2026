"""
Guardrail checks, kept separate from generator.py so each can be unit
tested and tuned independently.

Two layers, matching Requirement #6:
  1. is_off_topic_or_unsafe() -- pre-gen, cheap, no LLM call needed
  2. check_groundedness()     -- post-gen, verifies the answer traces back
     to the retrieved chunks rather than being hallucinated
"""

import os
import re

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.interfaces import RetrievedChunk

# --- Layer 1: pre-generation query screening -------------------------------

# Keep this short and obvious-case-only; over-blocking legitimate questions
# is its own failure mode. This is a first line of defense, not a full
# content-moderation system.
UNSAFE_PATTERNS = [
    r"\bhow (to|do i) (make|build|synthesize)\b.*\b(bomb|explosive|weapon|virus|malware)\b",
    r"\bkill (myself|someone|a person)\b",
    r"\bhack (into|someone'?s)\b",
    r"\bchild\s*(sexual|porn|abuse)\b",
]

# The pipeline is grounded in MSMARCO-XI (general knowledge/search-style
# queries) -- so "off-topic" here means clearly unrelated to
# information-seeking questions, e.g. requests to write poems, code, or
# carry out actions rather than asking something answerable from a
# knowledge-base context.
OFF_TOPIC_PATTERNS = [
    r"^\s*(write|compose)\s+(a|an)\s+(poem|song|story|essay)\b",
    r"^\s*(act as|pretend to be|roleplay as)\b",
    r"^\s*(write|generate)\s+code\b",
]


def is_off_topic_or_unsafe(query: str) -> tuple[bool, str]:
    """
    Returns (True, "unsafe_input") or (True, "off_topic") to block before
    hitting the LLM at all, or (False, "") to proceed.
    """
    if not query or not query.strip():
        return True, "off_topic"

    q_lower = query.lower().strip()

    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, q_lower):
            return True, "unsafe_input"

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, q_lower):
            return True, "off_topic"

    return False, ""


# --- Layer 2: post-generation groundedness check ----------------------------

GROUNDEDNESS_MODE = os.environ.get("GROUNDEDNESS_MODE", "overlap")  # "overlap" | "llm_judge"
OVERLAP_THRESHOLD = float(os.environ.get("GROUNDEDNESS_OVERLAP_THRESHOLD", "0.25"))

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of",
    "and", "or", "for", "with", "as", "by", "it", "this", "that", "these",
    "those", "be", "has", "have", "had", "not", "but", "from", "which", "who",
    "what", "when", "where", "how", "why", "does", "do", "did", "can", "will",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z\u0900-\u097F]+", text.lower())  # latin + devanagari
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _overlap_groundedness(answer: str, chunks: list[RetrievedChunk]) -> bool:
    """
    Free, no-API heuristic: does a meaningful fraction of the answer's
    content words appear in at least one retrieved chunk? Cheap and fast
    enough to run on every single generation, unlike an LLM-judge call.
    """
    if not answer.strip() or not chunks:
        return False

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return False

    best_overlap = 0.0
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        if not chunk_tokens:
            continue
        overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)
        best_overlap = max(best_overlap, overlap)

    return best_overlap >= OVERLAP_THRESHOLD


def check_groundedness(answer: str, chunks: list[RetrievedChunk], query: str = "") -> bool:
    """
    Verify the answer is actually supported by the retrieved chunks.
    Defaults to the free word-overlap heuristic; set GROUNDEDNESS_MODE=llm_judge
    in .env to use a second Groq call instead (slower, costs quota, but catches
    paraphrased hallucinations the overlap check would miss).
    """
    if GROUNDEDNESS_MODE == "llm_judge":
        from generation_guardrails.llm_client import call_groq_judge
        try:
            return call_groq_judge(query, answer, [c.text for c in chunks])
        except Exception:
            # If the judge call itself fails, fall back to the free heuristic
            # rather than letting an ungrounded answer through by default.
            return _overlap_groundedness(answer, chunks)

    return _overlap_groundedness(answer, chunks)
