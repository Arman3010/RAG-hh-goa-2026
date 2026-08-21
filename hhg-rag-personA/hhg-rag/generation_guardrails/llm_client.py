"""
Thin wrapper around the Groq API (OpenAI-compatible chat completions endpoint).
Groq is used because it's free-tier friendly and by far the fastest inference
available for a hackathon (sub-second generation), which matters since your
generation stage is where most of the total pipeline latency will live once
STT and retrieval are accounted for.

Get a free key at https://console.groq.com -> put it in .env as GROQ_API_KEY.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You are a strict, grounded question-answering assistant. "
    "You must answer ONLY using the information present in the provided context. "
    "If the context does not contain enough information to answer confidently, "
    "say so explicitly instead of guessing or using outside knowledge. "
    "Keep answers concise (2-4 sentences) and do not fabricate facts, numbers, "
    "or sources that are not present in the context."
)


def call_groq(query: str, context_blocks: list[str], timeout: float = 15.0) -> str:
    """
    Calls Groq's chat completions endpoint with a context-restricted prompt.
    Raises on network/auth failure -- the caller (generate()) is responsible
    for catching this and returning a graceful GenerationResult instead of
    crashing the pipeline.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is empty. Set it in .env (GROQ_API_KEY=your_key) "
            "or `export GROQ_API_KEY=...` before running."
        )

    import requests

    context_text = "\n\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(context_blocks))
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_groq_judge(question: str, answer: str, context_blocks: list[str], timeout: float = 15.0) -> bool:
    """
    Optional stronger groundedness check: asks the LLM itself whether the
    answer is actually supported by the context, yes/no. Slower and costs
    an extra call, so only used when GROUNDEDNESS_MODE=llm_judge (see
    guardrails.py) -- default is the free word-overlap heuristic.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is empty.")

    import requests

    context_text = "\n\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(context_blocks))
    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        "Is the proposed answer fully supported by the context above, with no "
        "invented facts? Reply with exactly one word: YES or NO."
    )

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 5,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    verdict = resp.json()["choices"][0]["message"]["content"].strip().upper()
    return verdict.startswith("YES")
