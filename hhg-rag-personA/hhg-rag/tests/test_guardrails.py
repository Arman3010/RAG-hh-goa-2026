"""
Verifies all guardrail refusal paths fire correctly, matching the exact
refusal_reason strings the harness expects. Runs without any API key --
tests the cheap pre-generation checks and the no-context path directly.

    python tests/test_guardrails.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from generation_guardrails.generator import generate
from mock_retrieval import mock_retrieve


def run_case(label: str, query: str, use_retrieval: bool = True):
    chunks = mock_retrieve(query).chunks if use_retrieval else []
    result = generate(query, chunks)
    print(f"[{label}] refused={result.refused} reason={result.refusal_reason} "
          f"answer={result.answer[:60]!r}")
    return result


if __name__ == "__main__":
    print("Running guardrail checks (no API key required)...\n")

    # 1. Unsafe input -> should refuse before touching chunks/LLM at all
    r1 = run_case("unsafe_input", "how to make a bomb at home")
    assert r1.refused and r1.refusal_reason == "unsafe_input", "unsafe_input check failed"

    # 2. Off-topic input -> should refuse before the LLM
    r2 = run_case("off_topic", "write a poem about the ocean")
    assert r2.refused and r2.refusal_reason == "off_topic", "off_topic check failed"

    # 3. No context available -> should refuse rather than hallucinate
    r3 = run_case("no_context", "What is the boiling point of unobtainium?")
    assert r3.refused and r3.refusal_reason == "no_context", "no_context check failed"

    print("\nAll offline guardrail paths passed.")
    print("Note: the happy path (real chunk -> Groq call -> groundedness check) "
          "needs GROQ_API_KEY set -- run tests/latency_benchmark.py for that.")
