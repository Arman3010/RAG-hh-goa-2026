"""
Person C owns this. Run against a batch of test audio queries (or text
queries if you bypass STT for a retrieval+generation-only pass) and report
P50/P70/P100 latency for total pipeline AND each individual stage --
required by submission item #4. Don't cherry-pick a best-case run: use at
least ~20-30 varied queries.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from stt_harness.harness import run_pipeline


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def run_benchmark(audio_paths: list[str]):
    total_latencies, stt_latencies, retrieval_latencies, generation_latencies = [], [], [], []

    for path in audio_paths:
        with open(path, "rb") as f:
            result = run_pipeline(f.read())
        total_latencies.append(result.total_latency_ms)
        stt_latencies.append(result.stage_latencies_ms.get("stt", 0))
        retrieval_latencies.append(result.stage_latencies_ms.get("retrieval", 0))
        generation_latencies.append(result.stage_latencies_ms.get("generation", 0))

    report = {}
    for name, values in [
        ("total", total_latencies),
        ("stt", stt_latencies),
        ("retrieval", retrieval_latencies),
        ("generation", generation_latencies),
    ]:
        report[name] = {
            "P50": round(percentile(values, 50), 2),
            "P70": round(percentile(values, 70), 2),
            "P100": round(percentile(values, 100), 2),
        }

    return report


if __name__ == "__main__":
    # TODO: point at a directory of test audio files, e.g. glob("../data/test_queries/*.wav")
    test_files = []
    report = run_benchmark(test_files)
    import json
    print(json.dumps(report, indent=2))
