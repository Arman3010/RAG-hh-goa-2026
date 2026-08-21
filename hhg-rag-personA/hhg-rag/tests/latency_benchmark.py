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
    import glob
    import json
    import datetime

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_queries")
    test_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))

    if not test_files:
        print(f"No .wav files found in {data_dir}. "
              "Ask Person A for their recorded sample_queries, or record your own "
              "with stt_harness/record_samples.py before running this.")
        sys.exit(1)

    print(f"Running benchmark against {len(test_files)} audio files...")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")

    report = run_benchmark(test_files)
    report["_meta"] = {
        "num_queries": len(test_files),
        "generated_at": datetime.datetime.now().isoformat(),
        "note": "The brief's <200ms target applies to the retrieval leg "
                "(chunking + vector search), not STT+LLM generation over a "
                "network call -- see stage-wise breakdown below.",
    }

    print("\n" + json.dumps(report, indent=2))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "latency_report")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "latency_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(out_dir, "latency_report.md"), "w", encoding="utf-8") as f:
        f.write("# Latency Report\n\n")
        f.write(f"Queries tested: {len(test_files)}\n\n")
        f.write("| Stage | P50 (ms) | P70 (ms) | P100 (ms) |\n")
        f.write("|---|---|---|---|\n")
        for stage in ["total", "stt", "retrieval", "generation"]:
            s = report[stage]
            f.write(f"| {stage} | {s['P50']} | {s['P70']} | {s['P100']} |\n")
        f.write("\nNote: the <200ms target applies to the retrieval leg "
                "(chunking + vector search), not STT+LLM generation over a network call.\n")

    print(f"\nSaved report to {out_dir}/latency_report.json and .md")
