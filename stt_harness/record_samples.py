"""
Person A: run this LOCALLY (needs a microphone) to record sample queries
for the team to test the pipeline against.

pip install sounddevice scipy

Usage:
    python record_samples.py                # interactive, records into ../data/
    python record_samples.py --seconds 6     # custom clip length
"""

import argparse
import os
import sys

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000  # Sarvam works best at 16kHz mono
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_queries")

# Suggested prompts spanning Hindi, English, and code-mixed -- matches the
# kind of queries MSMARCO-XI / the demo will need.
SUGGESTED_PROMPTS = [
    ("hi_01", "भारत की राजधानी क्या है?"),
    ("hi_02", "कृत्रिम बुद्धिमत्ता क्या होती है?"),
    ("en_01", "What is the capital of India?"),
    ("en_02", "Explain machine learning in simple terms."),
    ("mix_01", "Mujhe machine learning ke baare mein bataiye"),
    ("noisy_01", "(record with background noise / mumbling to test degraded STT)"),
]


def record_clip(filename: str, seconds: float):
    print(f"Recording '{filename}' for {seconds}s... speak now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{filename}.wav")
    write(path, SAMPLE_RATE, audio)
    print(f"Saved -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    print("Sample prompts (say each one when prompted, or Ctrl+C to stop early):\n")
    for name, prompt in SUGGESTED_PROMPTS:
        print(f"  [{name}] {prompt}")
    print()

    for name, prompt in SUGGESTED_PROMPTS:
        input(f"Press Enter, then say: \"{prompt}\"")
        record_clip(name, args.seconds)

    print("\nDone. These .wav files are what harness.py and latency_benchmark.py "
          "should read from data/sample_queries/.")


if __name__ == "__main__":
    main()
