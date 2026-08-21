"""
Person A: transcribe() implemented against shared/interfaces.py's contract.

Provider: Sarvam (saaras:v3, mode="transcribe") — chosen because MSMARCO-XI
is Hindi/Indic-adapted and Sarvam is purpose-built for Indian languages
(code-mixed Hindi/English included), whereas ElevenLabs STT is optimized
for English/European languages.

Set SARVAM_API_KEY in your environment (or a .env file) before running.
Get a key from https://dashboard.sarvam.ai/

pip install sarvamai   (official SDK, see requirements.txt)
"""

import sys, os, io, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.interfaces import TranscriptionResult

from dotenv import load_dotenv
load_dotenv()

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")


def _transcribe_sarvam(audio_bytes: bytes, language_code: str = "unknown") -> TranscriptionResult:
    """
    Calls Sarvam's /speech-to-text REST endpoint via the official SDK.
    model=saaras:v3, mode=transcribe -> normalized text in the spoken language
    (Hindi stays Devanagari, English stays English, code-mixed input handled natively).
    """
    from sarvamai import SarvamAI

    t0 = time.perf_counter()
    try:
        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

        # SDK accepts a file-like object; wrap the raw bytes as a .wav buffer.
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "query.wav"  # SDK/requests uses this for the multipart filename

        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="transcribe",
            language_code=language_code,   # "unknown" lets Sarvam auto-detect hi-IN/en-IN
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(
            text=(response.transcript or "").strip(),
            language=getattr(response, "language_code", None),
            confidence=getattr(response, "language_probability", None),
            latency_ms=latency_ms,
            raw_provider_response={"request_id": getattr(response, "request_id", None)},
        )
    except Exception as e:
        # Degrade gracefully: bad audio / network / auth errors must NOT crash the
        # pipeline. Return empty text and let the harness decide (it treats
        # empty text as "empty_transcription" and stops before retrieval).
        latency_ms = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(
            text="",
            latency_ms=latency_ms,
            raw_provider_response={"error": str(e)},
        )


def _transcribe_elevenlabs(audio_bytes: bytes) -> TranscriptionResult:
    """Fallback / alternative provider, kept for reference — not used by default."""
    import requests

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            files={"file": ("query.wav", audio_bytes, "audio/wav")},
            data={"model_id": "scribe_v1"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(
            text=(data.get("text") or "").strip(),
            language=data.get("language_code"),
            latency_ms=latency_ms,
            raw_provider_response=data,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(text="", latency_ms=latency_ms,
                                    raw_provider_response={"error": str(e)})


def transcribe(audio_bytes: bytes, provider: str = "sarvam") -> TranscriptionResult:
    """
    Entry point matching shared/interfaces.py. Always returns a TranscriptionResult
    (never raises) so the harness's retry loop only fires on genuinely transient
    issues, not on every bad/silent audio clip.
    """
    if not audio_bytes:
        return TranscriptionResult(text="", latency_ms=0.0,
                                    raw_provider_response={"error": "empty audio_bytes"})

    if provider == "elevenlabs":
        return _transcribe_elevenlabs(audio_bytes)
    return _transcribe_sarvam(audio_bytes)


if __name__ == "__main__":
    # Configure stdout to handle UTF-8 printing on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
        
    # quick standalone check: python stt_harness/stt.py ../data/sample_query.wav
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/sample_query.wav"
    with open(path, "rb") as f:
        result = transcribe(f.read())
    print(result)
