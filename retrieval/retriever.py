"""
Person B: Retrieval module.
Final strategy: metadata-aware chunking (best Recall@3 in testing: 38.21%
vs 37.21% baseline, 32.23% fixed-size, 22.92% semantic — splitting further
hurt recall since source passages are already short).
"""

import sys, os, time, json, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from shared.interfaces import RetrievalResult, RetrievedChunk

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SUBSET_PATH = os.path.join(DATA_DIR, "subset.json")

_model = None
_index = None
_chunked_corpus = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def chunk_metadata_aware(subset: list[dict]) -> list[dict]:
    chunked = []
    for entry in subset:
        query_id = entry["query_id"]
        query_type = entry["query_type"]
        for i, passage_text in enumerate(entry["translated_passages"]):
            chunked.append({
                "chunk_id": f"{query_id}_meta{i}",
                "text": f"[{query_type}] {passage_text}",
                "source_query_id": query_id,
                "is_selected": entry["is_selected"][i],
                "chunk_strategy": "metadata_aware",
                "query_type": query_type,
            })
    return chunked


def build_index():
    """Loads subset.json, chunks it, embeds it, builds the FAISS index.
    Called once at startup (or lazily on first retrieve() call)."""
    global _index, _chunked_corpus

    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        subset = json.load(f)

    _chunked_corpus = chunk_metadata_aware(subset)

    model = _load_model()
    texts = [c["text"] for c in _chunked_corpus]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)

    print(f"Index built: {len(_chunked_corpus)} chunks")


def retrieve(query: str, top_k: int = 5, strategy: str = "metadata_aware") -> RetrievalResult:
    global _index, _chunked_corpus

    t0 = time.perf_counter()

    if _index is None:
        build_index()

    model = _load_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    scores, indices = _index.search(q_emb, top_k)

    chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        c = _chunked_corpus[idx]
        chunks.append(RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            score=float(score),
            source_doc_id=str(c["source_query_id"]),
            chunk_strategy=c["chunk_strategy"],
            metadata={"query_type": c.get("query_type", "")},
        ))

    latency_ms = (time.perf_counter() - t0) * 1000
    return RetrievalResult(query=query, chunks=chunks, latency_ms=latency_ms, strategy_used=strategy)


if __name__ == "__main__":
    # quick manual smoke test
    build_index()
    result = retrieve("मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?", top_k=3)
    for c in result.chunks:
        print(f"{c.score:.4f} | {c.text[:80]}...")
    print(f"Latency: {result.latency_ms:.2f}ms")
