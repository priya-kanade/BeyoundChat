# src/llm_evaluator/embeddings.py
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _MODEL = None

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 256), dtype=float)
    if _MODEL is not None:
        return _MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    vecs = []
    for t in texts:
        arr = np.zeros(256, dtype=float)
        for i, ch in enumerate(t[:2000]):
            arr[i % 256] += ord(ch)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        vecs.append(arr)
    return np.vstack(vecs)

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0] if a.ndim>1 else 1, b.shape[0] if b.ndim>1 else 1))
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return a.dot(b.T)
