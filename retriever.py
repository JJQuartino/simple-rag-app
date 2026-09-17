"""
Stage 2b: Given a question, find the most relevant chunks.
"""

import numpy as np
from embedder import embed_query


def cosine_similarity(a, b) -> float:
    """How similar two vectors are, from -1 (opposite) to 1 (identical direction)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query: str, chunks: list, top_k: int = 3, min_similarity: float = 0.3) -> list:
    """
    Find the top_k chunks most relevant to the query.

    min_similarity: chunks below this score are considered irrelevant.
    This is what lets the system say "I don't have that information"
    instead of forcing an answer from unrelated text.
    """
    if not chunks:
        return []

    query_vec = embed_query(query)

    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_vec, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = [
        {**chunk, "similarity": score}
        for score, chunk in scored[:top_k]
        if score >= min_similarity
    ]

    return results