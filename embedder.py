"""
Stage 2a: Turn text into embeddings (vectors that capture meaning).
"""

from sentence_transformers import SentenceTransformer

# Multilingual model since our documents are in Spanish (also handles English fine)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None


def get_model():
    """Load the embedding model once and reuse it (loading is slow, embedding is fast)."""
    global _model
    if _model is None:
        print("Loading embedding model (first time only, downloads ~470MB)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: list) -> list:
    """
    Given a list of chunk dicts (from chunker.py), add an 'embedding' key to each.
    Returns the same list, mutated in place, for convenience.
    """
    model = get_model()
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


def embed_query(query: str):
    """Embed a single question the same way chunks were embedded."""
    model = get_model()
    return model.encode(query)