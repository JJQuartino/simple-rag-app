"""
Stage 1b: Split extracted text into smaller, overlapping chunks.
"""


def chunk_text(text: str, source_name: str, chunk_size: int = 500, overlap: int = 50) -> list:
    cleaned = " ".join(text.split())

    if not cleaned:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(cleaned):
        end = start + chunk_size
        chunk_str = cleaned[start:end]

        chunks.append({
            "text": chunk_str,
            "source": source_name,
            "chunk_index": chunk_index,
        })

        chunk_index += 1
        start += chunk_size - overlap

    return chunks


def chunk_all_documents(documents: dict, chunk_size: int = 500, overlap: int = 50) -> list:
    all_chunks = []
    for source_name, text in documents.items():
        if text is None:
            continue
        doc_chunks = chunk_text(text, source_name, chunk_size, overlap)
        all_chunks.extend(doc_chunks)

    return all_chunks