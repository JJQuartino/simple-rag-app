"""
Stage 1a: Load text out of PDF files.
"""

import os
from pypdf import PdfReader


def load_pdf_text(filepath: str) -> str:
    """Extract all text from a single PDF file, page by page."""
    reader = PdfReader(filepath)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return "\n".join(pages_text)


def load_pdfs(filepaths: list) -> dict:
    """
    Load a specific list of PDF files (as chosen via the GUI).
    Returns a dict: { "filename.pdf": "full extracted text", ... }
    """
    documents = {}

    for full_path in filepaths:
        filename = os.path.basename(full_path)
        try:
            text = load_pdf_text(full_path)
            documents[filename] = text
        except Exception as e:
            documents[filename] = None
            print(f"FAILED to load {filename}: {e}")

    return documents