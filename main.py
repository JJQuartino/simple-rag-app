"""
Stage 2 GUI: pick PDFs, extract + chunk + embed them, then ask questions
and see which chunks get retrieved. (Answer generation via LLM comes in Stage 3.)
"""

import os
import shutil
import tkinter as tk
from tkinter import filedialog, scrolledtext

from pdf_loader import load_pdfs
from chunker import chunk_all_documents
from embedder import embed_chunks
from retriever import retrieve

DOCS_DIR = "documents"
os.makedirs(DOCS_DIR, exist_ok=True)

loaded_chunks = []  # each chunk will hold text, source, chunk_index, embedding


def add_pdfs():
    filepaths = filedialog.askopenfilenames(
        title="Select PDF files",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not filepaths:
        return

    copied_paths = []
    for src in filepaths:
        dest = os.path.join(DOCS_DIR, os.path.basename(src))
        shutil.copy(src, dest)
        copied_paths.append(dest)

    log(f"Added {len(copied_paths)} file(s). Processing...")

    documents = load_pdfs(copied_paths)
    chunks = chunk_all_documents(documents)

    log("Embedding chunks (first run downloads the model, please wait)...")
    embed_chunks(chunks)  # adds 'embedding' to each chunk, in place

    loaded_chunks.extend(chunks)

    for name, text in documents.items():
        if text is None:
            log(f"  ✗ Failed to read: {name}")
        else:
            n_chunks = sum(1 for c in chunks if c["source"] == name)
            log(f"  ✓ {name} — {len(text)} characters, {n_chunks} chunks")

    log(f"\nTotal chunks ready for search: {len(loaded_chunks)}")


def ask_question():
    query = question_entry.get().strip()
    if not query:
        return

    log(f"\n> {query}")

    if not loaded_chunks:
        log("  (No documents loaded yet — add PDFs first.)")
        return

    results = retrieve(query, loaded_chunks, top_k=3)

    if not results:
        log("  I don't have the information to answer that.")
        return

    for r in results:
        log(f"  [{r['similarity']:.2f}] ({r['source']}) {r['text'][:200]}...")


def log(message):
    output.insert(tk.END, message + "\n")
    output.see(tk.END)


root = tk.Tk()
root.title("Document Q&A — Stage 2 (retrieval only, no LLM yet)")
root.geometry("700x500")

add_button = tk.Button(root, text="Add PDF(s)", command=add_pdfs)
add_button.pack(pady=10)

output = scrolledtext.ScrolledText(root, wrap=tk.WORD)
output.pack(expand=True, fill="both", padx=10, pady=(0, 10))

query_frame = tk.Frame(root)
query_frame.pack(fill="x", padx=10, pady=(0, 10))

question_entry = tk.Entry(query_frame)
question_entry.pack(side="left", expand=True, fill="x", padx=(0, 5))
question_entry.bind("<Return>", lambda event: ask_question())

ask_button = tk.Button(query_frame, text="Ask", command=ask_question)
ask_button.pack(side="right")

root.mainloop()