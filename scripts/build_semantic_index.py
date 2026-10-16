"""
Semantic Index and Author Profile Building Script.
Encodes rich structured semantic documents into 384-dimensional normalized dense vectors
using Sentence-BERT (all-MiniLM-L6-v2), builds a FAISS IndexFlatIP vector index,
and computes normalized Author Profile Embeddings.
Usage:
    python scripts/build_semantic_index.py
"""
import pickle
import time
from pathlib import Path
import sys
import numpy as np
import faiss

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import MODEL_DIR
from recommender.semantic import (
    create_sentence_model,
    generate_embeddings,
    build_faiss_index,
    compute_author_profile_embeddings,
)


def main():
    print("=" * 60)
    print("Semantic Index & Author Profile Construction Pipeline")
    print("=" * 60)
    start_time = time.time()

    processed_books_path = MODEL_DIR / "processed_books.pkl"
    if not processed_books_path.exists():
        print("Error: processed_books.pkl not found! Please run scripts/prepare_data.py first.")
        sys.exit(1)

    print("[1/5] Loading processed books catalog...")
    with open(processed_books_path, "rb") as f:
        books_df = pickle.load(f)
    print(f"      Total books to index: {len(books_df):,}")

    print("[2/5] Initializing SentenceTransformer ('all-MiniLM-L6-v2')...")
    model = create_sentence_model()

    print("[3/5] Generating 384-dimensional normalized semantic embeddings from structured documents...")
    # Use the rich structured Semantic_Document for each book
    semantic_docs = books_df["Semantic_Document"].tolist() if "Semantic_Document" in books_df.columns else books_df["Semantic_Text"].tolist()
    embeddings = generate_embeddings(semantic_docs, model=model, batch_size=128, show_progress=True)
    print(f"      Embeddings matrix shape: {embeddings.shape}")

    print("[4/5] Building and persisting FAISS IndexFlatIP vector index...")
    faiss_index = build_faiss_index(embeddings)
    print(f"      FAISS index total vectors: {faiss_index.ntotal}")

    print("[5/5] Computing normalized Author Profile Embeddings...")
    author_profiles = compute_author_profile_embeddings(books_df, embeddings)
    print(f"      Computed profiles for {len(author_profiles):,} distinct authors.")

    # Save embeddings, FAISS index, and author profiles
    np.save(MODEL_DIR / "book_embeddings.npy", embeddings)
    faiss.write_index(faiss_index, str(MODEL_DIR / "faiss_index.bin"))

    with open(MODEL_DIR / "author_profiles.pkl", "wb") as f:
        pickle.dump(author_profiles, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - start_time
    print(f"Semantic index and author profiles successfully saved in {elapsed:.2f} seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
