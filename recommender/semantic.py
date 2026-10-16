"""
Semantic representation, FAISS vector retrieval, and Author Profile Embeddings.
Uses Sentence-BERT (all-MiniLM-L6-v2) to generate 384-d normalized embeddings
from structured semantic documents.
"""
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM


def create_sentence_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Instantiate the SentenceTransformer model."""
    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: List[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate 384-d dense embeddings for semantic documents and L2-normalize them.
    """
    if model is None:
        model = create_sentence_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization ensures inner product = cosine similarity
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP index with normalized vectors.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def compute_author_profile_embeddings(
    books_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs Author Profile Embeddings for every unique author in the catalog.
    Aggregates the semantic book embeddings written by the author, computes the mean vector,
    and L2-normalizes the profile.

    Returns:
        dict mapping normalized_author_name -> {
            'author': original_author_name,
            'embedding': 384-d normalized numpy vector,
            'book_count': int,
            'book_indices': List[int]
        }
    """
    author_profiles: Dict[str, Dict[str, Any]] = {}
    grouped = books_df.groupby("Author")

    for author_name, group in grouped:
        author_clean = str(author_name).strip()
        if not author_clean or author_clean.lower() in ["unknown author", "various", "anonymous", ""]:
            continue

        book_indices = group.index.tolist()
        if not book_indices:
            continue

        # Extract embeddings for all books by this author
        author_book_vecs = embeddings[book_indices]

        # Compute mean embedding
        mean_vec = np.mean(author_book_vecs, axis=0)

        # L2 Normalize
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-6:
            author_vec = (mean_vec / norm).astype(np.float32)
        else:
            author_vec = mean_vec.astype(np.float32)

        author_profiles[author_clean.lower()] = {
            "author": author_clean,
            "embedding": author_vec,
            "book_count": len(book_indices),
            "book_indices": book_indices,
            "isbns": group["ISBN"].tolist(),
        }

    return author_profiles


def compute_user_semantic_preference_embedding(
    user_ratings_df: pd.DataFrame,
    isbn_to_idx: Dict[str, int],
    embeddings: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Computes a User Preference Embedding representing the user's semantic taste profile.
    Takes a user's rated books and aggregates their embeddings weighted by rating.
    """
    if user_ratings_df.empty:
        return None

    valid_vecs = []
    weights = []

    for _, row in user_ratings_df.iterrows():
        isbn = str(row.get("ISBN", ""))
        rating = float(row.get("Rating", 0))

        if isbn in isbn_to_idx:
            idx = isbn_to_idx[isbn]
            vec = embeddings[idx]

            # Weight by rating: 1-10 mapped to [0.2, 1.0]; 0 (implicit read) weighted at 0.3
            w = max(0.2, rating / 10.0) if rating > 0 else 0.3
            valid_vecs.append(vec)
            weights.append(w)

    if not valid_vecs:
        return None

    valid_vecs = np.array(valid_vecs, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    weights /= np.sum(weights)

    weighted_mean = np.sum(valid_vecs * weights[:, np.newaxis], axis=0)
    norm = np.linalg.norm(weighted_mean)
    if norm > 1e-6:
        user_vec = (weighted_mean / norm).astype(np.float32)
    else:
        user_vec = weighted_mean.astype(np.float32)

    return user_vec


def get_similar_books(
    isbn: str,
    top_n: int = 10,
    faiss_index: Optional[faiss.IndexFlatIP] = None,
    embeddings: Optional[np.ndarray] = None,
    books_df: Optional[pd.DataFrame] = None,
    isbn_to_idx: Optional[Dict[str, int]] = None,
    idx_to_isbn: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve semantically similar books for a given ISBN using FAISS cosine similarity.
    Excludes the query book itself.
    """
    if isbn_to_idx is None or isbn not in isbn_to_idx:
        return []

    query_idx = isbn_to_idx[isbn]
    query_vector = embeddings[query_idx : query_idx + 1]

    k = min(top_n + 1, faiss_index.ntotal)
    distances, indices = faiss_index.search(query_vector, k)

    similar_books = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == query_idx or idx < 0:
            continue

        target_isbn = idx_to_isbn.get(idx)
        if not target_isbn:
            continue

        book_rows = books_df[books_df["ISBN"] == target_isbn]
        if book_rows.empty:
            continue

        row = book_rows.iloc[0]
        sim_score = max(0.0, min(1.0, float(dist)))
        cover_url = row.get("Image_URL_L") or row.get("Image_URL_M") or row.get("Image_URL_S") or ""

        similar_books.append(
            {
                "isbn": target_isbn,
                "title": str(row.get("Title", "Unknown Title")),
                "author": str(row.get("Author", "Unknown Author")),
                "publisher": str(row.get("Publisher", "Unknown Publisher")),
                "genres": str(row.get("Genres", "")),
                "subjects": str(row.get("Subjects", "")),
                "description": str(row.get("Description", "")),
                "year": int(row.get("Year", 0)),
                "cover_url": cover_url,
                "similarity_score": round(sim_score, 4),
                "similarity_percentage": round(sim_score * 100, 1),
            }
        )

        if len(similar_books) >= top_n:
            break

    return similar_books
