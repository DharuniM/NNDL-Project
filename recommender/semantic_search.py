"""
Semantic Book Discovery and Query Understanding Engine.
Translates natural language descriptions, moods, themes, author profiles,
and multi-book references into vector embeddings for FAISS semantic retrieval and diversity reranking.
"""
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2 normalize a 1D vector."""
    norm = np.linalg.norm(v)
    if norm > 1e-6:
        return (v / norm).astype(np.float32)
    return v.astype(np.float32)


def extract_known_books_from_query(query_text: str, model_manager) -> List[str]:
    """
    Identifies if a natural language query specifically references one or more books in the catalog.
    Handles 'books like X', 'similar to X', 'I like X, Y and Z'.
    """
    if model_manager.books_df is None:
        return []

    q_lower = query_text.lower()
    # Strip common prefix patterns
    cleaned_q = re.sub(r"^(books like|books similar to|similar to|i like|i love|recommend books like|something like)\s+", "", q_lower, flags=re.IGNORECASE).strip()

    # Split by comma, 'and', 'or'
    potential_titles = [t.strip() for t in re.split(r"[,&]|\band\b", cleaned_q) if len(t.strip()) > 2]
    potential_titles.append(cleaned_q)

    matched_isbns = []
    for title_cand in potential_titles:
        title_cand_clean = re.sub(r"[^\w\s]", "", title_cand).strip()
        if len(title_cand_clean) < 3:
            continue

        for _, row in model_manager.books_df.iterrows():
            b_title = str(row.get("Title", "")).lower()
            b_title_clean = re.sub(r"[^\w\s]", "", b_title).strip()

            # Exact or high-overlap title match
            if b_title_clean == title_cand_clean or (len(title_cand_clean) >= 4 and b_title_clean.startswith(title_cand_clean)):
                isbn = str(row.get("ISBN", ""))
                if isbn not in matched_isbns:
                    matched_isbns.append(isbn)
                break

    return matched_isbns


def detect_author_in_query(query_text: str, model_manager) -> Optional[Tuple[str, str]]:
    """
    Detects if the query primarily targets an author's catalog.
    Returns (author_key, author_display_name) if matched, else None.
    """
    if not model_manager.author_profiles:
        return None

    q_lower = query_text.lower().strip()
    # Remove phrases like 'books by', 'novels by', 'stories by', 'written by', 'books'
    clean_author_q = re.sub(r"\b(books by|novels by|works by|written by|author|books|stories)\b", "", q_lower).strip()
    clean_author_q = re.sub(r"[^\w\s]", "", clean_author_q).strip()

    if not clean_author_q:
        return None

    # 1. Exact match with an author profile
    if clean_author_q in model_manager.author_profiles:
        prof = model_manager.author_profiles[clean_author_q]
        return clean_author_q, prof["author"]

    # 2. Check if clean_author_q matches author name exactly
    for a_key, prof in model_manager.author_profiles.items():
        if a_key == clean_author_q or prof["author"].lower() == clean_author_q:
            return a_key, prof["author"]

    # 3. Check for specific author name if query starts with 'by [Author]'
    if q_lower.startswith("by "):
        author_part = q_lower[3:].strip()
        if author_part in model_manager.author_profiles:
            prof = model_manager.author_profiles[author_part]
            return author_part, prof["author"]

    return None


def format_book_result(row, sim_score: float, reason: str = "") -> Dict[str, Any]:
    """Formats a book row with similarity score and explanation."""
    isbn = str(row.get("ISBN", ""))
    pct = max(10, min(99, int(round(sim_score * 100))))
    return {
        "isbn": isbn,
        "title": str(row.get("Title", "Unknown Title")),
        "author": str(row.get("Author", "Unknown Author")),
        "publisher": str(row.get("Publisher", "Unknown Publisher")),
        "year": int(row.get("Year", 0)),
        "genres": str(row.get("Genres", "")),
        "subjects": str(row.get("Subjects", "")),
        "description": str(row.get("Description", "")),
        "cover_url": str(row.get("Image_URL_L") or row.get("Image_URL_M") or row.get("Image_URL_S") or ""),
        "avg_rating": float(row.get("Avg_Rating", 0.0)),
        "rating_count": int(row.get("Rating_Count", 0)),
        "similarity_score": round(sim_score, 4),
        "match_percentage": pct,
        "reason": reason,
    }


def perform_diversity_reranking(
    candidate_indices: List[int],
    candidate_distances: List[float],
    model_manager,
    top_n: int = 15,
    max_per_author: int = 2,
    exclude_isbns: Optional[set] = None,
    query_desc: str = "",
) -> List[Dict[str, Any]]:
    """
    Reranks candidate books by semantic similarity while ensuring author and thematic diversity.
    """
    if exclude_isbns is None:
        exclude_isbns = set()

    author_counts: Dict[str, int] = {}
    results = []

    for dist, idx in zip(candidate_distances, candidate_indices):
        if idx < 0:
            continue
        isbn = model_manager.idx_to_isbn.get(idx)
        if not isbn or isbn in exclude_isbns:
            continue

        book_rows = model_manager.books_df[model_manager.books_df["ISBN"] == isbn]
        if book_rows.empty:
            continue

        row = book_rows.iloc[0]
        author = str(row.get("Author", "Unknown Author")).strip()

        # Diversity check: limit books per author
        if author_counts.get(author, 0) >= max_per_author:
            continue

        sim = max(0.0, min(1.0, float(dist)))

        # Construct meaningful explanation based on semantic match
        genres = str(row.get("Genres", "")).strip()
        subjects = str(row.get("Subjects", "")).strip()
        if genres and subjects:
            reason = f"High semantic vector match with {genres.split(',')[0]} themes and {subjects.split(',')[0]} motifs."
        elif genres:
            reason = f"Strong thematic alignment with requested {genres} concepts."
        elif query_desc:
            reason = f"Semantic match ({int(sim*100)}%) to your query '{query_desc[:40]}...'."
        else:
            reason = f"Close vector similarity ({int(sim*100)}%) across book metadata."

        results.append(format_book_result(row, sim, reason))
        author_counts[author] = author_counts.get(author, 0) + 1

        if len(results) >= top_n:
            break

    return results


def semantic_search(
    query: str,
    top_n: int = 16,
    model_manager=None,
) -> Dict[str, Any]:
    """
    Main Semantic Book Discovery Interface.
    Understands meaning, themes, author profiles, and multi-book references using vector embeddings.

    Returns dictionary containing:
        - query: original query string
        - query_type: 'AUTHOR' | 'BOOK_REFERENCE' | 'GENERAL_SEMANTIC'
        - author_name: display name if author query
        - author_books: list of books by author if author query
        - similar_books: list of semantically recommended books
        - explanation: high-level reasoning for the discovery results
    """
    if model_manager is None or not model_manager.is_ready():
        return {
            "query": query,
            "query_type": "ERROR",
            "results": [],
            "error": "Semantic recommendation models are not loaded.",
        }

    q_clean = query.strip()
    if not q_clean:
        return {
            "query": query,
            "query_type": "EMPTY",
            "results": [],
        }

    # 1. Check for Author-Aware Query
    author_match = detect_author_in_query(q_clean, model_manager)
    if author_match:
        author_key, author_display_name = author_match
        author_prof = model_manager.author_profiles[author_key]
        author_embedding = author_prof["embedding"]
        author_isbns = set(author_prof["isbns"])

        # Section A: Books written by the author
        author_books_df = model_manager.books_df[model_manager.books_df["ISBN"].isin(author_isbns)]
        author_books = [format_book_result(row, 1.0, f"Written by {author_display_name}") for _, row in author_books_df.iterrows()]

        # Section B: Semantically related books from other authors using AUTHOR PROFILE EMBEDDING
        query_vec = author_embedding.reshape(1, -1)
        k_search = min(60, model_manager.faiss_index.ntotal)
        distances, indices = model_manager.faiss_index.search(query_vec, k_search)

        similar_books = perform_diversity_reranking(
            candidate_indices=indices[0],
            candidate_distances=distances[0],
            model_manager=model_manager,
            top_n=top_n,
            max_per_author=2,
            exclude_isbns=author_isbns,
            query_desc=f"works of {author_display_name}",
        )

        return {
            "query": q_clean,
            "query_type": "AUTHOR",
            "author_name": author_display_name,
            "author_books": author_books,
            "similar_books": similar_books,
            "explanation": f"These recommendations are based on the semantic characteristics of {author_display_name}'s books, including available descriptions, genres, and subjects.",
        }

    # 2. Check for Book Reference / Multi-Book Preference Query (e.g. 'books like Gone Girl')
    resolved_isbns = extract_known_books_from_query(q_clean, model_manager)
    if resolved_isbns:
        # Retrieve stored embeddings of the resolved books
        book_vecs = []
        seed_titles = []
        for isbn in resolved_isbns:
            if isbn in model_manager.isbn_to_idx:
                idx = model_manager.isbn_to_idx[isbn]
                book_vecs.append(model_manager.embeddings[idx])
                b_info = model_manager.get_book_by_isbn(isbn)
                if b_info:
                    seed_titles.append(b_info.get("title", ""))

        if book_vecs:
            # Multi-Book Preference Vector
            preference_vec = np.mean(book_vecs, axis=0)
            preference_vec = normalize_vector(preference_vec).reshape(1, -1)

            k_search = min(60, model_manager.faiss_index.ntotal)
            distances, indices = model_manager.faiss_index.search(preference_vec, k_search)

            seed_names = ", ".join([f"'{t}'" for t in seed_titles[:3]])
            similar_books = perform_diversity_reranking(
                candidate_indices=indices[0],
                candidate_distances=distances[0],
                model_manager=model_manager,
                top_n=top_n,
                max_per_author=2,
                exclude_isbns=set(resolved_isbns),
                query_desc=f"books similar to {seed_names}",
            )

            return {
                "query": q_clean,
                "query_type": "BOOK_REFERENCE",
                "seed_titles": seed_titles,
                "similar_books": similar_books,
                "explanation": f"Recommendations generated from the combined 384-dimensional semantic embedding of {seed_names}.",
            }

    # 3. General Semantic Description / Mood / Theme Query
    # Convert entire sentence into dense embedding using Sentence-BERT
    query_vector = model_manager.encode_text(q_clean).reshape(1, -1)
    query_vector = normalize_vector(query_vector[0]).reshape(1, -1)

    k_search = min(80, model_manager.faiss_index.ntotal)
    distances, indices = model_manager.faiss_index.search(query_vector, k_search)

    results = perform_diversity_reranking(
        candidate_indices=indices[0],
        candidate_distances=distances[0],
        model_manager=model_manager,
        top_n=top_n,
        max_per_author=2,
        query_desc=q_clean,
    )

    return {
        "query": q_clean,
        "query_type": "GENERAL_SEMANTIC",
        "similar_books": results,
        "explanation": f"Retrieved via cosine vector similarity in 384-d semantic space matching your description '{q_clean}'.",
    }
