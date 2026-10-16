"""
Hybrid Recommendation Engine.
Fuses User Semantic Preference Embeddings (Sentence-BERT / MiniLM) with Neural Collaborative Filtering (NCF)
to generate personalized Top-N book recommendations.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from config import NCF_WEIGHT, SEMANTIC_WEIGHT, TOP_N
from recommender.semantic import compute_user_semantic_preference_embedding


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize an array of scores into [0.0, 1.0].
    If all scores are equal, returns uniform 0.5.
    """
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val > min_val:
        return (scores - min_val) / (max_val - min_val)
    return np.full_like(scores, 0.5)


def get_hybrid_recommendations(
    user_id: str,
    top_n: int = TOP_N,
    ncf_weight: float = NCF_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
    model_manager=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate personalized Top-N hybrid recommendations for a given User ID.

    Procedure:
        1. Obtain user's interaction/rating history.
        2. Compute user's composite Semantic Preference Embedding from highly rated books.
        3. Retrieve candidate books via FAISS semantic vector search around user's taste profile.
        4. Obtain NCF preference predictions for candidate books.
        5. Compute semantic preference scores against the user's semantic embedding.
        6. Normalize NCF and Semantic scores to [0, 1].
        7. Compute Hybrid Score = (ncf_weight * ncf_score) + (semantic_weight * semantic_score).
        8. Filter out books the user has already rated.
        9. Sort descending by Hybrid Score.
        10. Return Top-N recommendations with transparent 'Why this book?' reasoning.
    """
    if model_manager is None or not model_manager.is_ready():
        return [], {"error": "Models are not yet loaded or initialized."}

    user_id_str = str(user_id).strip()
    user_ratings = model_manager.get_user_ratings(user_id_str)
    user_idx = model_manager.user_to_idx.get(user_id_str)
    user_meta = model_manager.get_user_info(user_id_str)

    # 1. Cold Start handling for unknown users or users without ratings
    if user_idx is None or user_ratings.empty:
        popular_books = model_manager.get_popular_books(top_n=top_n)
        recommendations = []
        for b in popular_books:
            rec_item = dict(b)
            rec_item["ncf_score"] = 0.50
            rec_item["semantic_score"] = 0.50
            rec_item["hybrid_score"] = round(rec_item["avg_rating"] / 10.0, 4)
            rec_item["match_percentage"] = round(rec_item["hybrid_score"] * 100, 1)
            rec_item["reason"] = "Recommended based on community popularity and high reader satisfaction ratings."
            recommendations.append(rec_item)

        user_info = {
            "user_id": user_id_str,
            "is_known_user": False,
            "rated_count": 0,
            "avg_user_rating": 0.0,
            "demographics": user_meta,
            "cold_start": True,
        }
        return recommendations, user_info

    # 2. Extract user's rated books and compute User Semantic Preference Embedding
    rated_isbns = set(user_ratings["ISBN"].tolist())
    sorted_history = user_ratings.sort_values(by="Rating", ascending=False)
    favorite_isbns = sorted_history["ISBN"].head(8).tolist()

    user_pref_vec = compute_user_semantic_preference_embedding(
        user_ratings_df=user_ratings,
        isbn_to_idx=model_manager.isbn_to_idx,
        embeddings=model_manager.embeddings,
    )

    # 3. Retrieve Candidate Books via FAISS Semantic Search
    candidate_indices = set()
    seed_source_map: Dict[int, str] = {}

    if user_pref_vec is not None:
        query_vec = user_pref_vec.reshape(1, -1)
        k_search = min(80, model_manager.faiss_index.ntotal)
        distances, indices = model_manager.faiss_index.search(query_vec, k_search)

        for dist, c_idx in zip(distances[0], indices[0]):
            if c_idx < 0:
                continue
            c_isbn = model_manager.idx_to_isbn.get(c_idx)
            if c_isbn and c_isbn not in rated_isbns:
                candidate_indices.add(c_idx)
                seed_source_map[c_idx] = "your holistic reading taste profile"

    # Augment candidate pool with specific top favorite seeds
    for fav_isbn in favorite_isbns[:3]:
        if fav_isbn not in model_manager.isbn_to_idx:
            continue
        fav_idx = model_manager.isbn_to_idx[fav_isbn]
        fav_book_info = model_manager.get_book_by_isbn(fav_isbn)
        fav_title = fav_book_info.get("title") or fav_book_info.get("Title") or "your reading history"

        query_vec = model_manager.embeddings[fav_idx : fav_idx + 1]
        distances, indices = model_manager.faiss_index.search(query_vec, 30)

        for dist, c_idx in zip(distances[0], indices[0]):
            if c_idx < 0:
                continue
            c_isbn = model_manager.idx_to_isbn.get(c_idx)
            if c_isbn and c_isbn not in rated_isbns:
                candidate_indices.add(c_idx)
                if c_idx not in seed_source_map:
                    seed_source_map[c_idx] = fav_title

    candidate_list = list(candidate_indices)
    if not candidate_list:
        # Fallback to catalog sample
        candidate_list = [i for i in range(min(100, len(model_manager.idx_to_isbn))) if model_manager.idx_to_isbn.get(i) not in rated_isbns]

    # 4. Neural Collaborative Filtering Predictions
    ncf_raw_scores = model_manager.predict_ncf_scores(user_idx, candidate_list)

    # 5. Semantic Preference Scores (Cosine similarity between user preference vector and candidate book embeddings)
    if user_pref_vec is not None:
        candidate_vecs = model_manager.embeddings[candidate_list]
        semantic_raw_scores = np.dot(candidate_vecs, user_pref_vec)
    else:
        semantic_raw_scores = np.full(len(candidate_list), 0.5, dtype=np.float32)

    # 6. Normalize Scores to [0, 1]
    ncf_norm = normalize_scores(ncf_raw_scores)
    semantic_norm = normalize_scores(semantic_raw_scores)

    # 7. Compute Hybrid Score
    hybrid_scores = (ncf_weight * ncf_norm) + (semantic_weight * semantic_norm)

    # 8. Assemble, diversity rerank, and format recommendations
    ranked_candidates = []
    author_counts: Dict[str, int] = {}

    # Sort indices by hybrid_scores descending
    sorted_order = np.argsort(-hybrid_scores)

    for rank_pos in sorted_order:
        c_idx = candidate_list[rank_pos]
        isbn = model_manager.idx_to_isbn.get(c_idx)
        book_info = model_manager.get_book_by_isbn(isbn)
        if not book_info:
            continue

        author = str(book_info.get("author", "Unknown Author")).strip()
        # Diversity constraint: max 2 books per author
        if author_counts.get(author, 0) >= 2:
            continue

        h_score = float(hybrid_scores[rank_pos])
        n_score = float(ncf_norm[rank_pos])
        s_score = float(semantic_norm[rank_pos])
        source_title = seed_source_map.get(c_idx, "your profile history")

        genres = str(book_info.get("genres", "")).strip()
        if genres:
            reason = f"Blends your neural collaborative taste profile ({int(n_score*100)}% affinity) with strong {genres.split(',')[0]} thematic resonance to '{source_title}'."
        else:
            reason = f"Matches your learned reading taste profile ({int(n_score*100)}% affinity) and shares deep thematic relevance with '{source_title}'."

        ranked_candidates.append(
            {
                "isbn": isbn,
                "title": str(book_info.get("title") or book_info.get("Title") or "Unknown Title"),
                "author": author,
                "publisher": str(book_info.get("publisher") or book_info.get("Publisher") or "Unknown Publisher"),
                "genres": genres,
                "subjects": str(book_info.get("subjects", "")),
                "description": str(book_info.get("description", "")),
                "year": int(book_info.get("year") or book_info.get("Year") or 0),
                "cover_url": str(book_info.get("cover_url") or book_info.get("Image_URL_L") or book_info.get("Image_URL_M") or ""),
                "avg_rating": float(book_info.get("avg_rating") or book_info.get("Avg_Rating") or 0.0),
                "rating_count": int(book_info.get("rating_count") or book_info.get("Rating_Count") or 0),
                "hybrid_score": round(h_score, 4),
                "ncf_score": round(n_score, 4),
                "semantic_score": round(s_score, 4),
                "match_percentage": round(h_score * 100, 1),
                "reason": reason,
            }
        )
        author_counts[author] = author_counts.get(author, 0) + 1

        if len(ranked_candidates) >= top_n:
            break

    user_info = {
        "user_id": user_id_str,
        "is_known_user": True,
        "rated_count": len(user_ratings),
        "avg_user_rating": round(float(user_ratings["Rating"].mean()), 2),
        "demographics": user_meta,
        "cold_start": False,
        "top_favorite_book": favorite_isbns[0] if favorite_isbns else None,
    }

    return ranked_candidates, user_info
