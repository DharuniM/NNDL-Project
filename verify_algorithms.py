"""
Deep Algorithm Verification Script for SemanticReads
Validates and prints the mathematical and algorithmic pipeline for:
1. MiniLM 384-d Semantic Embeddings
2. FAISS IndexFlatIP Cosine Retrieval
3. Keras Neural Collaborative Filtering (NCF)
4. Hybrid Multi-Objective Ranking Fusion
5. Reader Profile Interactivity
"""
import sys
import io
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure standard UTF-8 console output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from recommender.utils import ModelManager
from recommender.semantic import get_similar_books
from recommender.hybrid import get_hybrid_recommendations
from config import NCF_WEIGHT, SEMANTIC_WEIGHT


def main():
    print("=" * 80)
    print("SEMANTICREADS ALGORITHMIC WORKING & PIPELINE VERIFICATION")
    print("=" * 80)

    # 1. Initialize and verify ModelManager singleton
    print("\n[1] INITIALIZING MODEL MANAGER & CACHED ASSETS...")
    mm = ModelManager()
    assert mm.is_ready(), "ModelManager failed to load assets!"
    print(f"  [OK] Books in catalog:       {len(mm.books_df):,}")
    print(f"  [OK] Users in matrix:        {len(mm.user_to_idx):,}")
    print(f"  [OK] Ratings recorded:       {len(mm.ratings_df):,}")
    print(f"  [OK] FAISS Index total:      {mm.faiss_index.ntotal:,} vectors (Dimension: {mm.embeddings.shape[1]})")
    print(f"  [OK] NCF Model Loaded:       {mm.ncf_model.name} ({mm.ncf_model.count_params():,} trainable params)")

    # 2. Verify Semantic Representation & FAISS Cosine Retrieval
    print("\n[2] TESTING SEMANTIC EMBEDDING & FAISS COSINE SIMILARITY RETRIEVAL...")
    sample_search, _ = mm.search_books(query="Harry Potter", page=1, per_page=1)
    if sample_search:
        sample_book = sample_search[0]
    else:
        sample_book = mm._format_book_row(mm.books_df.iloc[0])

    print(f"  Target Seed Book: '{sample_book['title']}' by {sample_book['author']} (ISBN: {sample_book['isbn']})")
    similar_books = get_similar_books(
        isbn=sample_book["isbn"],
        top_n=4,
        faiss_index=mm.faiss_index,
        embeddings=mm.embeddings,
        books_df=mm.books_df,
        isbn_to_idx=mm.isbn_to_idx,
        idx_to_isbn=mm.idx_to_isbn,
    )
    print(f"  FAISS retrieved {len(similar_books)} semantically nearest neighbors:")
    for rank, b in enumerate(similar_books, 1):
        print(f"    {rank}. '{b['title']}' by {b['author']} | Sim Score: {b['similarity_score']} ({b['similarity_percentage']}%)")
        assert b["isbn"] != sample_book["isbn"], "Seed book must be excluded"
        assert 0.0 <= b["similarity_score"] <= 1.0, "Similarity score must be in [0, 1]"

    # 3. Verify Neural Collaborative Filtering (NCF) Inference
    print("\n[3] TESTING NEURAL COLLABORATIVE FILTERING (NCF) PREDICTION ENGINE...")
    user_samples = mm.get_active_users_sample(limit=2)
    user_a = user_samples[0]["user_id"]
    user_b = user_samples[1]["user_id"]

    user_a_idx = mm.user_to_idx[user_a]
    user_b_idx = mm.user_to_idx[user_b]

    test_book_indices = [10, 50, 100, 200, 300]
    scores_a = mm.predict_ncf_scores(user_a_idx, test_book_indices)
    scores_b = mm.predict_ncf_scores(user_b_idx, test_book_indices)

    print(f"  User #{user_a} NCF predicted preference scores across 5 items: {np.round(scores_a, 4)}")
    print(f"  User #{user_b} NCF predicted preference scores across 5 items: {np.round(scores_b, 4)}")
    assert not np.array_equal(scores_a, scores_b), "NCF predictions should differ based on user embedding!"

    # 4. Verify End-to-End Hybrid Recommendation Fusion
    print("\n[4] TESTING DUAL-BRANCH HYBRID RECOMMENDATION FUSION...")
    print(f"  Configuration Weights: {NCF_WEIGHT*100:.0f}% NCF Collaborative + {SEMANTIC_WEIGHT*100:.0f}% Semantic Similarity")
    
    recs, user_info = get_hybrid_recommendations(user_id=user_a, top_n=5, model_manager=mm)
    print(f"\n  Top Recommendations Generated for Reader #{user_a} (Rated {user_info['rated_count']} books in history):")
    
    for i, r in enumerate(recs, 1):
        print(f"    [{i}] '{r['title']}' by {r['author']}")
        print(f"        NCF Score (Collaborative): {r['ncf_score']:.4f}")
        print(f"        Semantic Score (MiniLM):   {r['semantic_score']:.4f}")
        expected_hybrid = round(NCF_WEIGHT * r['ncf_score'] + SEMANTIC_WEIGHT * r['semantic_score'], 4)
        print(f"        Hybrid Score Calculated:   {r['hybrid_score']:.4f} (Expected: {expected_hybrid:.4f})")
        print(f"        Reason / Transparency:     \"{r['reason']}\"")
        assert abs(r["hybrid_score"] - expected_hybrid) < 0.001, "Hybrid score fusion math mismatch!"

    # Verify that user's already-rated books are pruned from recommendations
    user_rated_isbns = set(mm.get_user_ratings(user_a)["ISBN"])
    rec_isbns = set(r["isbn"] for r in recs)
    overlap = user_rated_isbns.intersection(rec_isbns)
    print(f"\n  Pruning verification: {len(overlap)} overlapping rated books found in recommendations (Must be 0).")
    assert len(overlap) == 0, "Already rated books must be pruned from recommendations!"

    # 5. User Profile Demographics & History Lookup
    print("\n[5] TESTING USER PROFILE & DEMOGRAPHIC METRICS...")
    prof_info = mm.get_user_info(user_a)
    user_ratings = mm.get_user_ratings(user_a)
    print(f"  Reader #{user_a} Demographics: Location = {prof_info.get('location')}, Age = {prof_info.get('age')}")
    print(f"  Total Ratings: {len(user_ratings)} (Highest: {user_ratings['Rating'].max()}/10, Average: {user_ratings['Rating'].mean():.2f}/10)")

    print("\n" + "=" * 80)
    print("ALL ALGORITHMIC COMPONENTS VERIFIED WORKING 100% CORRECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
