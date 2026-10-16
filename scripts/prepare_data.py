"""
Data preparation script.
Reads raw Book-Crossing CSVs and enriched metadata, cleans, builds structured semantic documents,
encodes IDs, and saves processed pickle assets.
Usage:
    python scripts/prepare_data.py
"""
import pickle
import time
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DATA_DIR, MODEL_DIR
from recommender.data_loader import load_raw_books, load_raw_users, load_raw_ratings
from recommender.preprocessing import process_dataset


def main():
    print("=" * 60)
    print("SemanticReads Data Preparation Pipeline")
    print("=" * 60)
    start_time = time.time()

    print("[1/4] Loading raw Book-Crossing CSV files...")
    df_books = load_raw_books(DATA_DIR)
    print(f"      Loaded {len(df_books):,} raw books.")

    df_users = load_raw_users(DATA_DIR)
    print(f"      Loaded {len(df_users):,} raw users.")

    df_ratings = load_raw_ratings(DATA_DIR)
    print(f"      Loaded {len(df_ratings):,} raw ratings.")

    print("[2/4] Integrating enriched metadata, cleaning, and building semantic documents...")
    (
        processed_books,
        processed_users,
        processed_ratings,
        user_to_idx,
        idx_to_user,
        isbn_to_idx,
        idx_to_isbn,
    ) = process_dataset(df_books, df_users, df_ratings, data_dir=DATA_DIR)

    enriched_count = int((processed_books["Genres"] != "").sum())
    print(f"      Cleaned books catalog size: {len(processed_books):,} books.")
    print(f"      Books with enriched metadata: {enriched_count:,} books.")
    print(f"      Active users count:          {len(processed_users):,} users.")
    print(f"      Active rating interactions:  {len(processed_ratings):,} ratings.")

    print("[3/4] Saving processed artifacts to models/ directory...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "processed_books.pkl", "wb") as f:
        pickle.dump(processed_books, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(MODEL_DIR / "processed_users.pkl", "wb") as f:
        pickle.dump(processed_users, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(MODEL_DIR / "processed_ratings.pkl", "wb") as f:
        pickle.dump(processed_ratings, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(MODEL_DIR / "user_mapping.pkl", "wb") as f:
        pickle.dump(
            {"user_to_idx": user_to_idx, "idx_to_user": idx_to_user},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with open(MODEL_DIR / "book_mapping.pkl", "wb") as f:
        pickle.dump(
            {"isbn_to_idx": isbn_to_idx, "idx_to_isbn": idx_to_isbn},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    elapsed = time.time() - start_time
    print(f"[4/4] Data preparation completed in {elapsed:.2f} seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
