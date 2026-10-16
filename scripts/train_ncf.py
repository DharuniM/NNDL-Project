"""
NCF Model Training Script.
Trains Neural Collaborative Filtering on real Book-Crossing rating interactions
and saves the trained model to models/ncf_model.keras.
Usage:
    python scripts/train_ncf.py
"""
import pickle
import time
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import MODEL_DIR, NCF_LATENT_DIM, NCF_BATCH_SIZE, NCF_EPOCHS, NCF_LEARNING_RATE
from recommender.ncf import build_ncf_model


def main():
    print("=" * 60)
    print("Neural Collaborative Filtering (NCF) Training Pipeline")
    print("=" * 60)
    start_time = time.time()

    ratings_path = MODEL_DIR / "processed_ratings.pkl"
    user_map_path = MODEL_DIR / "user_mapping.pkl"
    book_map_path = MODEL_DIR / "book_mapping.pkl"

    if not (ratings_path.exists() and user_map_path.exists() and book_map_path.exists()):
        print("Error: Required preprocessed assets not found! Please run scripts/prepare_data.py first.")
        sys.exit(1)

    print("[1/5] Loading preprocessed ratings and ID mappings...")
    with open(ratings_path, "rb") as f:
        ratings_df = pickle.load(f)

    with open(user_map_path, "rb") as f:
        user_mapping = pickle.load(f)
        num_users = len(user_mapping["user_to_idx"])

    with open(book_map_path, "rb") as f:
        book_mapping = pickle.load(f)
        num_books = len(book_mapping["isbn_to_idx"])

    print(f"      Users: {num_users:,} | Books: {num_books:,} | Total Ratings: {len(ratings_df):,}")

    print("[2/5] Preparing training tensors and normalized targets...")
    users = ratings_df["user_idx"].values
    books = ratings_df["book_idx"].values
    # Target normalization: 0-10 scale mapped to [0.0, 1.0]
    # Explicit ratings (1-10) scaled proportionally; 0 (implicit) mapped to baseline 0.1
    targets = np.where(ratings_df["Rating"].values > 0, ratings_df["Rating"].values / 10.0, 0.1).astype(np.float32)

    # Train / Validation Split
    u_train, u_val, b_train, b_val, y_train, y_val = train_test_split(
        users, books, targets, test_size=0.15, random_state=42
    )
    print(f"      Train samples: {len(u_train):,} | Validation samples: {len(u_val):,}")

    print("[3/5] Instantiating NCF MLP Model...")
    model = build_ncf_model(
        num_users=num_users,
        num_books=num_books,
        latent_dim=NCF_LATENT_DIM,
        learning_rate=NCF_LEARNING_RATE,
    )
    model.summary()

    print(f"[4/5] Training NCF model for {NCF_EPOCHS} epochs (batch size: {NCF_BATCH_SIZE})...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        [u_train, b_train],
        y_train,
        validation_data=([u_val, b_val], y_val),
        epochs=NCF_EPOCHS,
        batch_size=NCF_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print("[5/5] Saving trained model to models/ncf_model.keras...")
    model_save_path = MODEL_DIR / "ncf_model.keras"
    model.save(model_save_path)
    print(f"      Model saved successfully to {model_save_path}")

    elapsed = time.time() - start_time
    print(f"NCF training completed in {elapsed:.2f} seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
