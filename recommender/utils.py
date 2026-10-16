"""
Model Manager and Application Utilities.
Loads models, FAISS vector indices, author profiles, and preprocessed datasets ONCE into memory.
Provides fast query methods for book lookups, user profiles, search, popularity, and on-the-fly query encoding.
"""
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import faiss
import tensorflow as tf
from sentence_transformers import SentenceTransformer

from config import MODEL_DIR, POPULAR_MIN_RATINGS, EMBEDDING_MODEL_NAME


class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir: Path = MODEL_DIR):
        if self._initialized:
            return

        self.model_dir = model_dir
        self.books_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.isbn_to_idx: Dict[str, int] = {}
        self.idx_to_isbn: Dict[int, str] = {}

        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None
        self.author_profiles: Dict[str, Dict[str, Any]] = {}
        self.sentence_model: Optional[SentenceTransformer] = None
        self.ncf_model: Optional[tf.keras.Model] = None

        self.stats: Dict[str, Any] = {}
        self.popular_books_cache: List[Dict[str, Any]] = []
        self._is_ready = False
        self._load_status: Dict[str, str] = {
            "data": "Not loaded",
            "semantic_faiss": "Not loaded",
            "author_profiles": "Not loaded",
            "ncf_model": "Not loaded",
        }

        self.load_all()
        self._initialized = True

    def is_ready(self) -> bool:
        """Check if all ML models, indices, and datasets are loaded."""
        return self._is_ready

    def get_status(self) -> Dict[str, str]:
        """Return loading status for each component."""
        return self._load_status

    def load_all(self):
        """Load datasets, mappings, FAISS index, author profiles, and NCF model."""
        try:
            # 1. Load DataFrames & Mappings
            books_path = self.model_dir / "processed_books.pkl"
            users_path = self.model_dir / "processed_users.pkl"
            ratings_path = self.model_dir / "processed_ratings.pkl"
            user_map_path = self.model_dir / "user_mapping.pkl"
            book_map_path = self.model_dir / "book_mapping.pkl"

            if (
                books_path.exists()
                and users_path.exists()
                and ratings_path.exists()
                and user_map_path.exists()
                and book_map_path.exists()
            ):
                with open(books_path, "rb") as f:
                    self.books_df = pickle.load(f)
                with open(users_path, "rb") as f:
                    self.users_df = pickle.load(f)
                with open(ratings_path, "rb") as f:
                    self.ratings_df = pickle.load(f)

                with open(user_map_path, "rb") as f:
                    u_map = pickle.load(f)
                    self.user_to_idx = u_map["user_to_idx"]
                    self.idx_to_user = u_map["idx_to_user"]

                with open(book_map_path, "rb") as f:
                    b_map = pickle.load(f)
                    self.isbn_to_idx = b_map["isbn_to_idx"]
                    self.idx_to_isbn = b_map["idx_to_isbn"]

                self._compute_statistics()
                self._compute_popular_books()
                self._load_status["data"] = "Ready"
            else:
                self._load_status["data"] = "Missing pickle assets"

            # 2. Load FAISS & Semantic Embeddings
            embeddings_path = self.model_dir / "book_embeddings.npy"
            faiss_path = self.model_dir / "faiss_index.bin"
            author_profiles_path = self.model_dir / "author_profiles.pkl"

            if embeddings_path.exists() and faiss_path.exists():
                self.embeddings = np.load(embeddings_path)
                self.faiss_index = faiss.read_index(str(faiss_path))
                self._load_status["semantic_faiss"] = f"Ready ({self.faiss_index.ntotal} vectors)"

                if author_profiles_path.exists():
                    with open(author_profiles_path, "rb") as f:
                        self.author_profiles = pickle.load(f)
                    self._load_status["author_profiles"] = f"Ready ({len(self.author_profiles)} authors)"
                else:
                    self._load_status["author_profiles"] = "Missing author_profiles.pkl"
            else:
                self._load_status["semantic_faiss"] = "Missing index or embeddings"

            # 3. Load SentenceTransformer Encoder for on-the-fly query encoding
            try:
                self.sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception as e:
                print(f"[ModelManager] SentenceTransformer load warning: {e}")

            # 4. Load Keras NCF Model
            ncf_path = self.model_dir / "ncf_model.keras"
            if ncf_path.exists():
                self.ncf_model = tf.keras.models.load_model(ncf_path)
                self._load_status["ncf_model"] = "Ready"
            else:
                self._load_status["ncf_model"] = "Missing ncf_model.keras"

            self._is_ready = (
                self.books_df is not None
                and self.faiss_index is not None
                and self.embeddings is not None
                and self.ncf_model is not None
            )

        except Exception as e:
            print(f"[ModelManager] Initialization warning: {e}")
            self._is_ready = False

    def encode_text(self, text: str) -> np.ndarray:
        """Encode natural language query into 384-d L2-normalized vector."""
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        vec = self.sentence_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)

    def _compute_statistics(self):
        """Compute live statistics from loaded data."""
        if self.books_df is None or self.ratings_df is None:
            self.stats = {
                "total_books": 0,
                "total_users": 0,
                "total_ratings": 0,
                "avg_rating": 0.0,
                "unique_authors": 0,
                "unique_publishers": 0,
                "enriched_books_count": 0,
            }
            return

        enriched_count = int((self.books_df["Genres"] != "").sum()) if "Genres" in self.books_df.columns else 0

        self.stats = {
            "total_books": len(self.books_df),
            "total_users": len(self.user_to_idx),
            "total_ratings": len(self.ratings_df),
            "avg_rating": round(float(self.ratings_df["Rating"].mean()), 2),
            "unique_authors": int(self.books_df["Author"].nunique()),
            "unique_publishers": int(self.books_df["Publisher"].nunique()),
            "enriched_books_count": enriched_count,
        }

    def _compute_popular_books(self, min_ratings: int = POPULAR_MIN_RATINGS):
        """Compute popular books using Bayesian weighted rating formula."""
        if self.books_df is None or self.ratings_df is None:
            self.popular_books_cache = []
            return

        c_mean = float(self.ratings_df["Rating"].mean())
        m_threshold = min_ratings

        df = self.books_df.copy()
        v = df["Rating_Count"].values
        r = df["Avg_Rating"].values
        weighted_score = (v / (v + m_threshold)) * r + (m_threshold / (v + m_threshold)) * c_mean
        df["Popularity_Score"] = weighted_score

        top_popular = df.sort_values(by="Popularity_Score", ascending=False).head(30)
        self.popular_books_cache = []
        for _, row in top_popular.iterrows():
            self.popular_books_cache.append(self._format_book_row(row))

    def _format_book_row(self, row: pd.Series) -> Dict[str, Any]:
        """Format a book dataframe row into standard dictionary with all real metadata."""
        isbn = str(row.get("ISBN", ""))
        genres = str(row.get("Genres", "")).strip()
        subjects = str(row.get("Subjects", "")).strip()
        description = str(row.get("Description", "")).strip()

        return {
            "isbn": isbn,
            "title": str(row.get("Title", "Unknown Title")),
            "author": str(row.get("Author", "Unknown Author")),
            "publisher": str(row.get("Publisher", "Unknown Publisher")),
            "genres": genres if genres != "nan" else "",
            "subjects": subjects if subjects != "nan" else "",
            "description": description if description != "nan" else "",
            "year": int(row.get("Year", 0)),
            "cover_url": str(row.get("Image_URL_L") or row.get("Image_URL_M") or row.get("Image_URL_S") or ""),
            "rating_count": int(row.get("Rating_Count", 0)),
            "avg_rating": float(row.get("Avg_Rating", 0.0)),
            "explicit_avg": float(row.get("Explicit_Avg", 0.0)),
        }

    def get_popular_books(self, top_n: int = 8) -> List[Dict[str, Any]]:
        """Retrieve top popular books from precomputed cache."""
        return self.popular_books_cache[:top_n]

    def get_book_by_isbn(self, isbn: str) -> Optional[Dict[str, Any]]:
        """Look up book details by ISBN."""
        if self.books_df is None:
            return None
        match = self.books_df[self.books_df["ISBN"] == str(isbn).strip()]
        if match.empty:
            return None
        return self._format_book_row(match.iloc[0])

    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get demographics for a user ID if available."""
        user_id_str = str(user_id).strip()
        if self.users_df is not None:
            match = self.users_df[self.users_df["User_ID"] == user_id_str]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "user_id": user_id_str,
                    "location": str(row.get("Location", "Unknown Location")),
                    "age": int(row.get("Age")) if pd.notna(row.get("Age")) and row.get("Age") is not None else None,
                }
        return {"user_id": user_id_str, "location": "Unknown Location", "age": None}

    def get_user_ratings(self, user_id: str) -> pd.DataFrame:
        """Get user rating history merged with book metadata."""
        user_id_str = str(user_id).strip()
        if self.ratings_df is None:
            return pd.DataFrame()
        user_ratings = self.ratings_df[self.ratings_df["User_ID"] == user_id_str].copy()
        if user_ratings.empty or self.books_df is None:
            return user_ratings

        cols = ["ISBN", "Title", "Author", "Publisher", "Year", "Image_URL_L", "Image_URL_M"]
        if "Genres" in self.books_df.columns:
            cols.extend(["Genres", "Description"])
        return user_ratings.merge(
            self.books_df[cols],
            on="ISBN",
            how="inner",
        )

    def get_active_users_sample(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Return sample active users for easy testing and demo selector."""
        if self.ratings_df is None:
            return []
        top_user_counts = self.ratings_df["User_ID"].value_counts().head(limit)
        sample = []
        for uid, count in top_user_counts.items():
            user_meta = self.get_user_info(uid)
            sample.append({
                "user_id": uid,
                "ratings_count": int(count),
                "location": user_meta.get("location", "Unknown"),
                "age": user_meta.get("age"),
            })
        return sample

    def search_books(
        self,
        query: str = "",
        author: str = "",
        publisher: str = "",
        page: int = 1,
        per_page: int = 18,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Catalog search by title, author, and publisher with pagination.
        For exact lookup when user specifically searches a known book.
        """
        if self.books_df is None:
            return [], 0

        df = self.books_df
        if query:
            q_clean = query.strip().lower()
            df = df[
                df["Title"].str.lower().str.contains(q_clean, na=False)
                | df["Author"].str.lower().str.contains(q_clean, na=False)
                | df["Publisher"].str.lower().str.contains(q_clean, na=False)
                | df["ISBN"].str.lower().str.contains(q_clean, na=False)
            ]

        if author:
            a_clean = author.strip().lower()
            df = df[df["Author"].str.lower().str.contains(a_clean, na=False)]

        if publisher:
            p_clean = publisher.strip().lower()
            df = df[df["Publisher"].str.lower().str.contains(p_clean, na=False)]

        total_count = len(df)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        paginated_df = df.iloc[start_idx:end_idx]
        books = [self._format_book_row(row) for _, row in paginated_df.iterrows()]
        return books, total_count

    def predict_ncf_scores(self, user_idx: int, book_indices: List[int]) -> np.ndarray:
        """Batch predict NCF preference scores."""
        if self.ncf_model is None or not book_indices:
            return np.full(len(book_indices), 0.5, dtype=np.float32)

        # Ensure indices are within embedding bounds
        max_book_idx = self.ncf_model.get_layer("book_embedding").input_dim
        valid_indices = [min(idx, max_book_idx - 1) for idx in book_indices]

        num_items = len(valid_indices)
        user_arr = np.full(num_items, user_idx, dtype=np.int32)
        book_arr = np.array(valid_indices, dtype=np.int32)

        preds = self.ncf_model.predict([user_arr, book_arr], batch_size=512, verbose=0)
        return preds.flatten()
