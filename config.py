"""
Configuration module for the SemanticReads Recommendation System.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure runtime directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Recommendation Weights (Hybrid Ranking)
NCF_WEIGHT = 0.60
SEMANTIC_WEIGHT = 0.40
TOP_N = 10

# Semantic Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Data Filtering & Preprocessing Configuration
MIN_BOOK_RATINGS = 5      # Filter books with at least this many ratings for catalog density
MIN_USER_RATINGS = 5      # Filter users with at least this many ratings
MAX_CATALOG_BOOKS = 15000 # Top books to include in the semantic FAISS index
POPULAR_MIN_RATINGS = 8  # Minimum ratings to qualify for popular ranking

# NCF Training Hyperparameters
NCF_LATENT_DIM = 32
NCF_BATCH_SIZE = 128
NCF_EPOCHS = 8
NCF_LEARNING_RATE = 0.001

# Application Settings
DEBUG = True
SECRET_KEY = "semanticreads-secret-key-production-final-year"
ITEMS_PER_PAGE = 18
