# SemanticReads — Intelligent Digital Library Recommendation System

> **Final-Year Project in Computer Science & Engineering**  
> A production-ready hybrid recommendation platform combining **Sentence-BERT (MiniLM)** semantic vector embeddings, **FAISS** vector similarity retrieval, and **Neural Collaborative Filtering (NCF)** on the Book-Crossing dataset.

---

## 📖 System Overview

**SemanticReads** solves the fundamental challenges of modern digital libraries—information overload, keyword mismatch, cold-start sparsity, and popularity bias—by implementing a dual-branch neural architecture:

1. **Semantic Branch (Content Representation):** Transforms book titles, authors, and publisher metadata into continuous 384-dimensional dense semantic vectors using `all-MiniLM-L6-v2`. L2-normalized embeddings are indexed with Facebook AI Similarity Search (`FAISS IndexFlatIP`) for sub-millisecond cosine similarity retrieval.
2. **Collaborative Branch (Reader Taste Representation):** Implements a deep Neural Collaborative Filtering (NCF) Multi-Layer Perceptron (MLP) network in TensorFlow/Keras with 32-dimensional user and book latent embeddings to learn non-linear preference affinities from rating interactions.
3. **Hybrid Ranking Fusion:** Combines normalized collaborative preference scores and semantic similarity scores through weighted linear fusion:
   $$\text{Hybrid Score} = 0.60 \times \text{NCF Score} + 0.40 \times \text{Semantic Score}$$
4. **Explainable AI Delivery:** Provides transparent reasoning (*"Why this book?"*) showing exact collaborative match confidence and source-item semantic alignment.

---

## 🏛️ System Architecture

```
                                 USER
                                  ↓
                        WEB INTERFACE (HTML/CSS/JS)
                                  ↓
                        FLASK WEB APPLICATION
                                  ↓
                       BOOK-CROSSING DATASET
                                  ↓
                        DATA PREPROCESSING
                                  ↓
              ┌───────────────────┴───────────────────┐
              ↓                                       ↓
       SEMANTIC BRANCH                            NCF BRANCH
              ↓                                       ↓
    Sentence-BERT / MiniLM                      User Embedding (32-d)
    (all-MiniLM-L6-v2)                          Book Embedding (32-d)
              ↓                                       ↓
     384-d Dense Vectors                        Concatenate Embeddings
              ↓                                       ↓
        FAISS Index                             MLP Network
       (IndexFlatIP)                       (Dense 64 → 32 → 16 → 1)
              ↓                                       ↓
    Candidate Retrieval                     Learned Preference Score
              └───────────────────┬───────────────────┘
                                  ↓
                        HYBRID RANKING FUSION
                  0.60 × NCF + 0.40 × Semantic
                                  ↓
                      TOP-N RECOMMENDATIONS
                                  ↓
                         WEBSITE DELIVERY
```

---

## 💻 Tech Stack

- **Backend & Web Server:** Python 3, Flask 3.0, Gunicorn (WSGI)
- **Deep Learning & Neural Networks:** TensorFlow 2.x, Keras (Multi-Layer Perceptron NCF)
- **Semantic Text Representations:** Sentence-Transformers (`all-MiniLM-L6-v2`, 384 dimensions)
- **Vector Search Engine:** FAISS (Facebook AI Similarity Search - CPU, `IndexFlatIP`)
- **Data Engineering:** Pandas, NumPy, Scikit-Learn
- **Frontend & UI:** Semantic HTML5, Vanilla CSS3 (Custom Dark Navy Theme), Vanilla JavaScript (no heavy frontend frameworks required)

---

## 📁 Project Structure

```
Neuralnetworks_proj/
│
├── app.py                      # Production Flask web application & routes
├── config.py                   # Central configuration, paths & hybrid weights
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore definitions
│
├── data/                       # Book-Crossing tabular dataset
│   ├── BX-Books.csv
│   ├── BX-Users.csv
│   └── BX-Book-Ratings.csv
│
├── models/                     # Generated ML model artifacts & indices
│   ├── ncf_model.keras         # Trained Keras Neural Collaborative Filtering model
│   ├── faiss_index.bin         # Serialized FAISS IndexFlatIP vector index
│   ├── book_embeddings.npy     # 384-dimensional normalized semantic embeddings
│   ├── processed_books.pkl     # Cleaned active book catalog dataframe
│   ├── processed_users.pkl     # Active users demographic metadata
│   ├── processed_ratings.pkl   # Filtered interaction rating matrix
│   ├── user_mapping.pkl        # User ID <-> Sequential Index encodings
│   └── book_mapping.pkl        # ISBN <-> Sequential Index encodings
│
├── recommender/                # Recommendation engine modules
│   ├── __init__.py
│   ├── data_loader.py          # Safe CSV ingestion & error handling
│   ├── preprocessing.py        # Textual cleaning, ID encoding & rating normalization
│   ├── semantic.py             # Sentence-BERT embedding & FAISS similarity queries
│   ├── ncf.py                  # Keras NCF MLP architecture & inference
│   ├── hybrid.py               # Dual-branch hybrid ranking algorithm
│   └── utils.py                # ModelManager singleton & caching utility
│
├── scripts/                    # Offline training & preparation pipelines
│   ├── prepare_data.py         # Step 1: Ingest, clean, encode and save data pickles
│   ├── build_semantic_index.py # Step 2: Generate MiniLM embeddings & FAISS index
│   └── train_ncf.py            # Step 3: Train NCF deep neural network
│
├── templates/                  # Jinja2 HTML web templates
│   ├── base.html               # Main navbar, footer, and responsive container
│   ├── index.html              # Home landing page with live dataset statistics
│   ├── explore.html            # Searchable catalog with pagination & filters
│   ├── book.html               # Book detail view & live similar books
│   ├── similar.html            # Dedicated similarity report page
│   ├── recommendations.html    # Personalized hybrid recommendations page
│   ├── profile.html            # Reader profile, demographics & history
│   ├── architecture.html       # HTML/CSS interactive architecture diagram
│   ├── about.html              # Problem definition & methodology
│   ├── 404.html                # Not found error page
│   └── 500.html                # Server error page
│
└── static/                     # Static styling & web assets
    ├── css/
    │   ├── style.css           # Primary theme stylesheet
    │   └── responsive.css      # Mobile and tablet media queries
    ├── js/
    │   └── main.js             # Mobile navbar, loading state, image error fallback
    └── images/
        ├── logo.svg            # Custom SVG platform logo
        └── placeholder-book.svg # Fallback book cover artwork
```

---

## ⚙️ Installation & Setup

### 1. Clone & Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/book-recommendation-system.git
cd book-recommendation-system

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Placement

Ensure the three Book-Crossing dataset files are placed in `data/`:
- `data/BX-Books.csv`
- `data/BX-Users.csv`
- `data/BX-Book-Ratings.csv`

---

## 🚀 Model Preparation & Training Pipeline

Run the preparation scripts in order:

```bash
# 1. Clean data, encode IDs, and create active catalog
python scripts/prepare_data.py

# 2. Build 384-d Sentence-BERT embeddings and FAISS index
python scripts/build_semantic_index.py

# 3. Train Neural Collaborative Filtering (NCF) model
python scripts/train_ncf.py
```

---

## 🌐 Running the Web Application

Start the local Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🚢 Production Deployment

For production deployment on Linux servers, run via Gunicorn WSGI:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 🧪 Key Routes & Endpoints

| Route | Description |
|---|---|
| `GET /` | Home landing page with real-time dataset statistics and popular books |
| `GET /books` | Searchable and filterable catalog with pagination |
| `GET /book/<isbn>` | Comprehensive book view with FAISS similar books |
| `GET /similar/<isbn>` | Dedicated cosine similarity report |
| `GET/POST /recommendations` | Personalized hybrid recommendations with reader selector |
| `GET /profile/<user_id>` | Reader demographic profile, full reading history, and personalized shelf |
| `GET /architecture` | Pure HTML/CSS system architecture diagram and neural network breakdown |
| `GET /about` | Academic research statement and technology stack |
