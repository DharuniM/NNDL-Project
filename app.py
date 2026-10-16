"""
SemanticReads — Flask Digital Library Recommendation Web Application
Combines Sentence-BERT (MiniLM), FAISS Vector Retrieval, Author Profile Embeddings,
and Neural Collaborative Filtering (NCF).
"""
import math
from flask import Flask, render_template, request, redirect, url_for, abort, jsonify
from config import DEBUG, SECRET_KEY, ITEMS_PER_PAGE, NCF_WEIGHT, SEMANTIC_WEIGHT, TOP_N
from recommender.utils import ModelManager
from recommender.semantic import get_similar_books
from recommender.semantic_search import semantic_search
from recommender.hybrid import get_hybrid_recommendations

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

# Initialize singleton ModelManager (loads models, author profiles & FAISS index once)
model_manager = ModelManager()


@app.context_processor
def inject_global_context():
    """Inject readiness status and dataset stats globally into all templates."""
    return {
        "models_ready": model_manager.is_ready(),
        "stats": model_manager.stats,
    }


# ============================================================================
# Route 1: Home Page
# ============================================================================
@app.route("/")
def index():
    """
    Home landing page showing:
    - Hero banner with direct link to Semantic Discovery
    - Dynamic dataset metrics (Books, Users, Ratings, Average Score)
    - Top 8 popular books (weighted by rating count + explicit average)
    - 4-step architecture explanation
    """
    popular_books = model_manager.get_popular_books(top_n=8)
    return render_template(
        "index.html",
        popular_books=popular_books,
    )


# ============================================================================
# Route 2: Explore Catalog (Standard Filter & Search)
# ============================================================================
@app.route("/books")
@app.route("/explore")
def explore():
    """
    Searchable and filterable catalog with pagination.
    Used for locating specific known titles, authors, or publishers.
    """
    query = request.args.get("q", "").strip()
    author = request.args.get("author", "").strip()
    publisher = request.args.get("publisher", "").strip()

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    per_page = ITEMS_PER_PAGE
    books, total_matching = model_manager.search_books(
        query=query,
        author=author,
        publisher=publisher,
        page=page,
        per_page=per_page,
    )

    total_pages = math.ceil(total_matching / per_page) if total_matching > 0 else 1

    start_p = max(1, page - 2)
    end_p = min(total_pages, page + 2)
    page_range = list(range(start_p, end_p + 1))

    return render_template(
        "explore.html",
        books=books,
        total_books=model_manager.stats.get("total_books", 0),
        total_matching=total_matching,
        current_page=page,
        total_pages=total_pages,
        page_range=page_range,
        query=query,
        author=author,
        publisher=publisher,
    )


# ============================================================================
# Route 3: Standard Search Redirect
# ============================================================================
@app.route("/search")
def search():
    """Redirect navbar search queries directly into explore catalog."""
    q = request.args.get("q", "").strip()
    return redirect(url_for("explore", q=q))


# ============================================================================
# Route 4: Book Details
# ============================================================================
@app.route("/book/<isbn>")
def book_details(isbn):
    """
    Detailed book view showing metadata, large cover,
    genres/descriptions if available, and FAISS-retrieved semantically similar books.
    """
    book = model_manager.get_book_by_isbn(isbn)
    if not book:
        abort(404)

    similar_books = []
    if model_manager.is_ready():
        similar_books = get_similar_books(
            isbn=book["isbn"],
            top_n=6,
            faiss_index=model_manager.faiss_index,
            embeddings=model_manager.embeddings,
            books_df=model_manager.books_df,
            isbn_to_idx=model_manager.isbn_to_idx,
            idx_to_isbn=model_manager.idx_to_isbn,
        )

    return render_template(
        "book.html",
        book=book,
        similar_books=similar_books,
    )


# ============================================================================
# Route 5: Similar Books (FAISS Retrieval)
# ============================================================================
@app.route("/similar/<isbn>")
def similar_books(isbn):
    """Dedicated page displaying full list of semantically similar books."""
    book = model_manager.get_book_by_isbn(isbn)
    if not book:
        abort(404)

    similar_list = []
    if model_manager.is_ready():
        similar_list = get_similar_books(
            isbn=book["isbn"],
            top_n=12,
            faiss_index=model_manager.faiss_index,
            embeddings=model_manager.embeddings,
            books_df=model_manager.books_df,
            isbn_to_idx=model_manager.isbn_to_idx,
            idx_to_isbn=model_manager.idx_to_isbn,
        )

    return render_template(
        "similar.html",
        source_book=book,
        similar_books=similar_list,
    )


# ============================================================================
# Route 6: Semantic Discovery & Personalized Recommendations
# ============================================================================
@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    """
    Dual-functionality Recommendation Center:
    1. Semantic Book Discovery: Natural language theme/author/mood vector search via MiniLM & FAISS.
    2. Personalized Recommendations: NCF Collaborative Filtering + User Semantic Preference Embedding.
    """
    sample_users = model_manager.get_active_users_sample(limit=15)
    
    # 1. Check for Semantic Discovery Query
    semantic_query = request.values.get("semantic_query", "").strip()
    semantic_results = None

    if semantic_query:
        semantic_results = semantic_search(
            query=semantic_query,
            top_n=16,
            model_manager=model_manager,
        )

    # 2. Check for User ID Personalized Hybrid Recommendations
    user_id = request.values.get("user_id", "").strip()
    personalized_recs = []
    user_info = None

    if user_id:
        personalized_recs, user_info = get_hybrid_recommendations(
            user_id=user_id,
            top_n=TOP_N,
            ncf_weight=NCF_WEIGHT,
            semantic_weight=SEMANTIC_WEIGHT,
            model_manager=model_manager,
        )

    return render_template(
        "recommendations.html",
        sample_users=sample_users,
        semantic_query=semantic_query,
        semantic_results=semantic_results,
        current_user_id=user_id,
        recommendations=personalized_recs,
        user_info=user_info,
    )


# ============================================================================
# Route 7: Reader Profile
# ============================================================================
@app.route("/profile")
def profile_default():
    """Default profile view: loads top active reader."""
    sample_users = model_manager.get_active_users_sample(limit=15)
    target_uid = request.args.get("user_id") or (sample_users[0]["user_id"] if sample_users else "11676")
    return redirect(url_for("profile", user_id=target_uid))


@app.route("/profile/<user_id>")
def profile(user_id):
    """
    Reader Profile page showing demographics, rating statistics,
    reading history table, favorite authors, and personalized recommendations.
    """
    user_id_str = str(user_id).strip()
    user_meta = model_manager.get_user_info(user_id_str)
    ratings_df = model_manager.get_user_ratings(user_id_str)

    books_rated_count = len(ratings_df)
    avg_rating = round(float(ratings_df["Rating"].mean()), 2) if books_rated_count > 0 else 0.0
    highest_rating = int(ratings_df["Rating"].max()) if books_rated_count > 0 else 0

    fav_authors = []
    if not ratings_df.empty:
        high_rated = ratings_df[ratings_df["Rating"] >= 5]
        if high_rated.empty:
            high_rated = ratings_df
        fav_authors = high_rated["Author"].value_counts().head(5).index.tolist()

    reading_history = ratings_df.sort_values(by="Rating", ascending=False).to_dict(orient="records")

    personalized_recs, _ = get_hybrid_recommendations(
        user_id=user_id_str,
        top_n=6,
        ncf_weight=NCF_WEIGHT,
        semantic_weight=SEMANTIC_WEIGHT,
        model_manager=model_manager,
    )

    profile_data = {
        "user_id": user_id_str,
        "location": user_meta.get("location", "Unknown"),
        "age": user_meta.get("age"),
        "books_rated_count": books_rated_count,
        "avg_rating": avg_rating,
        "highest_rating": highest_rating,
        "favorite_authors": fav_authors,
    }

    sample_users = model_manager.get_active_users_sample(limit=15)

    return render_template(
        "profile.html",
        profile=profile_data,
        reading_history=reading_history,
        recommendations=personalized_recs,
        sample_users=sample_users,
    )


# ============================================================================
# Route 8: Architecture Page
# ============================================================================
@app.route("/architecture")
def architecture():
    """Detailed architectural diagram and neural pipeline documentation."""
    return render_template("architecture.html")


# ============================================================================
# Route 9: About Page
# ============================================================================
@app.route("/about")
def about():
    """Research problem formulation, motivation, and technology stack."""
    return render_template("about.html")


# ============================================================================
# Error Handlers
# ============================================================================
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=DEBUG)
