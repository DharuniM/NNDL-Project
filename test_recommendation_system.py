"""
Comprehensive Verification and Test Suite for SemanticReads
Tests syntax, model loading, FAISS retrieval, NCF prediction, Hybrid recommendation, and Flask endpoints.
"""
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app import app, model_manager
from recommender.semantic import get_similar_books
from recommender.hybrid import get_hybrid_recommendations


class TestSemanticReadsSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure model manager is loaded and test client is configured."""
        app.config["TESTING"] = True
        cls.client = app.test_client()
        # Force reload in case models were trained after initial import
        model_manager.load_all()

    def test_01_models_readiness(self):
        """Verify that all ML assets and datasets are loaded properly."""
        self.assertTrue(
            model_manager.is_ready(),
            f"ModelManager is not ready! Status: {model_manager.get_status()}",
        )
        self.assertIsNotNone(model_manager.books_df)
        self.assertIsNotNone(model_manager.faiss_index)
        self.assertIsNotNone(model_manager.embeddings)
        self.assertIsNotNone(model_manager.ncf_model)
        self.assertGreater(model_manager.stats.get("total_books", 0), 1000)
        self.assertGreater(model_manager.stats.get("total_users", 0), 1000)
        self.assertGreater(model_manager.stats.get("total_ratings", 0), 10000)
        print(f"  [PASS] Models Ready. Stats: {model_manager.stats}")

    def test_02_faiss_semantic_similarity(self):
        """Verify semantic similarity retrieval using FAISS IndexFlatIP."""
        # Pick first available ISBN
        first_isbn = model_manager.idx_to_isbn[0]
        similar_books = get_similar_books(
            isbn=first_isbn,
            top_n=5,
            faiss_index=model_manager.faiss_index,
            embeddings=model_manager.embeddings,
            books_df=model_manager.books_df,
            isbn_to_idx=model_manager.isbn_to_idx,
            idx_to_isbn=model_manager.idx_to_isbn,
        )
        self.assertEqual(len(similar_books), 5)
        for b in similar_books:
            self.assertNotEqual(b["isbn"], first_isbn, "Source book should be excluded.")
            self.assertGreaterEqual(b["similarity_score"], 0.0)
            self.assertLessEqual(b["similarity_score"], 1.0)
        print(f"  [PASS] FAISS Semantic search returned {len(similar_books)} valid similar items.")

    def test_03_ncf_scoring(self):
        """Verify NCF preference prediction for a user across candidate books."""
        user_idx = 0
        candidate_indices = [1, 2, 3, 4, 5]
        scores = model_manager.predict_ncf_scores(user_idx, candidate_indices)
        self.assertEqual(len(scores), len(candidate_indices))
        for score in scores:
            self.assertGreaterEqual(float(score), 0.0)
            self.assertLessEqual(float(score), 1.0)
        print(f"  [PASS] NCF scores predicted successfully: {scores[:3]}")

    def test_04_hybrid_recommendation_flow(self):
        """Verify hybrid recommendation ranking for an active reader."""
        sample_users = model_manager.get_active_users_sample(limit=3)
        self.assertGreater(len(sample_users), 0)
        test_user_id = sample_users[0]["user_id"]

        recs, user_info = get_hybrid_recommendations(
            user_id=test_user_id,
            top_n=10,
            model_manager=model_manager,
        )
        self.assertEqual(len(recs), 10)
        self.assertTrue(user_info["is_known_user"])
        for r in recs:
            self.assertIn("title", r)
            self.assertIn("hybrid_score", r)
            self.assertIn("ncf_score", r)
            self.assertIn("semantic_score", r)
            self.assertIn("reason", r)
            self.assertGreaterEqual(r["hybrid_score"], 0.0)
            self.assertLessEqual(r["hybrid_score"], 1.0)
        print(f"  [PASS] Hybrid recommendations generated for User #{test_user_id} with Top-10 rankings.")

    def test_05_route_home(self):
        """Verify GET / endpoint."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"SemanticReads", res.data)
        self.assertIn(b"Discover Your Next Great Read", res.data)
        print("  [PASS] GET / returned HTTP 200")

    def test_06_route_explore_and_search(self):
        """Verify GET /books and GET /search?q=harry."""
        res_explore = self.client.get("/books")
        self.assertEqual(res_explore.status_code, 200)
        self.assertIn(b"Explore Book Catalog", res_explore.data)

        res_search = self.client.get("/search?q=harry")
        self.assertEqual(res_search.status_code, 302)  # redirects to /explore?q=harry

        res_filtered = self.client.get("/explore?q=harry")
        self.assertEqual(res_filtered.status_code, 200)
        print("  [PASS] GET /books & search filter returned HTTP 200")

    def test_07_route_book_details_and_similar(self):
        """Verify GET /book/<isbn> and GET /similar/<isbn> for valid and invalid ISBNs."""
        first_isbn = model_manager.idx_to_isbn[0]
        res_valid = self.client.get(f"/book/{first_isbn}")
        self.assertEqual(res_valid.status_code, 200)

        res_sim = self.client.get(f"/similar/{first_isbn}")
        self.assertEqual(res_sim.status_code, 200)

        res_invalid = self.client.get("/book/INVALID_99999_ISBN")
        self.assertEqual(res_invalid.status_code, 404)
        print("  [PASS] Book details, similarity page, and 404 handler validated.")

    def test_08_route_recommendations(self):
        """Verify GET & POST /recommendations."""
        res_get = self.client.get("/recommendations")
        self.assertEqual(res_get.status_code, 200)

        sample_users = model_manager.get_active_users_sample(limit=1)
        test_uid = sample_users[0]["user_id"]
        res_post = self.client.post("/recommendations", data={"user_id": test_uid})
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b"Personalized For You", res_post.data)
        print("  [PASS] GET & POST /recommendations returned HTTP 200")

    def test_09_route_profile(self):
        """Verify GET /profile/<user_id> and invalid fallback."""
        sample_users = model_manager.get_active_users_sample(limit=1)
        test_uid = sample_users[0]["user_id"]
        res_prof = self.client.get(f"/profile/{test_uid}")
        self.assertEqual(res_prof.status_code, 200)
        self.assertIn(b"Reading History", res_prof.data)
        print(f"  [PASS] GET /profile/{test_uid} returned HTTP 200")

    def test_10_routes_architecture_and_about(self):
        """Verify GET /architecture and GET /about."""
        res_arch = self.client.get("/architecture")
        self.assertEqual(res_arch.status_code, 200)
        self.assertIn(b"System Architecture", res_arch.data)

        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 200)
        self.assertIn(b"About SemanticReads", res_about.data)
        print("  [PASS] Architecture & About pages returned HTTP 200")


if __name__ == "__main__":
    unittest.main(verbosity=2)
