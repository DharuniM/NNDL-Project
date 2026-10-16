"""
Comprehensive Semantic Discovery and Vector Query Verification Suite.
Tests the 5 core user queries and validates pure embedding-based retrieval.
"""
import sys
import io
import unittest
from pathlib import Path
import numpy as np

# Set UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app import app, model_manager
from recommender.semantic_search import semantic_search
from recommender.hybrid import get_hybrid_recommendations


class TestSemanticDiscovery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        model_manager.load_all()

    def test_01_psychological_thriller_family_secrets(self):
        """
        TEST 1:
        Query: 'psychological thriller involving family secrets'
        Must retrieve semantically relevant books using vector embeddings,
        even when the exact words 'psychological' or 'secrets' are absent from titles.
        """
        query = "psychological thriller involving family secrets"
        res = semantic_search(query, top_n=8, model_manager=model_manager)
        
        print(f"\n==================================================")
        print(f"TEST 1: '{query}'")
        print(f"Query Type: {res['query_type']}")
        print(f"Explanation: {res['explanation']}")
        print("Top Semantic Matches:")
        
        self.assertEqual(res["query_type"], "GENERAL_SEMANTIC")
        self.assertGreater(len(res["similar_books"]), 0)
        
        for i, b in enumerate(res["similar_books"], 1):
            print(f"  {i}. '{b['title']}' by {b['author']} | Match: {b['match_percentage']}%")
            if b.get("genres"):
                print(f"     Genres: {b['genres']}")
            print(f"     Why: {b['reason']}")
            self.assertGreaterEqual(b["similarity_score"], 0.25)

    def test_02_author_aware_discovery_claire_douglas(self):
        """
        TEST 2:
        Query: 'Claire Douglas'
        Must return TWO sections:
          Section A: Books written by Claire Douglas
          Section B: Semantically related books from other authors using Author Profile Embedding
        """
        query = "Claire Douglas"
        res = semantic_search(query, top_n=6, model_manager=model_manager)

        print(f"\n==================================================")
        print(f"TEST 2: '{query}'")
        print(f"Query Type: {res['query_type']}")
        print(f"Author Detected: {res.get('author_name')}")
        
        self.assertEqual(res["query_type"], "AUTHOR")
        self.assertIn("author_books", res)
        self.assertIn("similar_books", res)
        self.assertGreater(len(res["author_books"]), 0)
        self.assertGreater(len(res["similar_books"]), 0)

        print(f"Section A: Books written by {res['author_name']} ({len(res['author_books'])} found):")
        for b in res["author_books"]:
            print(f"  - '{b['title']}' ({b['year']})")
            self.assertEqual(b["author"].lower(), "claire douglas")

        print(f"Section B: Similar Books by Other Authors (Retrieved via Author Profile Vector):")
        for b in res["similar_books"]:
            print(f"  - '{b['title']}' by {b['author']} | Match: {b['match_percentage']}% | Why: {b['reason']}")
            self.assertNotEqual(b["author"].lower(), "claire douglas", "Section B must contain other authors")

    def test_03_books_like_gone_girl(self):
        """
        TEST 3:
        Query: 'books like Gone Girl'
        Must resolve Gone Girl to its stored book embedding and search FAISS vector index.
        """
        query = "books like Gone Girl"
        res = semantic_search(query, top_n=6, model_manager=model_manager)

        print(f"\n==================================================")
        print(f"TEST 3: '{query}'")
        print(f"Query Type: {res['query_type']}")
        print(f"Seed Titles Resolved: {res.get('seed_titles')}")
        
        self.assertEqual(res["query_type"], "BOOK_REFERENCE")
        self.assertGreater(len(res["similar_books"]), 0)
        
        print("Top Semantic Recommendations:")
        for i, b in enumerate(res["similar_books"], 1):
            print(f"  {i}. '{b['title']}' by {b['author']} | Match: {b['match_percentage']}%")
            print(f"     Why: {b['reason']}")
            # Make sure Gone Girl itself was excluded from recommendation results
            self.assertNotEqual(b["title"].lower(), "gone girl")

    def test_04_romantic_enemies_to_lovers(self):
        """
        TEST 4:
        Query: 'romantic story with enemies becoming lovers'
        Must return semantically related romance results without requiring title keyword matches.
        """
        query = "romantic story with enemies becoming lovers"
        res = semantic_search(query, top_n=6, model_manager=model_manager)

        print(f"\n==================================================")
        print(f"TEST 4: '{query}'")
        print(f"Query Type: {res['query_type']}")
        print("Top Semantic Matches:")
        
        self.assertEqual(res["query_type"], "GENERAL_SEMANTIC")
        self.assertGreater(len(res["similar_books"]), 0)
        
        for i, b in enumerate(res["similar_books"], 1):
            print(f"  {i}. '{b['title']}' by {b['author']} | Match: {b['match_percentage']}%")
            print(f"     Why: {b['reason']}")

    def test_05_mystery_disappearance_hidden_history(self):
        """
        TEST 5:
        Query: 'mystery involving disappearance and hidden family history'
        Must return relevant semantic results based on description/genre/subject embeddings.
        """
        query = "mystery involving disappearance and hidden family history"
        res = semantic_search(query, top_n=6, model_manager=model_manager)

        print(f"\n==================================================")
        print(f"TEST 5: '{query}'")
        print(f"Query Type: {res['query_type']}")
        print("Top Semantic Matches:")
        
        self.assertEqual(res["query_type"], "GENERAL_SEMANTIC")
        self.assertGreater(len(res["similar_books"]), 0)
        
        for i, b in enumerate(res["similar_books"], 1):
            print(f"  {i}. '{b['title']}' by {b['author']} | Match: {b['match_percentage']}%")
            print(f"     Why: {b['reason']}")

    def test_06_user_preference_embedding_hybrid_personalization(self):
        """
        TEST 6:
        Verify User Preference Embedding + NCF Hybrid Personalization for an active user.
        """
        sample_users = model_manager.get_active_users_sample(limit=1)
        test_uid = sample_users[0]["user_id"]
        recs, user_info = get_hybrid_recommendations(user_id=test_uid, top_n=5, model_manager=model_manager)

        print(f"\n==================================================")
        print(f"TEST 6: Hybrid Personalization for Reader #{test_uid}")
        print(f"Ratings in History: {user_info['rated_count']}")
        
        self.assertEqual(len(recs), 5)
        for i, r in enumerate(recs, 1):
            print(f"  {i}. '{r['title']}' by {r['author']} | Hybrid: {r['hybrid_score']} (NCF: {r['ncf_score']}, Semantic: {r['semantic_score']})")
            print(f"     Reason: {r['reason']}")
            expected = round(0.60 * r['ncf_score'] + 0.40 * r['semantic_score'], 4)
            self.assertAlmostEqual(r['hybrid_score'], expected, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
