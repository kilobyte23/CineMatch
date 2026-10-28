import os
import tempfile
import unittest

from CineMatch import MovieRecommendationSystem


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.system = MovieRecommendationSystem.with_default_catalogue()

    def test_add_validates_input(self):
        with self.assertRaises(ValueError):
            self.system.add_movie("Bad", "Drama", 11)
        with self.assertRaises(ValueError):
            self.system.add_movie("Inception", "Sci-Fi", 8.8)  # duplicate
        movie = self.system.add_movie("Heat", "Crime / Thriller", 8.3, 1995, "Michael Mann")
        self.assertEqual(movie.genres, ["Crime", "Thriller"])

    def test_search_matches_title_genre_and_director(self):
        titles = lambda ms: {m.title for m in ms}
        self.assertIn("Inception", titles(self.system.search_movie("incep")))
        self.assertIn("Alien", titles(self.system.search_movie("horror")))
        self.assertIn("Fight Club", titles(self.system.search_movie("fincher")))

    def test_top_movies_by_genre(self):
        top = self.system.recommend_top_movies(3, genre="sci-fi")
        self.assertEqual(len(top), 3)
        self.assertTrue(all("sci-fi" in {g.lower() for g in m.genres} for m in top))
        self.assertEqual(top[0].title, "Inception")

    def test_delete_removes_rating(self):
        self.system.rate_movie("Inception", 9)
        self.assertTrue(self.system.delete_movie("inception"))
        self.assertFalse(self.system.delete_movie("inception"))
        self.assertEqual(self.system.user_ratings, {})


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.system = MovieRecommendationSystem.with_default_catalogue()

    def test_similar_prefers_shared_genres_and_director(self):
        similar = [m.title for m, _ in self.system.recommend_similar("The Fellowship of the Ring", 2)]
        self.assertEqual(set(similar), {
            "The Lord of the Rings: The Two Towers",
            "The Lord of the Rings: The Return of the King",
        })

    def test_similar_unknown_title(self):
        with self.assertRaises(LookupError):
            self.system.recommend_similar("Not A Movie")

    def test_for_user_follows_taste(self):
        self.system.rate_movie("Inception", 10)
        self.system.rate_movie("The Matrix", 9)
        self.system.rate_movie("Forrest Gump", 2)
        recs = self.system.recommend_for_user(3)
        rated = {"Inception", "The Matrix", "Forrest Gump"}
        self.assertTrue(all(m.title not in rated for m, _, _ in recs))
        for movie, _, reason in recs:
            self.assertIn("sci-fi", {g.lower() for g in movie.genres})
            self.assertIn("Sci-Fi", reason)

    def test_for_user_without_ratings_falls_back_to_top_rated(self):
        recs = self.system.recommend_for_user(2)
        self.assertEqual([m.title for m, _, _ in recs],
                         ["The Shawshank Redemption", "The Godfather"])

    def test_save_and_load_round_trip(self):
        self.system.rate_movie("Alien", 8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "db.json")
            self.system.save(path)
            loaded = MovieRecommendationSystem.load(path)
        self.assertEqual(len(loaded.movies), len(self.system.movies))
        self.assertEqual(loaded.user_ratings, {"alien": 8.0})
        self.assertEqual(loaded.get_movie("Alien").director, "Ridley Scott")


if __name__ == "__main__":
    unittest.main()
