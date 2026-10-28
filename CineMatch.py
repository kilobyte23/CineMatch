"""CineMatch: a small content-based movie recommendation system.

Movies carry one or more genres, a year, a director and an average rating.
Recommendations come from two places:

- "More like this": movies similar to one you name (shared genres, same
  director, release year), with rating as a tie-breaker.
- "For you": you rate movies you've seen; CineMatch builds a genre taste
  profile from those ratings and ranks the movies you haven't rated.

Everything is standard library only. The catalogue and your ratings can be
saved to and loaded from a JSON file.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field

DEFAULT_DB_PATH = "cinematch.json"

# How much each signal counts toward "more like this" similarity.
GENRE_WEIGHT = 0.6
DIRECTOR_WEIGHT = 0.2
ERA_WEIGHT = 0.1
RATING_WEIGHT = 0.1
ERA_SPAN_YEARS = 30

# User ratings at or above this count as "liked" when building a profile.
NEUTRAL_USER_RATING = 5.5
# Share of a "for you" score that comes from taste match vs. overall rating.
TASTE_WEIGHT = 0.75


@dataclass
class Movie:
    title: str
    genres: list
    rating: float
    year: int = None
    director: str = None

    @property
    def genre(self):
        """Genres as a single display string (kept for backward compatibility)."""
        return ", ".join(self.genres)

    def __str__(self):
        extra = []
        if self.year:
            extra.append(str(self.year))
        if self.director:
            extra.append(self.director)
        suffix = f" ({', '.join(extra)})" if extra else ""
        return f"{self.title}{suffix} | {self.genre} | {self.rating:.1f}"


def _parse_genres(genre):
    if isinstance(genre, str):
        parts = genre.replace("/", ",").split(",")
    else:
        parts = list(genre)
    genres = []
    for part in parts:
        name = part.strip()
        if name and name.lower() not in (g.lower() for g in genres):
            genres.append(name)
    if not genres:
        raise ValueError("A movie needs at least one genre.")
    return genres


def _genre_set(movie):
    return {g.lower() for g in movie.genres}


@dataclass
class MovieRecommendationSystem:
    movies: list = field(default_factory=list)
    user_ratings: dict = field(default_factory=dict)  # lower-cased title -> 0..10

    # ---- catalogue -------------------------------------------------------

    def add_movie(self, title, genre, rating, year=None, director=None):
        """Add a movie. `genre` may be a string ("Sci-Fi, Action") or a list."""
        title = title.strip()
        if not title:
            raise ValueError("Title cannot be empty.")
        rating = float(rating)
        if not 0 <= rating <= 10:
            raise ValueError("Rating must be between 0 and 10.")
        if self.get_movie(title):
            raise ValueError(f"'{title}' is already in the catalogue.")
        movie = Movie(title, _parse_genres(genre), rating,
                      int(year) if year else None,
                      director.strip() if director else None)
        self.movies.append(movie)
        return movie

    def get_movie(self, title):
        key = title.strip().lower()
        return next((m for m in self.movies if m.title.lower() == key), None)

    def find_movie(self, title):
        """Exact match first, then a unique partial match."""
        movie = self.get_movie(title)
        if movie:
            return movie
        key = title.strip().lower()
        matches = [m for m in self.movies if key in m.title.lower()]
        return matches[0] if len(matches) == 1 else None

    def delete_movie(self, title):
        movie = self.get_movie(title)
        if not movie:
            return False
        self.movies.remove(movie)
        self.user_ratings.pop(movie.title.lower(), None)
        return True

    def search_movie(self, keyword):
        """Case-insensitive search over title, genres and director."""
        key = keyword.strip().lower()
        return [
            m for m in self.movies
            if key in m.title.lower()
            or any(key in g.lower() for g in m.genres)
            or (m.director and key in m.director.lower())
        ]

    def genres(self):
        seen = {}
        for movie in self.movies:
            for g in movie.genres:
                seen.setdefault(g.lower(), g)
        return sorted(seen.values(), key=str.lower)

    # ---- recommendations ------------------------------------------------

    def recommend_top_movies(self, n, genre=None):
        """Top `n` movies by rating, optionally limited to one genre."""
        pool = self.movies
        if genre:
            pool = [m for m in pool if genre.strip().lower() in _genre_set(m)]
        return sorted(pool, key=lambda m: m.rating, reverse=True)[:max(0, n)]

    def similarity(self, a, b):
        """Similarity between two movies in [0, 1]."""
        ga, gb = _genre_set(a), _genre_set(b)
        score = GENRE_WEIGHT * len(ga & gb) / len(ga | gb)
        if a.director and b.director and a.director.lower() == b.director.lower():
            score += DIRECTOR_WEIGHT
        if a.year and b.year:
            score += ERA_WEIGHT * max(0.0, 1 - abs(a.year - b.year) / ERA_SPAN_YEARS)
        return score

    def recommend_similar(self, title, n=5):
        """Movies most like `title`. Returns a list of (movie, score)."""
        base = self.find_movie(title)
        if not base:
            raise LookupError(f"No movie found matching '{title}'.")
        scored = []
        for movie in self.movies:
            if movie is base:
                continue
            content = self.similarity(base, movie)
            if content <= 0:
                continue
            scored.append((movie, content + RATING_WEIGHT * movie.rating / 10))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:max(0, n)]

    # ---- personal ratings -----------------------------------------------

    def rate_movie(self, title, rating):
        movie = self.find_movie(title)
        if not movie:
            raise LookupError(f"No movie found matching '{title}'.")
        rating = float(rating)
        if not 0 <= rating <= 10:
            raise ValueError("Rating must be between 0 and 10.")
        self.user_ratings[movie.title.lower()] = rating
        return movie

    def rated_movies(self):
        return [(m, self.user_ratings[m.title.lower()])
                for m in self.movies if m.title.lower() in self.user_ratings]

    def taste_profile(self):
        """Genre -> weight. Positive means you like it, negative means you don't."""
        profile = {}
        for movie, rating in self.rated_movies():
            weight = (rating - NEUTRAL_USER_RATING) / len(movie.genres)
            for g in _genre_set(movie):
                profile[g] = profile.get(g, 0.0) + weight
        return profile

    def recommend_for_user(self, n=5):
        """Unrated movies ranked by your taste profile.

        Returns a list of (movie, score, reason). With no ratings yet, falls
        back to the highest-rated movies.
        """
        profile = self.taste_profile()
        unrated = [m for m in self.movies if m.title.lower() not in self.user_ratings]
        if not profile or not any(v > 0 for v in profile.values()):
            top = sorted(unrated, key=lambda m: m.rating, reverse=True)[:max(0, n)]
            return [(m, m.rating / 10, "highly rated overall") for m in top]

        norm = math.sqrt(sum(v * v for v in profile.values()))
        scored = []
        for movie in unrated:
            genres = _genre_set(movie)
            # Cosine similarity between the profile and the movie's genre vector.
            taste = sum(profile.get(g, 0.0) for g in genres) / (norm * math.sqrt(len(genres)))
            score = TASTE_WEIGHT * taste + (1 - TASTE_WEIGHT) * movie.rating / 10
            liked = sorted((g for g in movie.genres if profile.get(g.lower(), 0) > 0),
                           key=lambda g: -profile[g.lower()])
            reason = f"you like {', '.join(liked)}" if liked else "highly rated overall"
            scored.append((movie, score, reason))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:max(0, n)]

    # ---- persistence ----------------------------------------------------

    def save(self, path=DEFAULT_DB_PATH):
        data = {"movies": [asdict(m) for m in self.movies],
                "user_ratings": self.user_ratings}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path=DEFAULT_DB_PATH):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        system = cls()
        for m in data.get("movies", []):
            system.add_movie(m["title"], m["genres"], m["rating"], m.get("year"), m.get("director"))
        system.user_ratings = {k.lower(): float(v) for k, v in data.get("user_ratings", {}).items()}
        return system

    @classmethod
    def with_default_catalogue(cls):
        system = cls()
        for title, genres, rating, year, director in DEFAULT_CATALOGUE:
            system.add_movie(title, genres, rating, year, director)
        return system


DEFAULT_CATALOGUE = [
    ("The Shawshank Redemption", "Drama", 9.3, 1994, "Frank Darabont"),
    ("The Godfather", "Crime, Drama", 9.2, 1972, "Francis Ford Coppola"),
    ("The Dark Knight", "Action, Crime, Drama", 9.0, 2008, "Christopher Nolan"),
    ("Pulp Fiction", "Crime, Drama", 8.9, 1994, "Quentin Tarantino"),
    ("The Lord of the Rings: The Return of the King", "Adventure, Fantasy, Drama", 8.9, 2003, "Peter Jackson"),
    ("Forrest Gump", "Drama, Romance", 8.8, 1994, "Robert Zemeckis"),
    ("Inception", "Action, Sci-Fi, Thriller", 8.8, 2010, "Christopher Nolan"),
    ("Fight Club", "Drama, Thriller", 8.8, 1999, "David Fincher"),
    ("The Matrix", "Action, Sci-Fi", 8.7, 1999, "The Wachowskis"),
    ("Goodfellas", "Crime, Drama", 8.7, 1990, "Martin Scorsese"),
    ("The Lord of the Rings: The Fellowship of the Ring", "Adventure, Fantasy, Drama", 8.8, 2001, "Peter Jackson"),
    ("The Lord of the Rings: The Two Towers", "Adventure, Fantasy, Drama", 8.7, 2002, "Peter Jackson"),
    ("Star Wars: Episode V - The Empire Strikes Back", "Action, Adventure, Sci-Fi", 8.7, 1980, "Irvin Kershner"),
    ("Interstellar", "Adventure, Drama, Sci-Fi", 8.6, 2014, "Christopher Nolan"),
    ("City of God", "Crime, Drama", 8.6, 2002, "Fernando Meirelles"),
    ("Spirited Away", "Animation, Adventure, Fantasy", 8.6, 2001, "Hayao Miyazaki"),
    ("Saving Private Ryan", "Drama, War", 8.6, 1998, "Steven Spielberg"),
    ("The Green Mile", "Crime, Drama, Fantasy", 8.6, 1999, "Frank Darabont"),
    ("Gladiator", "Action, Adventure, Drama", 8.5, 2000, "Ridley Scott"),
    ("The Lion King", "Animation, Adventure, Drama", 8.5, 1994, "Roger Allers"),
    ("Se7en", "Crime, Drama, Mystery", 8.6, 1995, "David Fincher"),
    ("Alien", "Horror, Sci-Fi", 8.5, 1979, "Ridley Scott"),
    ("Back to the Future", "Adventure, Comedy, Sci-Fi", 8.5, 1985, "Robert Zemeckis"),
    ("Parasite", "Drama, Thriller", 8.5, 2019, "Bong Joon Ho"),
    ("Whiplash", "Drama, Music", 8.5, 2014, "Damien Chazelle"),
    ("The Prestige", "Drama, Mystery, Sci-Fi", 8.5, 2006, "Christopher Nolan"),
    ("Toy Story", "Animation, Adventure, Comedy", 8.3, 1995, "John Lasseter"),
    ("Mad Max: Fury Road", "Action, Adventure, Sci-Fi", 8.1, 2015, "George Miller"),
    ("Amélie", "Comedy, Romance", 8.3, 2001, "Jean-Pierre Jeunet"),
    ("The Grand Budapest Hotel", "Adventure, Comedy, Crime", 8.1, 2014, "Wes Anderson"),
]


# ---- command-line interface ---------------------------------------------

def _ask(prompt, cast=str, default=None):
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            return cast(raw)
        except ValueError:
            print("  Invalid value, try again.")


def _print_movies(movies):
    if not movies:
        print("  (nothing found)")
    for i, movie in enumerate(movies, 1):
        print(f"  {i:>2}. {movie}")


MENU = """
CineMatch
 1. Add a movie
 2. Search by title, genre or director
 3. Top N movies (optionally by genre)
 4. Movies similar to one you like
 5. Rate a movie you've seen
 6. Recommendations for you
 7. Show your ratings and taste profile
 8. Delete a movie
 9. Save
 0. Exit"""


def main(db_path=DEFAULT_DB_PATH):
    if os.path.exists(db_path):
        system = MovieRecommendationSystem.load(db_path)
        print(f"Loaded {len(system.movies)} movies from {db_path}.")
    else:
        system = MovieRecommendationSystem.with_default_catalogue()

    while True:
        print(MENU)
        choice = input("Choose: ").strip()
        try:
            if choice == "1":
                title = _ask("Title: ")
                genre = _ask("Genres (comma-separated): ")
                rating = _ask("Rating (0-10): ", float)
                year = _ask("Year (optional): ", int, default="")
                director = _ask("Director (optional): ", default="")
                print(f"Added: {system.add_movie(title, genre, rating, year or None, director or None)}")
            elif choice == "2":
                _print_movies(system.search_movie(_ask("Search: ")))
            elif choice == "3":
                n = _ask("How many? ", int)
                print(f"Genres: {', '.join(system.genres())}")
                genre = _ask("Genre (blank for all): ", default="")
                _print_movies(system.recommend_top_movies(n, genre or None))
            elif choice == "4":
                title = _ask("Movie you like: ")
                for i, (movie, score) in enumerate(system.recommend_similar(title, 5), 1):
                    print(f"  {i}. {movie}  [match {score:.2f}]")
            elif choice == "5":
                movie = system.rate_movie(_ask("Movie: "), _ask("Your rating (0-10): ", float))
                print(f"Rated {movie.title}.")
            elif choice == "6":
                n = _ask("How many? ", int, default=5)
                for i, (movie, score, reason) in enumerate(system.recommend_for_user(n), 1):
                    print(f"  {i}. {movie}\n     because {reason}")
            elif choice == "7":
                rated = system.rated_movies()
                if not rated:
                    print("  You haven't rated anything yet.")
                for movie, rating in rated:
                    print(f"  {rating:>4.1f}  {movie.title}")
                profile = sorted(system.taste_profile().items(), key=lambda kv: -kv[1])
                if profile:
                    print("  Taste: " + ", ".join(f"{g} {w:+.1f}" for g, w in profile))
            elif choice == "8":
                title = _ask("Title to delete: ")
                print("Deleted." if system.delete_movie(title) else f"No movie titled '{title}'.")
            elif choice == "9":
                system.save(db_path)
                print(f"Saved to {db_path}.")
            elif choice == "0":
                break
            else:
                print("Pick a number from the menu.")
        except (ValueError, LookupError) as e:
            print(f"  {e}")


if __name__ == "__main__":
    main()
