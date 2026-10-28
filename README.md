# CineMatch

A command-line movie recommendation system in Python, using only the standard library. It ships with 30 well-known films and recommends movies in two ways:

- **More like this**: name a movie and get the most similar ones. Similarity is based on shared genres, same director and release era, with rating as a tie-breaker.
- **For you**: rate movies you've seen (0–10). CineMatch builds a genre taste profile from your ratings, where genres you rated highly count for and genres you rated low count against. It then ranks the movies you haven't seen and says why it picked each one.

It also lets you manage the catalogue:

- Add movies with one or more genres, plus an optional year and director
- Search by title, genre or director
- List the top **N** movies, optionally within one genre
- Delete a movie
- Save the catalogue and your ratings to `cinematch.json`, which loads again on the next start

## Usage

```bash
python CineMatch.py
```

```
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
 0. Exit
```

Example: rate *Inception* 10 and *The Matrix* 9, then choose **6**:

```
  1. Star Wars: Episode V - The Empire Strikes Back (1980, Irvin Kershner) | Action, Adventure, Sci-Fi | 8.7
     because you like Action, Sci-Fi
  2. Mad Max: Fury Road (2015, George Miller) | Action, Adventure, Sci-Fi | 8.1
     because you like Action, Sci-Fi
```

With no ratings yet, "for you" falls back to the highest-rated movies.

## Using it as a module

```python
from CineMatch import MovieRecommendationSystem

system = MovieRecommendationSystem.with_default_catalogue()
system.add_movie("Heat", "Crime, Thriller", 8.3, year=1995, director="Michael Mann")

system.recommend_similar("Inception", 5)      # [(Movie, score), ...]
system.rate_movie("Alien", 9)
system.recommend_for_user(5)                  # [(Movie, score, reason), ...]
system.recommend_top_movies(3, genre="Drama") # [Movie, ...]
system.save("cinematch.json")
```

Methods return data rather than printing it. The CLI handles the printing.

## Tests

```bash
python -m unittest -v
```
