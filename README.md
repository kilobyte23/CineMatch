# CineMatch

A simple command-line movie recommendation system in Python. It starts with a catalogue of 20 well-known films and lets you manage the list and get top-rated picks.

## Features

- Add a movie (title, genre, rating 0–10)
- Search by title or genre (case-insensitive)
- Recommend the top **N** movies by rating
- Delete a movie by title

## Usage

No dependencies are required beyond Python 3.

```bash
python CineMatch.py
```

Then pick an option from the menu:

```
Movie Recommendation System
1. Add a new movie
2. Search for movies by title or genre
3. Recommend top N movies based on rating
4. Delete a movie
5. Exit
```

## Using it as a module

```python
from CineMatch import MovieRecommendationSystem

system = MovieRecommendationSystem()
system.add_movie("Inception", "Sci-Fi", 8.8)
system.recommend_top_movies(5)
```
