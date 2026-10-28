"""Loading MovieLens into compact numpy arrays.

The dataset is downloaded on first use (it is not redistributed in this repo).
MovieLens is provided by GroupLens Research: https://grouplens.org/datasets/movielens/
"""

from __future__ import annotations

import csv
import io
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEFAULT_DATA_DIR = Path(os.environ.get("CINEMATCH_DATA", Path.home() / ".cache" / "cinematch"))
_YEAR = re.compile(r"\((\d{4})\)\s*$")


@dataclass
class Movie:
    title: str
    year: int | None
    genres: list[str]


@dataclass
class Ratings:
    """Explicit ratings as parallel arrays, with users and items re-indexed to 0..n-1."""

    users: np.ndarray       # int32
    items: np.ndarray       # int32
    values: np.ndarray      # float32, 0.5-5.0
    timestamps: np.ndarray  # int64

    def __len__(self) -> int:
        return len(self.users)

    def subset(self, mask: np.ndarray) -> "Ratings":
        return Ratings(self.users[mask], self.items[mask], self.values[mask], self.timestamps[mask])


@dataclass
class Dataset:
    movies: list[Movie]
    ratings: Ratings
    n_users: int
    tags: list[list[str]] = field(default_factory=list)  # per item, lower-cased user tags
    movie_ids: list[int] = field(default_factory=list)   # item index -> MovieLens movieId
    user_ids: list[int] = field(default_factory=list)    # user index -> MovieLens userId

    @property
    def n_items(self) -> int:
        return len(self.movies)

    def find(self, query: str) -> list[int]:
        """Item indexes whose title contains `query` (case-insensitive), most-rated first."""
        q = query.strip().lower()
        hits = [i for i, m in enumerate(self.movies) if q in m.title.lower()]
        counts = np.bincount(self.ratings.items, minlength=self.n_items)
        return sorted(hits, key=lambda i: -counts[i])

    def label(self, item: int) -> str:
        m = self.movies[item]
        return f"{m.title} ({m.year})" if m.year else m.title


def download(data_dir: Path = DEFAULT_DATA_DIR, url: str = DATA_URL) -> Path:
    """Downloads and unpacks MovieLens if needed. Returns the folder with the CSVs."""
    target = data_dir / "ml-latest-small"
    if (target / "ratings.csv").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens (about 1 MB) from {url} ...")
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            if member.endswith(".csv") or member.endswith("README.txt"):
                archive.extract(member, data_dir)
    return target


def _split_title(raw: str) -> tuple[str, int | None]:
    raw = raw.strip()
    match = _YEAR.search(raw)
    if not match:
        return raw, None
    title = raw[: match.start()].strip()
    # MovieLens stores "Matrix, The"; show it the way people say it.
    for article in ("The", "A", "An"):
        if title.endswith(f", {article}"):
            title = f"{article} {title[: -len(article) - 2]}"
    return title, int(match.group(1))


def load(folder: Path | None = None) -> Dataset:
    """Loads MovieLens from `folder` (downloading it when no folder is given)."""
    folder = folder or download()
    movie_index: dict[int, int] = {}
    movies: list[Movie] = []
    with open(folder / "movies.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title, year = _split_title(row["title"])
            genres = [] if row["genres"] == "(no genres listed)" else row["genres"].split("|")
            movie_index[int(row["movieId"])] = len(movies)
            movies.append(Movie(title, year, genres))

    user_index: dict[int, int] = {}
    users, items, values, stamps = [], [], [], []
    with open(folder / "ratings.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = int(row["movieId"])
            if mid not in movie_index:
                continue
            uid = int(row["userId"])
            users.append(user_index.setdefault(uid, len(user_index)))
            items.append(movie_index[mid])
            values.append(float(row["rating"]))
            stamps.append(int(row["timestamp"]))

    tags: list[list[str]] = [[] for _ in movies]
    tags_path = folder / "tags.csv"
    if tags_path.exists():
        with open(tags_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mid = int(row["movieId"])
                if mid in movie_index:
                    tags[movie_index[mid]].append(row["tag"].strip().lower())

    ratings = Ratings(
        np.array(users, dtype=np.int32),
        np.array(items, dtype=np.int32),
        np.array(values, dtype=np.float32),
        np.array(stamps, dtype=np.int64),
    )
    movie_ids = [0] * len(movies)
    for mid, i in movie_index.items():
        movie_ids[i] = mid
    user_ids = [0] * len(user_index)
    for uid, i in user_index.items():
        user_ids[i] = uid
    return Dataset(movies, ratings, len(user_index), tags, movie_ids, user_ids)


def temporal_split(ratings: Ratings, test_fraction: float = 0.2, min_ratings: int = 5) -> tuple[Ratings, Ratings]:
    """Per user, holds out their most recent ratings as the test set.

    This mirrors real use (predict what someone watches next from what they
    watched before) and avoids leaking the future into training. Users with
    fewer than `min_ratings` ratings stay entirely in training.
    """
    test = np.zeros(len(ratings), dtype=bool)
    order = np.lexsort((ratings.timestamps, ratings.users))
    users_sorted = ratings.users[order]
    boundaries = np.flatnonzero(np.diff(users_sorted)) + 1
    for group in np.split(order, boundaries):
        if len(group) < min_ratings:
            continue
        n_test = max(1, int(round(len(group) * test_fraction)))
        test[group[-n_test:]] = True
    return ratings.subset(~test), ratings.subset(test)
