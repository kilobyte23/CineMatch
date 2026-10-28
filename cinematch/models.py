"""Four recommenders behind one interface, from a baseline to matrix factorisation.

Every model implements `fit(train, n_users, n_items)` and `scores(user)`, which
returns a score per item (higher = recommend sooner). `recommend` then drops
items the user has already rated.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .data import Dataset, Ratings

LIKED = 4.0  # ratings at or above this count as "liked" (relevant) in evaluation and kNN
LIKE_THRESHOLD = 3.0  # ALS: ratings at or above this are positive preference


class Recommender:
    name = "base"

    def fit(self, train: Ratings, n_users: int, n_items: int) -> "Recommender":
        self.n_users, self.n_items = n_users, n_items
        self.seen: list[set[int]] = [set() for _ in range(n_users)]
        for u, i in zip(train.users.tolist(), train.items.tolist()):
            self.seen[u].add(i)
        self._fit(train)
        return self

    def _fit(self, train: Ratings) -> None:
        raise NotImplementedError

    def scores(self, user: int) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user: int, k: int = 10) -> list[int]:
        s = self.scores(user).astype(np.float64, copy=True)
        if self.seen[user]:
            s[list(self.seen[user])] = -np.inf
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return [int(i) for i in top[np.argsort(-s[top])] if np.isfinite(s[i])]


class Popularity(Recommender):
    """Most-liked movies overall. Everyone gets the same list; the bar to beat."""

    name = "Popularity"

    def _fit(self, train: Ratings) -> None:
        liked = train.values >= LIKED
        self._scores = np.bincount(train.items[liked], minlength=self.n_items).astype(np.float32)

    def scores(self, user: int) -> np.ndarray:
        return self._scores


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


_TOKEN = re.compile(r"[a-z0-9]+")


def item_documents(dataset: Dataset) -> list[list[str]]:
    """Tokens describing each movie: genres (weighted up), decade, and user tags."""
    docs = []
    for movie, tags in zip(dataset.movies, dataset.tags or [[] for _ in dataset.movies]):
        tokens = [f"genre_{g.lower().replace('-', '_')}" for g in movie.genres] * 2
        if movie.year:
            tokens.append(f"decade_{movie.year // 10 * 10}")
        for tag in tags:
            tokens.extend(_TOKEN.findall(tag))
        docs.append(tokens)
    return docs


def tfidf(docs: list[list[str]]) -> np.ndarray:
    """L2-normalised TF-IDF matrix (items x vocabulary)."""
    vocab: dict[str, int] = {}
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    for term, count in df.items():
        if count >= 2:  # drop one-off tags
            vocab[term] = len(vocab)
    matrix = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    n = len(docs)
    for row, doc in enumerate(docs):
        for term, tf in Counter(doc).items():
            col = vocab.get(term)
            if col is not None:
                matrix[row, col] = (1 + math.log(tf)) * math.log((1 + n) / (1 + df[term]))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


class ContentBased(Recommender):
    """Builds a taste vector from what you rated (above or below your average) and
    matches it against each movie's genres, era and tags."""

    name = "Content (TF-IDF)"

    def __init__(self, item_vectors: np.ndarray, popularity_weight: float = 1.0):
        self.item_vectors = item_vectors
        self.popularity_weight = popularity_weight

    def _fit(self, train: Ratings) -> None:
        dims = self.item_vectors.shape[1]
        self.profiles = np.zeros((self.n_users, dims), dtype=np.float32)
        sums = np.bincount(train.users, weights=train.values, minlength=self.n_users)
        counts = np.maximum(np.bincount(train.users, minlength=self.n_users), 1)
        means = sums / counts
        weights = (train.values - means[train.users]).astype(np.float32)
        np.add.at(self.profiles, train.users, self.item_vectors[train.items] * weights[:, None])
        # Pure content matching favours obscure films that happen to share tags;
        # a light popularity prior keeps recommendations to films people watch.
        self.prior = _zscore(np.log1p(np.bincount(train.items, minlength=self.n_items).astype(np.float32)))

    def scores(self, user: int) -> np.ndarray:
        return _zscore(self.item_vectors @ self.profiles[user]) + self.popularity_weight * self.prior

    def similar(self, item: int, k: int = 10) -> list[int]:
        sims = self.item_vectors @ self.item_vectors[item]
        sims[item] = -np.inf
        return [int(i) for i in np.argsort(-sims)[:k]]


class ItemKNN(Recommender):
    """"People who liked X also liked Y": item-item cosine similarity over who liked what."""

    name = "Item-kNN"

    def __init__(self, neighbours: int = 2000):
        self.neighbours = neighbours

    def _fit(self, train: Ratings) -> None:
        liked = train.values >= LIKED
        self.likes = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.likes[train.users[liked], train.items[liked]] = 1.0
        norms = np.sqrt(self.likes.sum(axis=0))
        co = self.likes.T @ self.likes
        sim = co / np.maximum(np.outer(norms, norms), 1e-9)
        np.fill_diagonal(sim, 0.0)
        # Keep only each item's strongest neighbours: less noise, sharper recommendations.
        if self.neighbours < self.n_items:
            cutoff = np.partition(sim, -self.neighbours, axis=1)[:, -self.neighbours][:, None]
            sim = np.where(sim >= cutoff, sim, 0.0)
        self.sim = sim.astype(np.float32)

    def scores(self, user: int) -> np.ndarray:
        return self.likes[user] @ self.sim

    def similar(self, item: int, k: int = 10) -> list[int]:
        return [int(i) for i in np.argsort(-self.sim[item])[:k] if self.sim[item, i] > 0]


class ImplicitALS(Recommender):
    """Matrix factorisation for implicit feedback (Hu, Koren & Volinsky, 2008).

    Ratings of 3 or more mean "wants films like this" (preference 1); lower
    ratings mean "watched it and didn't like it" (preference 0). Either way the
    rating sets how confident we are, while unrated films get a weak "probably
    not" (preference 0, confidence 1). Alternating least squares learns a small
    taste vector per user and per movie; a recommendation is their dot product.
    """

    name = "Implicit ALS"

    # Defaults chosen by NDCG@10 on a validation split carved out of the training data.
    def __init__(self, factors: int = 32, regularization: float = 30.0, alpha: float = 2.0,
                 iterations: int = 10, seed: int = 0):
        self.factors, self.reg, self.alpha, self.iterations, self.seed = factors, regularization, alpha, iterations, seed

    def _fit(self, train: Ratings) -> None:
        rng = np.random.default_rng(self.seed)
        f = self.factors
        self.user_factors = (rng.standard_normal((self.n_users, f)) * 0.01).astype(np.float64)
        self.item_factors = (rng.standard_normal((self.n_items, f)) * 0.01).astype(np.float64)
        pref, conf = self._signal(train.values)
        by_user = self._group(train.users, train.items, conf, pref, self.n_users)
        by_item = self._group(train.items, train.users, conf, pref, self.n_items)
        for _ in range(self.iterations):
            self.user_factors = self._solve(self.item_factors, by_user)
            self.item_factors = self._solve(self.user_factors, by_item)

    def _signal(self, ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Preference (1 = liked) and confidence for explicit ratings on a 0.5-5 scale."""
        ratings = np.asarray(ratings, dtype=np.float64)
        pref = (ratings >= LIKE_THRESHOLD).astype(np.float64)
        strength = np.where(pref > 0, ratings / 5.0, (LIKE_THRESHOLD - ratings) / LIKE_THRESHOLD)
        return pref, 1.0 + self.alpha * strength

    @staticmethod
    def _group(keys, others, conf, pref, n):
        order = np.argsort(keys, kind="stable")
        keys, others, conf, pref = keys[order], others[order], conf[order], pref[order]
        starts = np.searchsorted(keys, np.arange(n + 1))
        return [(others[s:e], conf[s:e], pref[s:e]) for s, e in zip(starts[:-1], starts[1:])]

    def _solve(self, fixed: np.ndarray, groups) -> np.ndarray:
        f = fixed.shape[1]
        gram = fixed.T @ fixed
        eye = self.reg * np.eye(f)
        out = np.zeros((len(groups), f))
        for row, (idx, conf, pref) in enumerate(groups):
            if len(idx) == 0:
                continue
            y = fixed[idx]
            # (YᵀY + Yᵀ(C−I)Y + λI) x = YᵀCp
            a = gram + (y.T * (conf - 1.0)) @ y + eye
            b = y.T @ (conf * pref)
            out[row] = np.linalg.solve(a, b)
        return out

    def scores(self, user: int) -> np.ndarray:
        return self.item_factors @ self.user_factors[user]

    def fold_in(self, items: list[int], ratings: list[float]) -> np.ndarray:
        """Taste vector for a brand-new user from a handful of ratings, without retraining."""
        pref, conf = self._signal(np.asarray(ratings))
        y = self.item_factors[items]
        a = self.item_factors.T @ self.item_factors + (y.T * (conf - 1.0)) @ y + self.reg * np.eye(self.factors)
        return np.linalg.solve(a, y.T @ (conf * pref))

    def similar(self, item: int, k: int = 10) -> list[int]:
        v = self.item_factors
        norms = np.linalg.norm(v, axis=1) + 1e-9
        sims = (v @ v[item]) / (norms * norms[item])
        sims[item] = -np.inf
        return [int(i) for i in np.argsort(-sims)[:k]]


class Hybrid(Recommender):
    """Blends ALS (broad taste) with item-kNN (close neighbours) on standardised scores."""

    name = "Hybrid (ALS + kNN)"

    def __init__(self, als_weight: float = 0.3, als: ImplicitALS | None = None, knn: ItemKNN | None = None):
        self.als_weight = als_weight
        self.als = als or ImplicitALS()
        self.knn = knn or ItemKNN()

    def _fit(self, train: Ratings) -> None:
        self.als.fit(train, self.n_users, self.n_items)
        self.knn.fit(train, self.n_users, self.n_items)

    def scores(self, user: int) -> np.ndarray:
        return self.als_weight * _zscore(self.als.scores(user)) + (1 - self.als_weight) * _zscore(self.knn.scores(user))
