"""Top-k ranking metrics for offline evaluation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


from .data import Ratings
from .models import LIKED, Recommender


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return sum(1 for i in recommended[:k] if i in relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return sum(1 for i in recommended[:k] if i in relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Rewards putting relevant movies near the top of the list."""
    dcg = sum(1 / math.log2(rank + 2) for rank, i in enumerate(recommended[:k]) if i in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


@dataclass
class Report:
    model: str
    precision: float
    recall: float
    ndcg: float
    coverage: float  # share of the catalogue that appears in anyone's top-k
    users: int
    fit_seconds: float


def evaluate(model: Recommender, train: Ratings, test: Ratings, n_users: int, n_items: int, k: int = 10) -> Report:
    """Fits on `train`, then checks each user's top-k against movies they went on to like in `test`."""
    started = time.perf_counter()
    model.fit(train, n_users, n_items)
    fit_seconds = time.perf_counter() - started

    relevant: dict[int, set[int]] = {}
    for u, i, r in zip(test.users.tolist(), test.items.tolist(), test.values.tolist()):
        if r >= LIKED:
            relevant.setdefault(u, set()).add(i)

    p = r = n = 0.0
    shown: set[int] = set()
    for user, liked in relevant.items():
        recs = model.recommend(user, k)
        shown.update(recs)
        p += precision_at_k(recs, liked, k)
        r += recall_at_k(recs, liked, k)
        n += ndcg_at_k(recs, liked, k)
    count = max(len(relevant), 1)
    return Report(model.name, p / count, r / count, n / count, len(shown) / n_items, len(relevant), fit_seconds)


def markdown_table(reports: list[Report], k: int) -> str:
    lines = [
        f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | Catalogue coverage | Fit time |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    best = max(reports, key=lambda rep: rep.ndcg)
    for rep in reports:
        name = f"**{rep.model}**" if rep is best else rep.model
        lines.append(
            f"| {name} | {rep.precision:.3f} | {rep.recall:.3f} | {rep.ndcg:.3f} | {rep.coverage:.1%} | {rep.fit_seconds:.1f}s |"
        )
    return "\n".join(lines)


def lift(reports: list[Report]) -> float:
    """NDCG of the best model divided by the popularity baseline's (assumed first)."""
    base = reports[0].ndcg
    return max(rep.ndcg for rep in reports) / base if base else float("nan")


