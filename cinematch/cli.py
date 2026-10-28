"""Command-line interface: evaluate models, recommend for a user, find similar films, onboard a new user."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .data import Dataset, load, temporal_split
from .metrics import evaluate, lift, markdown_table
from .models import ContentBased, Hybrid, ImplicitALS, ItemKNN, Popularity, item_documents, tfidf

MODELS = ("hybrid", "als", "knn", "content", "popularity")


def build(name: str, dataset: Dataset):
    if name == "popularity":
        return Popularity()
    if name == "content":
        return ContentBased(tfidf(item_documents(dataset)))
    if name == "knn":
        return ItemKNN()
    if name == "als":
        return ImplicitALS()
    return Hybrid()


def cmd_evaluate(args, dataset: Dataset) -> None:
    train, test = temporal_split(dataset.ratings, test_fraction=args.test_fraction)
    print(f"{dataset.n_users} users, {dataset.n_items} movies, {len(dataset.ratings):,} ratings")
    print(f"Temporal split: each user's latest {args.test_fraction:.0%} of ratings held out "
          f"({len(train):,} train / {len(test):,} test). Relevant = rated 4 or more.\n")
    reports = []
    for name in ("popularity", "content", "knn", "als", "hybrid"):
        print(f"  fitting {name}...", file=sys.stderr)
        reports.append(evaluate(build(name, dataset), train, test, dataset.n_users, dataset.n_items, args.k))
    print(markdown_table(reports, args.k))
    print(f"\nBest model: {lift(reports):.2f}x the popularity baseline's NDCG@{args.k}.")


def because(knn: ItemKNN, user_likes: np.ndarray, item: int, dataset: Dataset) -> str | None:
    """The liked film that contributes most to recommending `item`."""
    contrib = user_likes * knn.sim[:, item]
    best = int(np.argmax(contrib))
    return dataset.label(best) if contrib[best] > 0 else None


def print_recs(dataset: Dataset, recs: list[int], reasons: list[str | None] | None = None) -> None:
    for rank, item in enumerate(recs, 1):
        genres = ", ".join(dataset.movies[item].genres[:3])
        print(f"  {rank:>2}. {dataset.label(item)}  [{genres}]")
        if reasons and reasons[rank - 1]:
            print(f"      because you liked {reasons[rank - 1]}")


def cmd_recommend(args, dataset: Dataset) -> None:
    try:
        user = dataset.user_ids.index(args.user)
    except ValueError:
        sys.exit(f"No user {args.user}. MovieLens user ids run from {min(dataset.user_ids)} to {max(dataset.user_ids)}.")
    r = dataset.ratings
    mine = np.flatnonzero(r.users == user)
    favourites = mine[np.argsort(-r.values[mine], kind="stable")][:5]
    print(f"User {args.user} rated {len(mine)} movies. Favourites:")
    for idx in favourites:
        print(f"  {r.values[idx]:.1f}  {dataset.label(int(r.items[idx]))}")

    model = build(args.model, dataset).fit(r, dataset.n_users, dataset.n_items)
    recs = model.recommend(user, args.n)
    knn = model.knn if isinstance(model, Hybrid) else model if isinstance(model, ItemKNN) else None
    reasons = [because(knn, knn.likes[user], i, dataset) for i in recs] if knn else None
    print(f"\nRecommended by {model.name}:")
    print_recs(dataset, recs, reasons)


def pick_movie(dataset: Dataset, query: str) -> int:
    hits = dataset.find(query)
    if not hits:
        sys.exit(f'No movie matching "{query}".')
    return hits[0]


def cmd_similar(args, dataset: Dataset) -> None:
    item = pick_movie(dataset, args.title)
    model = build(args.model, dataset)
    if not hasattr(model, "similar"):
        sys.exit(f"{args.model} has no notion of similarity; use knn, als or content.")
    model.fit(dataset.ratings, dataset.n_users, dataset.n_items)
    print(f"More like {dataset.label(item)} ({model.name}):")
    print_recs(dataset, model.similar(item, args.n))


def onboarding_candidates(dataset: Dataset, count: int = 15) -> list[int]:
    """Popular films spread across genres, so a few ratings say a lot about taste."""
    popularity = np.bincount(dataset.ratings.items, minlength=dataset.n_items)
    chosen, genres_seen = [], set()
    for item in np.argsort(-popularity):
        main = dataset.movies[item].genres[0] if dataset.movies[item].genres else ""
        if main in genres_seen and len(genres_seen) < 8:
            continue
        chosen.append(int(item))
        genres_seen.add(main)
        if len(chosen) == count:
            break
    return chosen


def cmd_onboard(args, dataset: Dataset) -> None:
    print("Rate a few movies (0.5-5), press Enter to skip, or type / to search for a title.\n")
    rated: dict[int, float] = {}
    for item in onboarding_candidates(dataset):
        while True:
            answer = input(f"  {dataset.label(item)}: ").strip()
            if answer.startswith("/"):
                found = pick_movie(dataset, answer[1:])
                value = input(f"    {dataset.label(found)}: ").strip()
                if value:
                    rated[found] = float(value)
                continue
            if not answer:
                break
            try:
                value = float(answer)
            except ValueError:
                print("    Enter a number from 0.5 to 5.")
                continue
            if 0.5 <= value <= 5:
                rated[item] = value
                break
            print("    Enter a number from 0.5 to 5.")
        if len(rated) >= args.ratings:
            break
    if len(rated) < 3:
        sys.exit("Rate at least 3 movies to get recommendations.")

    print("\nLearning your taste...")
    als = ImplicitALS().fit(dataset.ratings, dataset.n_users, dataset.n_items)
    knn = ItemKNN().fit(dataset.ratings, dataset.n_users, dataset.n_items)
    items, values = list(rated), list(rated.values())
    # New user, no retraining: fold their ratings into the ALS item space and blend with kNN.
    taste = als.fold_in(items, values)
    likes = np.zeros(dataset.n_items, dtype=np.float32)
    likes[[i for i, v in rated.items() if v >= 4]] = 1.0
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    scores = 0.3 * z(als.item_factors @ taste) + 0.7 * z(likes @ knn.sim)
    scores[items] = -np.inf
    recs = [int(i) for i in np.argsort(-scores)[: args.n]]
    print("\nYou might like:")
    print_recs(dataset, recs, [because(knn, likes, i, dataset) for i in recs])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cinematch", description="Movie recommendations on MovieLens.")
    parser.add_argument("--data", type=Path, help="folder with MovieLens CSVs (default: download to ~/.cache/cinematch)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("evaluate", help="compare all models offline")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.set_defaults(run=cmd_evaluate)

    p = sub.add_parser("recommend", help="top picks for an existing MovieLens user")
    p.add_argument("--user", type=int, required=True)
    p.add_argument("--model", choices=MODELS, default="hybrid")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(run=cmd_recommend)

    p = sub.add_parser("similar", help='movies like a given title, e.g. "toy story"')
    p.add_argument("title")
    p.add_argument("--model", choices=("knn", "als", "content"), default="knn")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(run=cmd_similar)

    p = sub.add_parser("onboard", help="rate a few films, get recommendations (cold start)")
    p.add_argument("--ratings", type=int, default=8, help="stop asking after this many ratings")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(run=cmd_onboard)

    args = parser.parse_args(argv)
    dataset = load(args.data) if args.data else load()
    args.run(args, dataset)


if __name__ == "__main__":
    main()
