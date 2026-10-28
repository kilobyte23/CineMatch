"""Tests run on small synthetic data with a planted structure, so no download is needed."""

import csv

import numpy as np
import pytest

from cinematch.cli import main
from cinematch.data import Dataset, Movie, Ratings, load, temporal_split
from cinematch.metrics import evaluate, ndcg_at_k, precision_at_k, recall_at_k
from cinematch.models import ContentBased, Hybrid, ImplicitALS, ItemKNN, Popularity, item_documents, tfidf

N_USERS, N_ITEMS = 40, 20


def planted_ratings(seed: int = 0) -> Ratings:
    """Two taste clusters: users 0-19 love items 0-9 (sci-fi), users 20-39 love items 10-19 (romance).
    Each user rates 7 of their cluster's films highly and 2 of the other cluster's films low."""
    rng = np.random.default_rng(seed)
    users, items, values, stamps = [], [], [], []
    for u in range(N_USERS):
        mine = np.arange(10) if u < 20 else np.arange(10, 20)
        other = np.arange(10, 20) if u < 20 else np.arange(10)
        picks = [(i, 4.5) for i in rng.choice(mine, 7, replace=False)] + [(i, 1.5) for i in rng.choice(other, 2, replace=False)]
        # Liked films come last in time, so the temporal test split contains films each user loved.
        picks = picks[7:] + picks[:7]
        for t, (i, r) in enumerate(picks):
            users.append(u)
            items.append(int(i))
            values.append(r)
            stamps.append(1_000 + t)
    return Ratings(np.array(users, np.int32), np.array(items, np.int32), np.array(values, np.float32), np.array(stamps, np.int64))


def planted_dataset() -> Dataset:
    movies = [Movie(f"Space {i}", 1990 + i, ["Sci-Fi", "Action"]) for i in range(10)]
    movies += [Movie(f"Love {i}", 2000 + i, ["Romance", "Drama"]) for i in range(10)]
    return Dataset(movies, planted_ratings(), N_USERS, tags=[[] for _ in movies])


def cluster(item: int) -> int:
    return 0 if item < 10 else 1


# ---- metrics -------------------------------------------------------------

def test_ranking_metrics_on_a_hand_example():
    recs, relevant = [3, 1, 7, 9], {1, 9, 42}
    assert precision_at_k(recs, relevant, 4) == 0.5
    assert recall_at_k(recs, relevant, 4) == pytest.approx(2 / 3)
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(recs, relevant, 4) == pytest.approx((1 / np.log2(3) + 1 / np.log2(5)) / ideal)
    assert ndcg_at_k([1, 9, 42], relevant, 3) == pytest.approx(1.0)
    assert recall_at_k(recs, set(), 4) == 0.0


# ---- data ----------------------------------------------------------------

def test_temporal_split_holds_out_each_users_latest_ratings():
    ratings = planted_ratings()
    train, test = temporal_split(ratings, test_fraction=0.2)
    assert len(train) + len(test) == len(ratings)
    for u in range(N_USERS):
        tr = train.timestamps[train.users == u]
        te = test.timestamps[test.users == u]
        assert len(te) == 2  # round(9 * 0.2)
        assert tr.max() < te.min()  # nothing from the future leaks into training


def test_temporal_split_keeps_light_users_in_training():
    r = Ratings(np.array([0, 0, 1], np.int32), np.array([0, 1, 2], np.int32), np.ones(3, np.float32), np.arange(3))
    train, test = temporal_split(r, min_ratings=5)
    assert len(test) == 0 and len(train) == 3


def test_loader_parses_movielens_csvs(tmp_path):
    with open(tmp_path / "movies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        w.writerow([1, "Matrix, The (1999)", "Action|Sci-Fi"])
        w.writerow([7, "Untitled", "(no genres listed)"])
    with open(tmp_path / "ratings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["userId", "movieId", "rating", "timestamp"])
        w.writerows([[10, 1, 4.5, 100], [10, 7, 2.0, 101], [20, 1, 5.0, 102], [20, 99, 3.0, 103]])
    with open(tmp_path / "tags.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["userId", "movieId", "tag", "timestamp"])
        w.writerow([10, 1, "Cyberpunk", 100])

    ds = load(tmp_path)
    assert ds.movies[0].title == "The Matrix" and ds.movies[0].year == 1999
    assert ds.movies[1].genres == [] and ds.movies[1].year is None
    assert ds.n_users == 2 and len(ds.ratings) == 3  # rating for unknown movie 99 dropped
    assert ds.tags[0] == ["cyberpunk"]
    assert ds.user_ids == [10, 20] and ds.movie_ids == [1, 7]
    assert ds.find("matrix") == [0]


# ---- models --------------------------------------------------------------

def small_als() -> ImplicitALS:
    # Two planted clusters -> the true structure is rank 2; more factors just fit noise on 40 users.
    return ImplicitALS(factors=2, regularization=1.0, alpha=10, iterations=20)


@pytest.mark.parametrize("model", [ItemKNN(), small_als(), Hybrid(als=small_als())], ids=["knn", "als", "hybrid"])
def test_collaborative_models_recover_the_planted_clusters(model):
    model.fit(planted_ratings(), N_USERS, N_ITEMS)
    for user in (0, 5, 25, 39):
        recs = model.recommend(user, 3)
        assert recs, f"no recommendations for user {user}"
        assert all(cluster(i) == (0 if user < 20 else 1) for i in recs), (user, recs)
        assert not set(recs) & model.seen[user]  # never recommends what they already rated


def test_popularity_gives_everyone_the_same_list():
    model = Popularity().fit(planted_ratings(), N_USERS, N_ITEMS)
    assert np.array_equal(model.scores(0), model.scores(30))


def test_content_model_matches_genres():
    ds = planted_dataset()
    vectors = tfidf(item_documents(ds))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    model = ContentBased(vectors, popularity_weight=0.0).fit(ds.ratings, N_USERS, N_ITEMS)
    assert all(cluster(i) == 0 for i in model.similar(0, 5))
    assert all(cluster(i) == 1 for i in model.recommend(30, 3))


def test_als_fold_in_handles_a_brand_new_user():
    als = small_als().fit(planted_ratings(), N_USERS, N_ITEMS)
    taste = als.fold_in([12, 13, 14], [5.0, 4.5, 5.0])
    scores = als.item_factors @ taste
    top = [i for i in np.argsort(-scores) if i not in (12, 13, 14)][:3]
    assert all(cluster(int(i)) == 1 for i in top)


def test_evaluate_reports_sensible_numbers():
    train, test = temporal_split(planted_ratings())
    good = evaluate(ItemKNN(), train, test, N_USERS, N_ITEMS, k=3)
    base = evaluate(Popularity(), train, test, N_USERS, N_ITEMS, k=3)
    assert 0 <= base.ndcg <= 1 and 0 <= good.ndcg <= 1
    assert good.ndcg > base.ndcg
    assert good.users == N_USERS


# ---- CLI -----------------------------------------------------------------

def test_cli_evaluate_and_similar(tmp_path, capsys):
    ds = planted_dataset()
    with open(tmp_path / "movies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        for i, m in enumerate(ds.movies):
            w.writerow([i + 1, f"{m.title} ({m.year})", "|".join(m.genres)])
    with open(tmp_path / "ratings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["userId", "movieId", "rating", "timestamp"])
        r = ds.ratings
        for u, i, v, t in zip(r.users, r.items, r.values, r.timestamps):
            w.writerow([int(u) + 1, int(i) + 1, float(v), int(t)])

    main(["--data", str(tmp_path), "evaluate", "-k", "3"])
    out = capsys.readouterr().out
    assert "| Popularity |" in out and "Hybrid (ALS + kNN)" in out and "popularity baseline" in out

    main(["--data", str(tmp_path), "similar", "space 3", "-n", "3"])
    out = capsys.readouterr().out
    assert "More like Space 3" in out and "Love" not in out
