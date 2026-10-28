# CineMatch

[![CI](https://github.com/kilobyte23/CineMatch/actions/workflows/ci.yml/badge.svg)](https://github.com/kilobyte23/CineMatch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Movie recommendations on 100,000 real ratings, five models compared honestly.**

CineMatch implements five recommenders from scratch in NumPy, from a popularity baseline to implicit-feedback matrix factorisation. It evaluates them the way production recommenders are evaluated: predict what each person watches *next*, from what they watched before. A CLI then recommends for existing users, finds similar films, and onboards brand-new users from a handful of ratings.

## Results

[MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest-small`: 610 users, 9,742 movies, 100,836 ratings. Each user's latest 20% of ratings are held out. A recommendation is a hit if the user went on to rate that film 4 or more.

| Model | Precision@10 | Recall@10 | NDCG@10 | Catalogue coverage |
| --- | --- | --- | --- | --- |
| Popularity | 0.060 | 0.050 | 0.078 | 0.9% |
| Content (TF-IDF) | 0.060 | 0.048 | 0.073 | 6.5% |
| Item-kNN | 0.086 | 0.084 | 0.109 | 5.3% |
| Implicit ALS | 0.088 | 0.084 | 0.108 | 4.3% |
| **Hybrid (ALS + kNN)** | **0.095** | **0.094** | **0.118** | 4.3% |

The hybrid ranks held-out favourites **1.51×** better than popularity (NDCG@10). Reproduce it with `cinematch evaluate`, which also runs in CI.

**What the numbers say**

- **Collaborative filtering wins.** Who-liked-what beats what-a-film-is-about on this data. Content-only TF-IDF over genres, era and tags scored NDCG 0.008 before adding a popularity prior, because it surfaced obscure films that happen to share tags. It's kept for "more like this" and cold-start items, not accuracy.
- **kNN and ALS make different mistakes**, so blending their standardised scores beats both.
- **Popularity is a strong baseline** under a temporal split. Coverage shows its cost: the same 0.9% of the catalogue for everyone.
- **Treating low ratings as negatives** (preference 0, high confidence) instead of "watched, so positive" improved ALS NDCG from 0.098 to 0.108. A unit test with planted taste clusters caught the original behaviour.

Hyperparameters were chosen on a validation split carved out of the *training* data. The test split was used only for the table above.

## Models

| Model | Idea | Code |
| --- | --- | --- |
| Popularity | Count of 4+ ratings. Same list for everyone | [`models.py`](cinematch/models.py) `Popularity` |
| Content | TF-IDF over genres, decade and user tags. A user profile is the rating-weighted sum of watched films (centred on that user's mean rating), plus a popularity prior | `ContentBased` |
| Item-kNN | Cosine similarity between films over who liked them. Score = similarity to films you liked. Also gives the "because you liked X" explanations | `ItemKNN` |
| Implicit ALS | Hu, Koren & Volinsky (2008) matrix factorisation, solved with alternating least squares in closed form. 32 factors, λ = 30, α = 2 | `ImplicitALS` |
| Hybrid | 0.3 × ALS + 0.7 × kNN on z-scored outputs | `Hybrid` |

**Cold start.** `onboard` asks a new user to rate a few popular films chosen to span genres. It **folds** those ratings into the trained ALS item space with one least-squares solve, so no retraining is needed, and blends the result with kNN.

## Try it

Requires Python 3.10 or later. The dataset (about 1 MB) downloads on first run to `~/.cache/cinematch`.

```bash
pip install -e .
cinematch evaluate
cinematch recommend --user 1 -n 5
cinematch similar "toy story"
cinematch onboard
```

```
$ cinematch recommend --user 1 -n 3
User 1 rated 232 movies. Favourites:
  5.0  Seven (a.k.a. Se7en) (1995)
  5.0  The Usual Suspects (1995)
  ...
Recommended by Hybrid (ALS + kNN):
   1. The Godfather (1972)  [Crime, Drama]
      because you liked Goodfellas (1990)
   2. Die Hard (1988)  [Action, Crime, Thriller]
      because you liked The Terminator (1984)
   3. Ferris Bueller's Day Off (1986)  [Comedy]
      because you liked Big (1988)
```

Use it as a library:

```python
from cinematch.data import load, temporal_split
from cinematch.models import Hybrid
from cinematch.metrics import evaluate

ds = load()
train, test = temporal_split(ds.ratings)
print(evaluate(Hybrid(), train, test, ds.n_users, ds.n_items, k=10))
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

12 tests run in under a second on synthetic data with **planted taste clusters**, with no download. They check that:

- each collaborative model recommends within the right cluster
- no model recommends films a user has already rated
- the temporal split never leaks future ratings into training
- the metrics match hand-computed values
- the CSV loader handles MovieLens quirks such as "Matrix, The" and missing genres
- ALS fold-in places a new user correctly
- the CLI works end to end

CI runs the tests on Python 3.10, 3.12 and 3.13, then reproduces the results table on the real dataset.

## Project layout

```
cinematch/
  data.py      download, CSV loading, temporal split
  models.py    Popularity, ContentBased, ItemKNN, ImplicitALS, Hybrid
  metrics.py   precision/recall/NDCG@k, coverage, evaluation loop
  cli.py       evaluate · recommend · similar · onboard
tests/
```

## Limitations and next steps

- `ml-latest-small` is small. The 25M-rating set would need sparse matrices, since item-kNN here uses a dense item × item matrix.
- Offline metrics reward predicting what people watched anyway. An online A/B test is the real judge.
- Ideas: BPR or LightGCN, diversity re-ranking (coverage is low for every model), and a small web UI with posters.

## Data and license

Code: [MIT](LICENSE). MovieLens data © GroupLens Research, University of Minnesota, under its own [terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). It's downloaded on demand and not redistributed here.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5, 4: 19:1–19:19.
