# Team 15 · Epoch 1 — real-time recommendations (Datathon FME 2026, RawTree challenge)

For every shopper request we answer, within 3 s, with 10 product IDs that are in order and in stock.
The final model is **`starter/loco5.py`**: a LightGBM LambdaRank ranker over ~50 candidates per request,
with a small session transformer and live signals from the event stream.

---

## Installation

Requires Python 3.10+. A GPU is optional: the transformers use CUDA when available.

```bash
git clone https://github.com/ReallyTimba/datathon26_epoch1
cd datathon26_epoch1
pip install requests numpy pandas torch xgboost lightgbm scikit-learn
```

Create a `.env` file in the repository root. It is git-ignored, so never commit it.

```
TEAM_ID=team_15
RAWTREE_API_KEY=...
```

## Running

From the repository root:

```bash
python starter/loco5.py      # final model: pulls requests, answers, pushes submissions
python starter/monitor.py    # optional, in a second terminal: live scoreboard every 15 s
```

- **loco5.py.** The first start reads the history from RawTree and trains both models, which takes a few
  minutes. It saves them to `model_cache_loco5_live.pt`, and later starts load that file in under a second.
  Delete the file to retrain on fresh data. Answers carry `model_version = "loco5"`.
- **monitor.py.** Shows conversion, cold-start conversion and €/request for each `model_version`, from
  `team_15.feedback` joined with `team_15.submissions`.

---

## How it works

```
                live stream (every 2.5 s) · catalog signals (every 10 s) · stock (every 1 s)
                                     │
request ─► candidates (~50) ─► 14 features ─► LambdaRank + blend ─► top 10 ─► checks ─► push
```

### 1. Live context

The request's `recent_views` are enriched with the shopper's latest views and add-to-carts from
`shop.events` (the last 3 minutes, refreshed every 2.5 s). The model also tracks new products and price drops
from `shop.catalog_updates`.

### 2. Candidates

- **Shopper with views:**
  - products co-viewed with the session (`shop.history_events`, weighted by recency);
  - TF-IDF + SVD text neighbours for products with too little history;
  - the best products of the session's main category;
  - the best products for the device × region segment;
  - products in the shopper's live cart.
- **New shopper (no views):** the best products for the segment, the global best, new products and
  the products most added to carts right now.

"Best" is a Bayesian-smoothed conversion rate: purchases over impressions in `shop.history_sessions`, with a
prior on the global rate so that rare products are not over-ranked.

### 3. Features (14 per candidate)

| Group | Features |
|---|---|
| Model scores | transformer score, co-view score, text similarity |
| Conversion | smoothed conversion rate (global, device × region) |
| Fit | category match, subcategory match |
| Price | price, price ratio to the session, price-drop flag |
| Context | never-sold flag, session length, already viewed, mobile |

### 4. Models

- **Session transformer** (1 layer, last 15 views). It is trained with an in-batch softmax loss in which a
  purchase weighs 4× a click. Its score of each candidate is the first feature.
- **LightGBM `LGBMRanker(objective="lambdarank")`**:
  - it optimises NDCG@1/3/5 within each session;
  - graded labels: purchased = 3, clicked = 1, other shown products = 0;
  - trained on 18 000 sessions with a purchase and 7 000 with a click.

We rank instead of classifying because the position decides the sale. On our live feedback, slot 1 converts
about 2.5× slot 2 and about 8× slots 6–10, so the model is trained to get the top of the list right.

### 5. Final score and selection

- **Shopper with views:** the LambdaRank score plus conversion rate, category match, transformer score and a
  boost for products in the live cart.
- **New shopper:** the LambdaRank score plus segment and global conversion. The top 6 slots go to
  **6 different categories**, so the list matches an unknown intent sooner. Slots 7–10 follow the score.

### 6. Checks

- Out-of-stock (checked every second) and inactive products are excluded.
- The answer always has 10 distinct products: if fewer candidates survive, it is filled with best sellers.
- An error in one batch is logged and the loop keeps serving.

---

## Other models in this repository

Earlier stages, kept with their training code.

- **hybrid+ (`models/`).**
  - A 4-layer transformer encodes the last 15 views plus context (device, region, page). It is trained with
    BPR loss on 150 000 sessions (`train_transformer.py`).
  - An `XGBRanker(objective="rank:ndcg")` with 13 features re-ranks its top 35 candidates. It is trained on
    80 000 sessions (`train_xgb.py`).
  - Both scripts read `.env` from the root and write their weights to `models/`.
- **RL agent (`rl/rl_agent.py`, `run_rl.py`, `pretrain_rl.py`).** A contextual bandit on top of hybrid+.
  - It scores ~70 candidates with 18 features and learns from `team_15.feedback` every 5 s (purchase = 1,
    click = 0.15) with Bayesian linear regression.
  - It corrects for slot position and explores with Thompson sampling.
  - `state/rl_pretrain.npz` is its warm start, learned from 14 days of history.
- **Rule-based versions (`recommenders/recommender_v1.py` … `recommender_v6.py`).**
  - Popular products of the viewed subcategory (v1).
  - The session's own views first (v2).
  - The shopper's 60-day history of viewed but not bought products (v5).
  - Today's live activity without re-buys (v6).
  - `ab_test.py` is the live A/B harness that split traffic between versions by a hash of `request_id`.

## Repository layout

```
starter/      loco5.py (final model), monitor.py, feature_study.py, kbest_study.py
models/       hybrid+: transformer + XGBoost ranker, with training scripts
rl/           RL agent (run with run_rl.py, warm start with pretrain_rl.py, state in state/)
recommenders/ rule-based versions v0–v6 and the A/B harness
sql/          RawTree queries used for analysis and A/B checks
```

