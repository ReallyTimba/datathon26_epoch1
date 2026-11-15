# Team 15 · Epoch 1 — real-time recommendations (Datathon FME 2026, RawTree challenge)

Every ~0.3 s a shopper opens a page. Within 3 s we must answer with 10 product IDs, in order and in stock.
The baseline shows everyone the same historical best sellers (0.70 % conversion).
This repository has every model we built in one and a half days, the one we ran in the final round, and the
tools we used to decide between them.

**Final model: `starter/loco5.py`.** It is a LightGBM LambdaRank ranker over ~50 candidates per request,
with a small session transformer and live signals from the event stream.

---

## Quick start

```bash
pip install -r requirements.txt
```

Create `.env` in the repository root. It is git-ignored, so never commit it.

```
TEAM_ID=team_15
RAWTREE_API_KEY=...
```

| What | Command (from the repository root) | Notes |
|---|---|---|
| **Final model (loco5)** | `python starter/loco5.py` | The first start trains for a few minutes and writes `model_cache_loco5_live.pt`. Later starts load it in under 1 s. Delete the file to retrain. |
| Live scoreboard | `python starter/monitor.py` | Conversion, cold-start conversion and €/request per `model_version`, every 15 s |
| RL agent | `python run_rl.py` | Needs the hybrid+ weights in `models/` (see *Training*). Settings live in `.env`, see [`rl/README.md`](rl/README.md). |
| RL offline warm start | `python pretrain_rl.py` | Writes `state/rl_pretrain.npz` and never touches the live state |

The answer `model_version` tells you which model answered: `loco5`, `rl-lints`, `hybrid-plus`, `v6-...`, and so on.

---

## Repository layout

```
.
├── starter/                    final model and analysis (latest work)
│   ├── loco5.py                FINAL: LambdaRank + transformer + live signals, with its own serve loop
│   ├── monitor.py              live scoreboard from team_15.feedback × submissions
│   ├── feature_study.py        offline feature study: AUC, permutation importance, SHAP
│   └── kbest_study.py          SelectKBest ablation curve
├── run_rl.py, pretrain_rl.py   RL agent: live runner and offline warm start
├── _paths.py                   import paths for the RL agent (rl/, models/, recommenders/), finds .env
├── rl/rl_agent.py              contextual bandit (linear Thompson sampling, position-based model)
├── models/                     hybrid+ = session transformer + XGBoost ranker
│   ├── hybrid_plus.py, recommenderd.py
│   └── train_transformer.py, train_xgb.py
├── recommenders/               rule-based versions v0–v6 and the live A/B harness (history of the project)
├── sql/queries.sql             RawTree queries used for analysis and A/B checks
└── state/rl_pretrain.npz       RL warm start learned from 14 days of history
```

---

## How the final model works (loco5)

```
request ─► candidates (~50) ─► features (14) ─► LambdaRank + blend ─► top 10 ─► validity checks ─► push
             ▲                                                                     ▲
     live stream every 2.5 s, catalog signals every 10 s                   stock every 1 s
```

1. **Live context.** The request's `recent_views` are enriched with the shopper's last views and add-to-carts
   from the stream (the last 3 minutes, refreshed every 2.5 s).
2. **Candidates.**
   - Active shopper: products co-viewed with the session (decayed by recency); TF-IDF text neighbours for
     products with no sales history; best products of the session's main category; best products for the
     device × region segment; products in the shopper's live cart.
   - New shopper (no views): best products for the segment, global best, new products and live cart trends.
3. **Features (14):** transformer score, co-view score, Bayesian-smoothed conversion rate (global and per
   device × region), text similarity, category and subcategory match, price and price ratio to the session,
   price-drop flag, never-sold flag, session length, already viewed, mobile.
4. **Ranking.**
   - Session transformer: 1 layer, last 15 views, in-batch softmax loss in which purchases weigh 4×.
   - **LightGBM `LGBMRanker(objective="lambdarank")`**: graded labels, purchase = 3, click = 1, other shown
     products = 0. It optimises NDCG@1/3/5 within each session. Training data: 18 000 sessions with a
     purchase and 7 000 with a click.
   - Final score:
     - active shoppers: LambdaRank plus conversion rate, category match, transformer score and live cart;
     - new shoppers: LambdaRank plus segment and global conversion, then **6 different categories in the
       top 6 slots** so we match their intent sooner.
5. **Validity.** No out-of-stock or inactive product, always 10 (filled with best sellers).

**Why rank instead of classify.** The position decides the sale. On our own live feedback (~22 000 answers),
slot 1 converted 0.25 %, slot 2 0.10 %, slots 6–10 about 0.03 %. Getting the top right matters more than
getting a product somewhere in the list, so we optimise the order directly.

---

## How we got here (and why)

Every idea went through the same loop: **hypothesis → check on history (SQL + pandas over 12 M events and
2 M carousels) → small new version → live A/B test**, with traffic split by a hash of `request_id`.
Only the live A/B decided what stayed.

| Version | Idea | Conversion | Kept? |
|---|---|---|---|
| v0 baseline | same best sellers for everyone | 0.70 % | reference |
| v1 | popular products of the viewed subcategory | 0.83 % | ✓ |
| v2 | products viewed in this session first | 0.81–0.95 % | ✓ |
| v3 | ranking by position-corrected lift | 0.56 % | ✗ |
| v4 | "viewed together" after v2 | 0.81 % (v2: 0.89 %) | ✗ |
| **v5** | **the shopper's own 60-day history, not yet bought** | **1.12 %** (+18 % vs v2, +60 % vs baseline) | ✓ biggest jump |
| v6 | + today's live activity, no re-buys | — | ✓ |
| hybrid+ | transformer + XGBoost over our candidates | 0.85 % | stage |
| RL agent | contextual bandit learning from live feedback | 0.83 % (same window as hybrid+) | stage |
| loco2 | LightGBM over conversion, co-views and segment | 0.95 %, cold start 1.44 % | ✓ |
| loco3–4 | weighted co-views; extra session features | less coverage; one leaky feature | ✗ |
| **loco5** | **LambdaRank + cold-start diversification** | final round | **final** |

Compare numbers only within one row's window. The baseline alone moved between 0.56 % and 0.86 % depending
on the hour, so every decision was taken with both versions running at the same time.

What the data taught us:
- **Personal beats popular.** 70 % of live requests come from returning shoppers with 60 days of history, and
  49 % of purchases are a product the shopper had viewed or put in the cart in an earlier session. That gave v5.
- **The log is biased by the old policy.** Offline, an already-viewed product looked 18× better than average;
  live it was 3×. The history only shows what the old system chose to show, so offline numbers chose ideas,
  never winners.
- **Region, subcategory and lift did not move sales.** Category shares are within ±2 points across regions.

---

## The other models, briefly

**hybrid+ (`models/`).** A 4-layer transformer encodes the last 15 views plus context (device, region, page).
It is trained with BPR loss on 150 000 sessions for 15 epochs on GPU (`train_transformer.py`) and gives the
top 35 candidates. An `XGBRanker(objective="rank:ndcg")` with 13 features re-ranks them, trained on 80 000
sessions (`train_xgb.py`). "Plus" means we added v6's personal products to its candidates and blocked
products bought today.

**RL agent (`rl/`).** Each request is one decision and one reward, so it is a *contextual bandit*.
- Features: 18 per candidate.
- Reward: purchase = 1, click = 0.15, read from `team_15.feedback` every 5 s.
- Model: Bayesian linear regression under a position-based model (slot exposure estimated from history:
  1.00, 0.56, 0.41, … 0.15), with Thompson sampling for exploration and forgetting with a ~2 h half-life.
- Prior: all the weight on the hybrid+ rank, so it starts as hybrid+ and moves only when data says so.

The offline warm start learned the same lesson as v5: after the hybrid+ rank, the largest weights are
"in the 60-day history" and "viewed in this session". Live, it tied with hybrid+ (0.83 % vs 0.85 %, standard
error ±0.24 pp). Details in [`rl/README.md`](rl/README.md).

We also found a bug in the hybrid+ transformer: for sessions with no views, PyTorch's inference fast path
returned a different embedding for a single request than for a batch or in training. Turning the fast path
off fixed exactly the cold-start requests.

---

## Training

| Model | Command | Output |
|---|---|---|
| loco5 | `python starter/loco5.py` (first run) | `model_cache_loco5_live.pt` |
| hybrid+ transformer | `python models/train_transformer.py` | `models/sasrec_ctx.pt`, `models/meta.json` |
| hybrid+ ranker | `python models/train_xgb.py` (after the transformer) | `models/xgb_ranker.json` |
| RL warm start | `python pretrain_rl.py` | `state/rl_pretrain.npz` |

Trained weights are not committed. Each script rebuilds them from `shop.*` in RawTree.

---

## Reliability

- Stock is checked every second.
- Every answer has exactly 10 distinct, active, in-stock products, and nothing the shopper bought today
  (v6/RL).
- The RL runner falls back RL → hybrid+ → v6 → best sellers.
- `loco5.py` logs an error and keeps serving the next batch.

---

## Team and tools

| Who | What |
|---|---|
| David | loco line (loco2 → loco5, the final model), feature studies, live monitor |
| Tim | RL agent, hybrid+ cold-start fix, repository organisation |
| Marc | TODO |

Coding agents: Claude (RL agent and repository organisation). TODO: other agents and what they were used for.

## Known limitations

- `feature_study.py` and `kbest_study.py` import `loco2`/`loco3`/`loco4`, which are not in this repository.
- `recommenders/ab_test.py` and `run_hybrid_plus.py` are the earlier live runners, kept for history.
  `run_rl.py` replaces them.

