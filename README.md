# RL agent: contextual bandit on top of hybrid-plus

## The idea in one paragraph (for the jury / a product manager)

Every shopper is one decision (which 10 products, in which order) followed by one reward (did they buy).
That is reinforcement learning in its one-step form, a **contextual bandit**. Hybrid-plus is a model trained
once on the past; the bandit keeps **learning from every answer we give**, through `team_NN.feedback`, and it
**explores** on purpose: when it is unsure whether a product would sell, it sometimes shows it to find out.
It starts as an exact copy of hybrid-plus, so it can only move away from it when live data says so.

## How it decides (per request, ~2 ms)

1. **Candidates** (~70): hybrid-plus pool (transformer top 35 + v6 personal products + live trends + best sellers)
   plus v2's popular products from the same subcategory / category as the last view.
2. **Features** for every candidate (18): hybrid rank and 1/(1+rank), XGB score, transformer probability,
   viewed in this session / last viewed, seen today (live events), in the 60-day history, same subcategory /
   category, trending in the last hour, popularity, conversion rate, price gap with the session,
   **new product** (cold start), new user × popularity, product page × same subcategory.
3. **Thompson sampling**: draw weights θ from the posterior, score = x·θ, show the top 10 in that order.
   Uncertain weights vary between draws, which gives exploration for free, and it shrinks as data arrives.

## How it learns (background thread, every 5 s)

Position-based model: a product at slot *k* is seen with probability *e_k* (estimated from the click share
per slot in `shop.history_sessions`). For every shown product we observe
`y = purchased + 0.15 × clicked` with `E[y] = e_k · (x·θ)`. That is Bayesian linear regression on `e_k·x`,
so the posterior update is exact and takes microseconds. Old data is slowly forgotten (half-life ≈ 2 h),
so the agent follows the new trends of the final round.

* Prior: all the weight on the hybrid reciprocal rank, so before any feedback the agent's top-1 is always
  hybrid-plus's top-1.
* It learns from **both** A/B arms: for requests answered by hybrid-plus it logs the features of what was
  shown, so the A/B test does not slow learning down.
* State is saved to `state/rl_state.npz` every minute and loaded at start; a weight snapshot goes to
  `team_NN.scratch` every 5 minutes (query 9 in `sql/queries.sql`).

## Layout

```
starter/
├── run_rl.py            entry point: the live agent (pull -> act -> push, learns in the background)
├── pretrain_rl.py       entry point: offline warm start + validation on shop.history_sessions
├── _paths.py            makes the folders below importable, finds .env and the model files
├── .env                 TEAM_ID, RAWTREE_API_KEY and the RL_* settings (never commit)
├── requirements.txt
├── rl/
│   └── rl_agent.py      the contextual bandit (LinTS-PBM): features, acting, learning
├── models/              what the agent builds on: hybrid-plus = transformer + XGB ranker
│   ├── hybrid_plus.py, recommenderd.py
│   ├── meta.json, sasrec_ctx.pt, xgb_ranker.json      trained weights
│   └── train_transformer.py, train_xgb.py             how they were trained
├── recommenders/        catalogue, personal products, stock, fallbacks
│   └── recommender.py (baseline), recommender_v2.py, recommender_v5.py, recommender_v6.py
├── sql/queries.sql      RawTree queries: A/B per policy (8), learned weights (9), ...
└── state/
    ├── rl_state.npz     what the agent has learned live (saved every minute, loaded at start)
    └── rl_pretrain.npz  offline warm start (pretrain_rl.py); used only when rl_state.npz does not exist yet
```

## Running it

From any folder (the scripts find `starter/` themselves):

```bash
pip install -r starter/requirements.txt
python starter/run_rl.py           # the live agent; RL_SHARE / RL_EXPLORE are read from .env
python starter/pretrain_rl.py      # optional; writes state/rl_pretrain.npz, never the live state
python starter/models/train_transformer.py && python starter/models/train_xgb.py   # retrain the base models
```

| Env var | Default | Meaning |
|---|---|---|
| `RL_SHARE` | 0.5 | share of requests answered by the RL agent (rest: hybrid-plus) |
| `RL_EXPLORE` | 1.0 | exploration (posterior std multiplier); 0 = greedy |
| `RL_DECAY` | 0.9995 | forgetting per 5-s batch (half-life ≈ 2 h) |
| `RL_LAMBDA_CLICK` | 0.15 | value of a click relative to a purchase in the reward |
| `RL_STATE` | `state/rl_state.npz` | where the live state is saved |

Safety net: RL fails → hybrid-plus → v6 → official baseline. `model_version` says who answered.

## Validation so far

* **Simulation** (fake RawTree, real transformer + XGB, synthetic shoppers):
  * world where hybrid-plus is exactly right: RL 10.99 vs hybrid 11.05 expected purchases / 100 requests
    (the price of exploration is ~0.5 %);
  * world where shoppers react to something hybrid-plus does not model (what they saw today, new products):
    RL goes from 1.8 to 9.9 while hybrid stays at 1.8. The learned weights point to the right signal.
* **Offline**: `pretrain_rl.py` reports MRR / hit@k of the bought product within the 10 shown, hybrid vs RL.
  On 14 days of `shop.history_sessions` (≈32 000 sessions, 452 purchases) the largest learned weights after the
  hybrid rank are *in the 60-day history* (+0.0117) and *viewed in this session* (+0.0075): the same lesson as v5.
  Measured slot exposure: 1.00, 0.56, 0.41, 0.32, 0.26, 0.24, 0.21, 0.18, 0.16, 0.15.
* **Live**: A/B with query 8 in `sql/queries.sql`. Decide on conversion with the standard error column: with ~1 % conversion you
  need a few thousand requests per arm to see a 0.2-point difference.

## Bug fix found on the way

For a session with no recent views, the teammate's transformer gave a different output when a request
ran alone (as in hybrid-plus live) than in a batch or in training: PyTorch's inference fast path treats
the all-padding sequence differently. `rl_agent.py` turns the fast path off, so both arms of `run_rl.py`
get the training-consistent output. These are exactly the cold-start requests (new users), so it is worth
adding `torch.backends.mha.set_fastpath_enabled(False)` to `run_hybrid_plus.py` too.
