"""RL agent: a contextual bandit (Linear Thompson Sampling, position-based model) on top of hybrid-plus.

Why a bandit. Every request is one decision (10 products) followed by one reward (purchase or not),
and our answer barely changes the next shopper: that is exactly the one-step RL problem, a
*contextual bandit*. The agent has to balance exploitation (show what the model thinks sells)
with exploration (try products it is unsure about), and learn from `team_NN.feedback` live.

How it works, per request:
  1. candidates: hybrid-plus (transformer + XGB ranker + v6 personal products) plus v2's
     same-subcategory / same-category popular products (~70 products);
  2. features: for every candidate a vector x (hybrid rank and 1/(1+rank), XGB score, "viewed in session",
     "seen today", "in 60-day history", same subcategory, trending, popularity, price, new product...);
  3. Thompson sampling: draw theta from the posterior, score = x . theta, show the top 10 in order;
  4. learning (background thread): when the feedback arrives, every shown product at slot k gives
     one observation  y = purchased + LAMBDA_CLICK * clicked  with  E[y] = e_k * (x . theta),
     where e_k is the probability that the shopper looks at slot k (position-based model).
     That is Bayesian linear regression on e_k * x, so the posterior update is exact and cheap.

Safety: the prior puts all the weight on the hybrid-plus rank, so before any feedback the agent
answers exactly like hybrid-plus; feedback then moves it away only as far as the data supports.
Forgetting (DECAY) lets it follow new trends, which matters in the final round (new data).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import recommender_v2 as v2
import hybrid_plus

# Bug fix for the teammate's transformer: for a session with no recent views every item position is
# padding, and PyTorch's inference "fast path" then returns a different embedding when the request
# is run alone than when it is run in a batch / during training (max prob diff ~0.97). With the fast
# path off, single requests, batches and training all agree. Affects every module in this process.
torch.backends.mha.set_fastpath_enabled(False)

# ---------------------------------------------------------------- settings (env overridable)

STATE_PATH = Path(os.environ.get("RL_STATE", Path(__file__).resolve().parent.parent / "state" / "rl_state.npz"))
LAMBDA_CLICK = float(os.environ.get("RL_LAMBDA_CLICK", "0.15"))   # a click is worth this much of a purchase
EXPLORE = float(os.environ.get("RL_EXPLORE", "1.0"))              # posterior std multiplier (0 = greedy)
DECAY = float(os.environ.get("RL_DECAY", "0.9995"))              # per feedback batch (~5 s): half-life ~2 h
NOISE_VAR = 0.01                                                  # observation noise of y (Bernoulli ~ 1%)
PRIOR_STD = 0.003                                                 # prior std of every weight (small: start close to hybrid-plus)
PRIOR_HYB = 0.03                                                  # prior weight on the hybrid reciprocal rank
MAX_CANDIDATES = 70

# probability that the shopper looks at slot k (overwritten by pretrain_rl.py from the history)
DEFAULT_EXPOSURE = np.array([1.0, 0.72, 0.58, 0.49, 0.42, 0.37, 0.33, 0.30, 0.27, 0.25])

FEATURES = [
    "bias", "hyb_rank", "hyb_rr", "xgb_z", "trans_prob", "in_session", "last_viewed",
    "seen_today", "in_history", "same_sub", "same_cat", "trending", "popularity", "conversion",
    "price_gap", "new_product", "newuser_x_pop", "productpage_x_samesub",
]
D = len(FEATURES)

NEW_PRODUCTS_SQL = """
SELECT DISTINCT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String = 'new_product'"""

FEEDBACK_SQL = """
SELECT request_id::String AS rid, shown, clicked::String AS clicked, purchased::String AS purchased,
       served_by::String AS served_by
FROM {team}.feedback
WHERE ts::DateTime64(3) > now() - INTERVAL 3 MINUTE AND request_id::String != '__init__'"""


# ---------------------------------------------------------------- the bandit

class LinTSPBM:
    """Bayesian linear regression  y ~ N(e_k * x.theta, NOISE_VAR)  with Thompson sampling."""

    def __init__(self, d: int = D, exposure: np.ndarray = DEFAULT_EXPOSURE):
        self.d = d
        self.exposure = np.asarray(exposure, dtype=float)
        mu0 = np.zeros(d)
        mu0[FEATURES.index("hyb_rr")] = PRIOR_HYB
        self.A0 = np.eye(d) / PRIOR_STD ** 2          # prior precision
        self.b0 = self.A0 @ mu0
        self.A, self.b = self.A0.copy(), self.b0.copy()
        self.n_obs = 0
        self.n_purchases = 0
        self.lock = threading.Lock()
        self._refresh()

    def _refresh(self) -> None:
        cov = np.linalg.inv(self.A)
        cov = (cov + cov.T) / 2
        mean = cov @ self.b
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(self.d))
        self.mean, self.chol = mean, chol                 # swapped as a pair, read without lock

    def sample(self, explore: float = EXPLORE) -> np.ndarray:
        mean, chol = self.mean, self.chol
        if explore <= 0:
            return mean
        return mean + explore * chol @ np.random.standard_normal(self.d)

    def update(self, X: np.ndarray, slots: np.ndarray, y: np.ndarray, weight: float = 1.0) -> None:
        """X (n, d) features of shown products, slots (n,) their 0-based slot, y (n,) reward."""
        if len(X) == 0:
            return
        Xe = X * self.exposure[np.minimum(slots, len(self.exposure) - 1)][:, None]
        with self.lock:
            self.A += weight * (Xe.T @ Xe) / NOISE_VAR
            self.b += weight * (Xe.T @ y) / NOISE_VAR
            self.n_obs += len(X)
            self.n_purchases += int((y >= 1).sum())
            self._refresh()

    def decay(self, gamma: float = DECAY) -> None:
        """Forget old data towards the prior (non-stationary stream)."""
        with self.lock:
            self.A = self.A0 + gamma * (self.A - self.A0)
            self.b = self.b0 + gamma * (self.b - self.b0)
            self._refresh()

    def weights(self) -> dict[str, float]:
        return {f: round(float(w), 6) for f, w in zip(FEATURES, self.mean)}

    def save(self, path: Path = STATE_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            tmp = Path(str(path) + ".tmp.npz")
            np.savez(tmp, A=self.A, b=self.b, exposure=self.exposure,
                     n_obs=self.n_obs, n_purchases=self.n_purchases, features=np.array(FEATURES))
            os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "LinTSPBM":
        bandit = cls()
        warm = Path(path).with_name("rl_pretrain.npz")
        if not Path(path).exists() and warm.exists():
            print(f"no live state yet, starting from the offline warm start {warm}")
            path = warm
        if Path(path).exists():
            z = np.load(path)
            if list(z["features"]) == FEATURES:
                bandit.exposure = z["exposure"]
                bandit.A, bandit.b = z["A"], z["b"]
                bandit.n_obs, bandit.n_purchases = int(z["n_obs"]), int(z["n_purchases"])
                bandit._refresh()
                print(f"RL state loaded from {path}: {bandit.n_obs} observations, {bandit.n_purchases} purchases")
            else:
                print(f"RL state {path} has other features, starting from the prior")
        return bandit


# ---------------------------------------------------------------- the agent

class RLAgent:
    def __init__(self, hybrid, model, bandit: LinTSPBM | None = None):
        self.hybrid = hybrid            # hybrid_plus.HybridPlus (transformer + XGB)
        self.model = model              # recommender_v6.Model (catalogue, personal products)
        self.bandit = bandit or LinTSPBM.load()
        self.new_products: set[str] = set()
        self.decisions: OrderedDict[str, tuple[float, list[str], np.ndarray]] = OrderedDict()
        self.processed: OrderedDict[str, float] = OrderedDict()
        self.max_purch = max([p for _, p in hybrid.pop.values()] + [1])

    # ---- live data refresh (call every ~60 s): keeps the hybrid's catalogue current
    def refresh_catalog(self) -> None:
        catalog, active = {}, set()
        for r in hybrid_plus.rd.query(hybrid_plus.rd.CATALOG_SQL):
            catalog[r["product_id"]] = r
            if r.get("is_active", True) in (True, 1, "true", "1"):
                active.add(r["product_id"])
        self.hybrid.catalog, self.hybrid.active_products = catalog, active
        try:
            self.new_products = {r["product_id"] for r in v2.query(NEW_PRODUCTS_SQL)}
        except Exception as e:  # empty before the stream starts
            print(f"new products not available: {e}")

    # ---- features
    def candidates_and_features(self, reqs: list[dict], unavailable: set[str],
                                live: bool = True, extra: list[list[str]] | None = None):
        """For every request: (candidates, X, hybrid order). `extra` adds products that must be scored."""
        h, m = self.hybrid, self.model
        probs = h._run_transformer_batch(reqs)
        rows, all_feats = [], []
        for i, req in enumerate(reqs):
            uid = str(req.get("user_id", ""))
            u_prods = h.user_hist.get(uid, set())
            personal, bought = hybrid_plus.personal_items(req, m) if live else ([], set())
            blocked = set(unavailable) | set(bought)
            h.extra_items = personal
            try:
                cands = h.get_candidates(probs[i], u_prods, blocked, hybrid_plus.PLUS_LIMIT)
            finally:
                h.extra_items = []
            for p in v2.recommend(req.get("recent_views"), m, blocked):
                if p not in cands and len(cands) < MAX_CANDIDATES:
                    cands.append(p)
            for p in (extra[i] if extra else []):        # always scored, even if blocked
                if p not in cands:
                    cands.append(p)
            feats = h.build_features_for_request(req, cands, probs[i], u_prods)
            rows.append((req, cands, feats, probs[i], u_prods, blocked, personal))
            all_feats.append(feats)
        scores = h.booster.inplace_predict(np.ascontiguousarray(np.vstack(all_feats), dtype=np.float32))

        out, cursor = [], 0
        for req, cands, feats, prob, u_prods, blocked, personal in rows:
            s = np.asarray(scores[cursor:cursor + len(cands)], dtype=float)
            cursor += len(cands)
            X = self._features(req, cands, feats, s, personal, live)
            out.append((cands, X, np.argsort(-s), blocked))
        return out

    def _features(self, req, cands, xf, s, personal, live) -> np.ndarray:
        m = self.model
        n = len(cands)
        uid = str(req.get("user_id", ""))
        ctx = req.get("context") or {}
        rv = [v["product_id"] for v in (req.get("recent_views") or []) if isinstance(v, dict) and "product_id" in v]
        session, last = set(rv), (rv[-1] if rv else None)
        today = set(m.live_items.get(uid, [])) if live else set()
        history = set(m.user_items.get(uid, [])) | self.hybrid.user_hist.get(uid, set())
        hot = set(self.hybrid.live_hot[:30]) if live else set()
        rank = np.empty(n)
        rank[np.argsort(-s)] = np.arange(n)
        z = np.clip((s - s.mean()) / (s.std() + 1e-9), -3, 3) / 3
        new_user = 1.0 if req.get("is_new_user") in (True, 1, "true", "1") else 0.0
        product_page = 1.0 if str(ctx.get("page", "")) == "product" else 0.0

        X = np.zeros((n, D))
        X[:, 0] = 1.0
        X[:, 1] = 1.0 - rank / max(n - 1, 1)
        X[:, 2] = 1.0 / (1.0 + rank)                                  # reciprocal rank: top-heavy
        X[:, 3] = z
        X[:, 4] = np.minimum(xf[:, 1] / 10.0, 1.0)                      # log1p(t_prob * 1e4) / 10
        X[:, 5] = [p in session for p in cands]
        X[:, 6] = [p == last for p in cands]
        X[:, 7] = [p in today for p in cands]
        X[:, 8] = [p in history for p in cands]
        X[:, 9] = xf[:, 7]                                              # same subcategory as last view
        X[:, 10] = xf[:, 6]                                             # same category as last view
        X[:, 11] = [p in hot for p in cands]
        X[:, 12] = xf[:, 4] / math.log1p(self.max_purch)                # log1p(purchases), normalised
        X[:, 13] = np.minimum(xf[:, 5] * 50, 1.0)                       # conversion rate
        X[:, 14] = np.minimum(np.abs(np.log(np.maximum(xf[:, 8], 1e-3))), 3) / 3
        X[:, 15] = [p in self.new_products for p in cands]
        X[:, 16] = new_user * X[:, 12]
        X[:, 17] = product_page * X[:, 9]
        return X

    # ---- acting
    def act(self, reqs: list[dict], unavailable: set[str], explore: float = EXPLORE,
            log: bool = True) -> list[list[str]]:
        answers = []
        for req, (cands, X, _, blocked) in zip(reqs, self.candidates_and_features(reqs, unavailable)):
            theta = self.bandit.sample(explore)                 # one posterior draw per shopper
            order = np.argsort(-(X @ theta))
            items, rows = [], []
            for j in order:
                p = cands[j]
                if p in self.model.info and p not in blocked and p not in items:
                    items.append(p)
                    rows.append(j)
                    if len(items) == 10:
                        break
            if len(items) < 10:
                raise ValueError(f"only {len(items)} valid candidates")
            answers.append(items)
            if log:
                self.log_decision(req["request_id"], items, X[rows])
        return answers

    def log_decision(self, rid: str, items: list[str], X: np.ndarray) -> None:
        """Remember the features of what was shown, to learn when the feedback arrives."""
        now = time.time()
        self.decisions[rid] = (now, items, X)
        while self.decisions and now - next(iter(self.decisions.values()))[0] > 600:
            self.decisions.popitem(last=False)

    def features_for(self, req: dict, items: list[str], unavailable: set[str]) -> np.ndarray:
        """Features of a list chosen by another policy (A/B arm), so the bandit learns from it too."""
        cands, X, _, _ = self.candidates_and_features([req], unavailable, extra=[items])[0]
        idx = {p: j for j, p in enumerate(cands)}
        return X[[idx[p] for p in items]]

    # ---- learning
    def learn_once(self, team: str) -> int:
        rows = v2.query(FEEDBACK_SQL.format(team=team))
        Xs, slots, ys, n = [], [], [], 0
        for r in rows:
            rid = r["rid"]
            if rid in self.processed or rid not in self.decisions:
                continue
            self.processed[rid] = time.time()
            if r["served_by"] != "team":
                continue
            _, items, X = self.decisions.pop(rid)
            shown = r["shown"] or []
            shown = [str(x) for x in (json.loads(shown) if isinstance(shown, str) else shown)]
            if shown[:10] != items:
                continue
            for k, p in enumerate(items):
                ys.append((p == (r["purchased"] or "")) * 1.0 + (p == (r["clicked"] or "")) * LAMBDA_CLICK)
                slots.append(k)
                Xs.append(X[k])
            n += 1
        now = time.time()
        while self.processed and now - next(iter(self.processed.values())) > 900:
            self.processed.popitem(last=False)
        if Xs:
            self.bandit.decay()
            self.bandit.update(np.array(Xs), np.array(slots), np.array(ys))
        return n

    def learn_forever(self, team: str, every: float = 5.0) -> None:
        saved = logged = time.time()
        while True:
            try:
                n = self.learn_once(team)
                if n:
                    print(f"RL learned from {n} requests (total {self.bandit.n_obs // 10} req, "
                          f"{self.bandit.n_purchases} purchases)")
                if time.time() - saved > 60:
                    saved = time.time()
                    self.bandit.save()
                if time.time() - logged > 300:
                    logged = time.time()
                    self.publish_weights(team)
            except Exception as e:
                print(f"RL learning failed: {e}")
            time.sleep(every)

    def publish_weights(self, team: str) -> None:
        """Snapshot of what the agent has learned, in team_NN.scratch (for the dashboard / demo)."""
        rec = {"kind": "rl_weights", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "n_obs": self.bandit.n_obs, "n_purchases": self.bandit.n_purchases,
               "weights": json.dumps(self.bandit.weights())}
        v2.http.post(f"{v2.API}/v1/tables/scratch", params={"database": team}, json=[rec],
                     timeout=10).raise_for_status()
        print("RL weights:", self.bandit.weights())
