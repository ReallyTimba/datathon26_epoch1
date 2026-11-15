"""Offline warm start + offline validation of the RL agent on shop.history_sessions.

    python starter/pretrain_rl.py                # 40 000 sessions of the last 14 days
    RL_PRETRAIN_N=100000 python pretrain_rl.py

1. Position bias: click share per slot in the old system's log -> exposure e_k of the bandit.
2. Sessions are split 80/20 by a hash of request_id.
3. Train: every shown product at slot k is one observation y = purchased + LAMBDA_CLICK * clicked
   of E[y] = e_k * x.theta, added with weight PRETRAIN_WEIGHT (history is a different
   distribution from the live stream, so live feedback must be able to override it quickly).
4. Test: among the 10 products the old system showed, where does each policy put the one that was
   bought (and the one clicked)? Mean reciprocal rank + hit@3 for: the old system's order,
   hybrid-plus (XGB score) and the bandit (posterior mean). This is a within-slate ranking test:
   it does not need the propensities, because every policy reorders the same 10 products.
   The "old system" row is inflated by position bias (shoppers click the top slots because they are
   on top), so compare hybrid-plus vs RL bandit; the old system is only a reference.
5. Saves state/rl_pretrain.npz. It never touches the live state (state/rl_state.npz); run_rl.py
   starts from the warm start only when there is no live state yet.

Caveat (say it in the presentation): the hybrid's user history covers the whole 60 days, so a
session also "sees" the user's later activity; the offline numbers are optimistic for every
policy that uses the history, the comparison between them is what matters.
"""
from __future__ import annotations

import _paths  # noqa: F401  (must be first: import paths, working dir, model dir)

import hashlib
import json
import os

import numpy as np

import recommender_v2 as v2
import recommender_v6 as v6
import hybrid_plus
import rl_agent

N = int(os.environ.get("RL_PRETRAIN_N", "40000"))
PRETRAIN_WEIGHT = float(os.environ.get("RL_PRETRAIN_WEIGHT", "0.3"))
BATCH = 256

EXPOSURE_SQL = """
SELECT indexOf(shown, clicked::String) AS slot, count() AS clicks
FROM shop.history_sessions
WHERE clicked IS NOT NULL AND clicked::String != ''
GROUP BY slot
ORDER BY slot"""

SESSIONS_SQL = """
SELECT request_id::String AS request_id, user_id::String AS user_id, context, recent_views, shown,
       clicked::String AS clicked, purchased::String AS purchased
FROM shop.history_sessions
WHERE ts::DateTime64(3) > (SELECT max(ts::DateTime64(3)) FROM shop.history_sessions) - INTERVAL 14 DAY
ORDER BY cityHash64(request_id::String)
LIMIT {n}"""


def is_test(rid: str) -> bool:
    return int(hashlib.md5(rid.encode()).hexdigest(), 16) % 5 == 0


def exposure() -> np.ndarray:
    clicks = np.zeros(10)
    for r in v2.query(EXPOSURE_SQL):
        k = int(r["slot"])
        if 1 <= k <= 10:
            clicks[k - 1] = int(r["clicks"])
    if clicks[0] <= 0:
        return rl_agent.DEFAULT_EXPOSURE
    e = clicks / clicks[0]
    return np.maximum.accumulate(e[::-1])[::-1]          # make it non-increasing


def as_obj(x):
    return json.loads(x) if isinstance(x, str) else x


def rank_of(target: str, items: list[str], scores: np.ndarray) -> int | None:
    if target not in items:
        return None
    order = [items[j] for j in np.argsort(-scores, kind="stable")]
    return order.index(target) + 1


def main() -> None:
    hybrid = hybrid_plus.load()
    hybrid.booster.set_param({"device": "cpu"})   # tiny batches: XGB on CPU, transformer stays on GPU
    model = v6.Model()
    agent = rl_agent.RLAgent(hybrid, model, rl_agent.LinTSPBM())
    agent.refresh_catalog()

    e = exposure()
    agent.bandit.exposure = e
    print("exposure by slot:", np.round(e, 3).tolist())

    r = v2.http.post(f"{v2.API}/v1/query", json={"sql": SESSIONS_SQL.format(n=N)}, timeout=600)
    r.raise_for_status()
    sessions = r.json()["data"]
    print(f"{len(sessions)} sessions loaded")

    test = []          # (shown, clicked, purchased, X)
    n_train = 0
    for start in range(0, len(sessions), BATCH):
        batch = []
        for s in sessions[start:start + BATCH]:
            shown = [str(p) for p in (as_obj(s["shown"]) or [])][:10]
            if len(shown) != 10:
                continue
            req = {"request_id": s["request_id"], "user_id": s["user_id"],
                   "context": as_obj(s["context"]) or {}, "recent_views": as_obj(s["recent_views"]) or []}
            batch.append((req, shown, s["clicked"] or "", s["purchased"] or ""))
        if not batch:
            continue
        feats = agent.candidates_and_features([b[0] for b in batch], set(), live=False,
                                              extra=[b[1] for b in batch])
        Xs, slots, ys = [], [], []
        for (req, shown, clicked, purchased), (cands, X, _, _) in zip(batch, feats):
            idx = {p: j for j, p in enumerate(cands)}
            Xshown = X[[idx[p] for p in shown]]
            if is_test(req["request_id"]):
                test.append((shown, clicked, purchased, Xshown))
                continue
            n_train += 1
            for k, p in enumerate(shown):
                Xs.append(Xshown[k])
                slots.append(k)
                ys.append((p == purchased) * 1.0 + (p == clicked) * rl_agent.LAMBDA_CLICK)
        agent.bandit.update(np.array(Xs), np.array(slots), np.array(ys), weight=PRETRAIN_WEIGHT)
        print(f"  {start + len(batch)}/{len(sessions)}")

    print(f"\ntrained on {n_train} sessions ({agent.bandit.n_purchases} purchases), testing on {len(test)}")
    print("learned weights:", json.dumps(agent.bandit.weights(), indent=1))

    theta = agent.bandit.mean
    zi = rl_agent.FEATURES.index("xgb_z")
    for what, col in (("purchase", 2), ("click", 1)):
        res = {"old system": [], "hybrid-plus": [], "RL bandit": []}
        for row in test:
            shown, target, X = row[0], row[col], row[3]
            if not target:
                continue
            for name, sc in (("old system", -np.arange(10.0)), ("hybrid-plus", X[:, zi]), ("RL bandit", X @ theta)):
                k = rank_of(target, shown, sc)
                if k:
                    res[name].append(k)
        print(f"\nheld-out sessions with a {what}: {len(res['old system'])}")
        for name, ks in res.items():
            if ks:
                ks = np.array(ks)
                print(f"  {name:12s}  MRR {np.mean(1 / ks):.3f}   hit@1 {np.mean(ks <= 1):.3f}   hit@3 {np.mean(ks <= 3):.3f}")

    out = rl_agent.STATE_PATH.with_name("rl_pretrain.npz")
    agent.bandit.save(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
