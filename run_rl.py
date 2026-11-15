"""Production runner for the RL agent (contextual bandit on top of hybrid-plus), with a live A/B test.

    python starter/run_rl.py              # from any folder
    RL_SHARE=1.0 python run_rl.py          # every request to the RL agent (final round, once it wins)
    RL_SHARE=0.5 python run_rl.py          # default: half RL, half hybrid-plus, split by request_id hash

model_version tells who answered: "rl-lints", "hybrid-plus", or a fallback.
The bandit learns from BOTH arms (it logs the features of what hybrid-plus showed too),
so the A/B test does not slow its learning down.
Safety net, so that a broken model never costs answers:
  RL fails -> hybrid-plus -> v6 -> official baseline.
State is saved to state/rl_state.npz every minute and loaded at start, so a restart keeps what it learned.
"""
from __future__ import annotations

import _paths  # noqa: F401  (must be first: import paths, working dir, model dir)

import hashlib
import os
import threading
import time

import recommender as base
import recommender_v2 as v2
import recommender_v6 as v6
import hybrid_plus
import rl_agent

RL_SHARE = float(os.environ.get("RL_SHARE", "0.5"))


def is_rl(request_id: str) -> bool:
    h = int(hashlib.md5(("rl:" + request_id).encode()).hexdigest(), 16)
    return (h % 10_000) < RL_SHARE * 10_000


def valid(items, model, unavailable) -> bool:
    return len(items) == 10 and len(set(items)) == 10 and all(p in model.info and p not in unavailable for p in items)


def answer_batch(reqs, agent, hybrid, model, unavailable, best):
    answers, to_log = [], []
    rl_reqs = [r for r in reqs if is_rl(r["request_id"])]
    rl_items: dict[str, list[str]] = {}
    if rl_reqs:
        try:
            for req, items in zip(rl_reqs, agent.act(rl_reqs, unavailable)):
                rl_items[req["request_id"]] = items
        except Exception as e:
            print(f"RL batch failed: {e}")
    for req in reqs:
        rid = req["request_id"]
        items, version = rl_items.get(rid), "rl-lints"
        if items is None or not valid(items, model, unavailable):
            try:
                items, version = hybrid_plus.recommend(hybrid, req, unavailable,
                                                       *hybrid_plus.personal_items(req, model)), "hybrid-plus"
                if rid not in rl_items:     # hybrid arm: log its features after pushing, to learn from it too
                    to_log.append((req, items))
            except Exception as e:
                print(f"hybrid-plus failed: {e}")
                items = None
        if items is None or not valid(items, model, unavailable):
            try:
                items, version = v6.recommend(req, model, unavailable), "v6-fallback"
            except Exception as e:
                print(f"v6 failed: {e}")
                items, version = [p for p in best if p in model.info and p not in unavailable][:10], "baseline-fallback"
        answers.append({"request_id": rid, "items": items, "model_version": version})
    return answers, to_log


def log_other_arm(agent, to_log, unavailable) -> None:
    for req, items in to_log:
        try:
            agent.log_decision(req["request_id"], items, agent.features_for(req, items, unavailable))
        except Exception as e:
            print(f"feature logging failed: {e}")


def main() -> None:
    hybrid = hybrid_plus.load()
    hybrid.booster.set_param({"device": "cpu"})   # tiny batches: XGB on CPU, transformer stays on GPU
    model = v6.Model()
    agent = rl_agent.RLAgent(hybrid, model)
    agent.refresh_catalog()
    best = base.best_sellers()
    threading.Thread(target=agent.learn_forever, args=(v2.TEAM,), daemon=True).start()
    print(f"RL agent running, RL_SHARE={RL_SHARE}, explore={rl_agent.EXPLORE}")

    popularity_loaded = catalog_loaded = time.time()
    seen: set[str] = set()
    unavailable: set[str] = set()
    stock_checked = trends_checked = 0.0
    while True:
        try:
            now = time.time()
            if now - popularity_loaded >= v2.REFRESH_SECONDS:
                popularity_loaded = now
                try:
                    model.load_popularity()
                except Exception as e:
                    print(f"popularity refresh failed: {e}")
            if now - catalog_loaded >= v2.CATALOG_SECONDS:
                catalog_loaded = now
                try:
                    model.load_catalog()
                    agent.refresh_catalog()
                except Exception as e:
                    print(f"catalog refresh failed: {e}")
            if now - trends_checked >= 15:
                trends_checked = now
                try:
                    hybrid.refresh_live_trends()
                except Exception as e:
                    print(f"trends refresh failed: {e}")
            if now - stock_checked >= 1:
                stock_checked = now
                try:
                    unavailable = v2.out_of_stock()
                except Exception as e:
                    print(f"stock check failed: {e}")
            reqs = v2.pull(seen)
            if reqs:
                answers, to_log = answer_batch(reqs, agent, hybrid, model, unavailable, best)
                v2.push(answers)
                log_other_arm(agent, to_log, unavailable)
        except Exception as e:  # keep running: shoppers keep arriving while you sleep
            print(f"error: {e}")
            time.sleep(2)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
