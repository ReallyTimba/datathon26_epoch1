"""A/B test: v6 vs the teammate's hybrid model vs the hybrid model fed with v6's personal candidates.

    python starter/ab_test.py

Every request goes to one policy, chosen by a hash of its request_id (random, but always
the same for the same request). The policy name is written in model_version, so the
feedback can be compared per policy over exactly the same shoppers and hours.
"""
from __future__ import annotations

import hashlib
import time

import recommender as base
import recommender_v2 as v2
import recommender_v6 as v6

try:  # the teammate's model needs torch + xgboost + its model files
    import hybrid_plus
except Exception as e:
    print(f"hybrid model not available, running without it: {e}")
    hybrid_plus = None
HYBRID = None  # loaded in main()


def baseline(recent_views, model, unavailable, best):
    """The official baseline: all-time best sellers, active and in stock."""
    return [p for p in best if p in model.info and p not in unavailable][:10]


POLICIES = {
    "ab-v6-personal": lambda req, m, u, best: v6.recommend(req, m, u),
}
if hybrid_plus is not None:
    POLICIES["ab-hybrid"] = lambda req, m, u, best: hybrid_plus.recommend(HYBRID, req, u)
    POLICIES["ab-hybrid-plus"] = lambda req, m, u, best: hybrid_plus.recommend(
        HYBRID, req, u, *hybrid_plus.personal_items(req, m))
NAMES = list(POLICIES)


def arm(request_id: str) -> str:
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return NAMES[h % len(NAMES)]


def predict(reqs, model, unavailable, best):
    answers = []
    for req in reqs:
        name = arm(req["request_id"])
        try:
            items = POLICIES[name](req, model, unavailable, best)
        except Exception as e:  # a bug in one policy must not lose the answer
            print(f"{name} failed: {e}")
            name, items = "ab-fallback-baseline", baseline(None, model, unavailable, best)
        answers.append({"request_id": req["request_id"], "items": items, "model_version": name})
    return answers


def main() -> None:
    global HYBRID
    model = None
    best: list[str] = []
    popularity_loaded = catalog_loaded = 0.0
    seen = set[str]()
    unavailable: set[str] = set()
    stock_checked = trends_checked = 0.0
    while True:
        try:
            if model is None:
                if hybrid_plus is not None and HYBRID is None:
                    HYBRID = hybrid_plus.load()
                model = v6.Model()
                best = base.best_sellers()
                popularity_loaded = catalog_loaded = time.time()
            if time.time() - popularity_loaded >= v2.REFRESH_SECONDS:
                popularity_loaded = time.time()
                try:
                    model.load_popularity()
                except Exception as e:
                    print(f"popularity refresh failed: {e}")
            if time.time() - catalog_loaded >= v2.CATALOG_SECONDS:
                catalog_loaded = time.time()
                try:
                    model.load_catalog()
                except Exception as e:
                    print(f"catalog refresh failed: {e}")
            if HYBRID is not None and time.time() - trends_checked >= 15:
                trends_checked = time.time()
                try:
                    HYBRID.refresh_live_trends()
                except Exception as e:
                    print(f"trends refresh failed: {e}")
            if time.time() - stock_checked >= 1:
                stock_checked = time.time()
                try:
                    unavailable = v2.out_of_stock()
                except Exception as e:
                    print(f"stock check failed: {e}")
            answers = predict(v2.pull(seen), model, unavailable, best)
            if answers:
                v2.push(answers)
        except Exception as e:  # keep running: shoppers keep arriving while you sleep
            print(f"error: {e}")
            time.sleep(2)
        time.sleep(0.3)


if __name__ == "__main__":
    main()