"""Production runner: the teammate's hybrid model fed with v6's personal candidates.

    python starter/run_hybrid_plus.py

Every request is answered by "hybrid-plus" (recommenderd.py + hybrid_plus.py).
Safety net, so that a broken model never costs answers:
  * if the hybrid model cannot be loaded at start, every request is answered by v6;
  * if it fails on one request, that request is answered by v6 ("v6-fallback");
  * if v6 fails too, the official baseline.
The model_version tells which one answered.
"""
from __future__ import annotations

import time

import recommender as base
import recommender_v2 as v2
import recommender_v6 as v6

try:  # the teammate's model needs torch + xgboost + its model files
    import hybrid_plus
except Exception as e:
    print(f"hybrid model not available, answering with v6: {e}")
    hybrid_plus = None


def baseline(model, unavailable, best):
    return [p for p in best if p in model.info and p not in unavailable][:10]


def answer(req, model, hybrid, unavailable, best):
    if hybrid is not None:
        try:
            items = hybrid_plus.recommend(hybrid, req, unavailable, *hybrid_plus.personal_items(req, model))
            if len(set(items)) == 10:
                return items, "hybrid-plus"
        except Exception as e:
            print(f"hybrid-plus failed: {e}")
    try:
        return v6.recommend(req, model, unavailable), "v6-fallback" if hybrid is not None else "v6-personal"
    except Exception as e:
        print(f"v6 failed: {e}")
        return baseline(model, unavailable, best), "baseline-fallback"


def main() -> None:
    hybrid = None
    if hybrid_plus is not None:
        try:
            hybrid = hybrid_plus.load()
        except Exception as e:
            print(f"hybrid model failed to load, answering with v6: {e}")
    model = v6.Model()
    best = base.best_sellers()
    popularity_loaded = catalog_loaded = time.time()
    seen = set[str]()
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
                except Exception as e:
                    print(f"catalog refresh failed: {e}")
            if hybrid is not None and now - trends_checked >= 15:
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
            answers = []
            for req in v2.pull(seen):
                items, version = answer(req, model, hybrid, unavailable, best)
                answers.append({"request_id": req["request_id"], "items": items, "model_version": version})
            if answers:
                v2.push(answers)
        except Exception as e:  # keep running: shoppers keep arriving while you sleep
            print(f"error: {e}")
            time.sleep(2)
        time.sleep(0.3)


if __name__ == "__main__":
    main()