"""v2: the products the shopper just viewed first, then their subcategory / category.

Offline check on shop.history_sessions: a carousel slot showing a product the shopper had
already viewed converted ~1.8% of the time, vs ~0.22% for the same subcategory and ~0.10%
for anything else. So viewed products go first.

    python starter/recommender_v2.py     # reads TEAM_ID and RAWTREE_API_KEY from .env or the environment

Same pull -> predict -> push loop as the baseline. Differences:
  * every 5 minutes it reloads the current catalogue and a popularity score
    (60 days of history + the last 2 hours of the live stream, weighted higher);
  * recommend() fills the 10 slots with popular products from the subcategories,
    then the categories, of the last viewed products, then overall popularity.
"""
from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

MODEL_VERSION = "v2-viewed"
REFRESH_SECONDS = 300   # how often popularity is recomputed
CATALOG_SECONDS = 60    # how often the catalogue (new products, activity) is reloaded
MAX_VIEWED = 5          # how many viewed products go first
PER_SUBCATEGORY = 3
PER_CATEGORY = 2
RECENT_WEIGHT = 20      # one purchase in the last 2 hours counts like 20 purchases in the history

# ---------------------------------------------------------------- SQL

CATALOG = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id"""
"""Current catalogue: shop.catalog with the latest update of each product on top."""

HISTORY_SALES = """
SELECT product_id::String AS product_id, count() AS n
FROM shop.history_events
WHERE event_type = 'purchase'
GROUP BY product_id"""

RECENT_SALES = """
SELECT product_id::String AS product_id, count() AS n
FROM shop.events
WHERE event_type = 'purchase' AND ts > now() - INTERVAL 2 HOUR
GROUP BY product_id"""

NEW_REQUESTS = """
SELECT request_id, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 50"""

OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""

Record = dict[str, Any]


def query(sql: str) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]


# ---------------------------------------------------------------- model

class Model:
    """Everything recommend() needs. Popularity every REFRESH_SECONDS, catalogue every CATALOG_SECONDS."""

    def __init__(self) -> None:
        self.score: Counter[str] = Counter()
        self.info: dict[str, tuple[str, str]] = {}
        self.load_popularity()
        self.load_catalog()

    def load_popularity(self) -> None:
        score: Counter[str] = Counter()
        for r in query(HISTORY_SALES):
            score[r["product_id"]] += int(r["n"])
        try:
            for r in query(RECENT_SALES):
                score[r["product_id"]] += RECENT_WEIGHT * int(r["n"])
        except Exception as e:  # shop.events can be empty before the stream starts
            print(f"recent sales not available: {e}")
        self.score = score
        self.rebuild()

    def load_catalog(self) -> None:
        info: dict[str, tuple[str, str]] = {}
        for r in query(CATALOG):
            if r["is_active"] in (True, 1, "true", "1"):
                info[r["product_id"]] = (r["category"], r["subcategory"])
        self.info = info
        self.rebuild()

    def rebuild(self) -> None:
        """Active products, most popular first; the same ranking split by subcategory / category."""
        ranked = sorted(self.info, key=lambda p: self.score[p], reverse=True)
        by_sub: dict[str, list[str]] = defaultdict(list)
        by_cat: dict[str, list[str]] = defaultdict(list)
        for p in ranked:
            category, subcategory = self.info[p]
            by_sub[subcategory].append(p)
            by_cat[category].append(p)
        self.ranked, self.by_sub, self.by_cat = ranked, by_sub, by_cat
        print(f"model rebuilt: {len(self.info)} active products, {len(by_sub)} subcategories")


def recommend(recent_views: list[Record] | None, model: Model, unavailable: set[str],
              max_viewed: int = MAX_VIEWED) -> list[str]:
    """10 distinct, active, in-stock product IDs for one shopper."""
    out: list[str] = []

    def add(products: list[str], limit: int) -> None:
        added = 0
        for p in products:
            if len(out) >= 10 or added >= limit:
                return
            if p in model.info and p not in unavailable and p not in out:
                out.append(p)
                added += 1

    # last viewed products first (recent_views is in chronological order)
    viewed = [v["product_id"] for v in reversed(recent_views or []) if isinstance(v, dict) and "product_id" in v]
    known = list(dict.fromkeys(p for p in viewed if p in model.info))  # unique, newest first

    add(known, max_viewed)                   # 1. the products they just viewed, newest first
    for p in known[:3]:                      # 2. same subcategory as the last viewed ones
        add(model.by_sub[model.info[p][1]], PER_SUBCATEGORY)
    for p in known[:3]:                      # 3. same category
        add(model.by_cat[model.info[p][0]], PER_CATEGORY)
    add(model.ranked, 10)                    # 4. fill the rest with overall popularity
    return out


# ---------------------------------------------------------------- loop

def out_of_stock() -> set[str]:
    return {r["product_id"] for r in query(OUT_OF_STOCK)}


def pull(seen: set[str]) -> list[Record]:
    fresh = []
    for req in query(NEW_REQUESTS):
        if req["request_id"] not in seen:
            seen.add(req["request_id"])
            fresh.append(req)
    return fresh


def predict(reqs: list[Record], model: Model, unavailable: set[str]) -> list[Record]:
    return [{"request_id": req["request_id"],
             "items": recommend(req.get("recent_views"), model, unavailable),
             "model_version": MODEL_VERSION} for req in reqs]


def push(answers: list[Record]) -> None:
    http.post(f"{API}/v1/tables/submissions", params={"database": TEAM}, json=answers, timeout=10).raise_for_status()
    print(f"answered {len(answers)}")


def main() -> None:
    model: Model | None = None
    popularity_loaded = catalog_loaded = 0.0
    seen = set[str]()
    unavailable: set[str] = set()
    stock_checked = 0.0
    while True:
        try:
            if model is None:
                model = Model()
                popularity_loaded = catalog_loaded = time.time()
            if time.time() - popularity_loaded >= REFRESH_SECONDS:
                popularity_loaded = time.time()
                try:
                    model.load_popularity()
                except Exception as e:  # keep answering with the previous popularity
                    print(f"popularity refresh failed: {e}")
            if time.time() - catalog_loaded >= CATALOG_SECONDS:
                catalog_loaded = time.time()
                try:
                    model.load_catalog()
                except Exception as e:
                    print(f"catalog refresh failed: {e}")
            if time.time() - stock_checked >= 1:
                stock_checked = time.time()
                try:
                    unavailable = out_of_stock()
                except Exception as e:
                    print(f"stock check failed: {e}")
            answers = predict(pull(seen), model, unavailable)
            if answers:
                push(answers)
        except Exception as e:  # keep running: shoppers keep arriving while you sleep
            print(f"error: {e}")
            time.sleep(2)
        time.sleep(0.3)


if __name__ == "__main__":
    main()