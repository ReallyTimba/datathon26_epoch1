"""v4: v2 (viewed products first) + co-views instead of most of the subcategory slots.

Live A/B (leaderboard, same time window): v2-viewed ~0.88% vs baseline ~0.56%; ranking by
lift did not beat best sellers. So we keep v2's order and strengthen the "what is this
shopper interested in" signal: after the viewed products come the products other shoppers
viewed in the same sessions (co-views) as those products.

recommend: viewed products (newest first) -> co-viewed products -> same subcategory ->
           same category -> overall popularity.
"""
from __future__ import annotations

from collections import Counter

import recommender_v2 as v2

MAX_VIEWED = 5
MAX_COVIEW = 4
PER_SUBCATEGORY = 1
PER_CATEGORY = 1
DECAY = 0.7  # older views count less: weight 1, 0.7, 0.49, ...

COVIEWS = """
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'view') AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
            FROM shop.history_events WHERE event_type = 'view') AS b
  ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY p, q
ORDER BY n DESC
LIMIT 20 BY p"""
"""For every product, the 20 products most often viewed in the same sessions."""


def load_coviews() -> dict[str, list[str]]:
    r = v2.http.post(f"{v2.API}/v1/query", json={"sql": COVIEWS}, timeout=600)
    r.raise_for_status()
    cov: dict[str, list[str]] = {}
    for row in r.json()["data"]:  # already sorted by n DESC inside each product
        cov.setdefault(row["p"], []).append(row["q"])
    return cov


class Model(v2.Model):
    def __init__(self) -> None:
        try:
            self.cov = load_coviews()
            print(f"co-views loaded for {len(self.cov)} products")
        except Exception as e:  # without co-views v4 behaves like v2
            print(f"co-views failed: {e}")
            self.cov = {}
        super().__init__()


def recommend(recent_views, model: Model, unavailable: set[str]) -> list[str]:
    out: list[str] = []

    def add(products, limit: int) -> None:
        added = 0
        for p in products:
            if len(out) >= 10 or added >= limit:
                return
            if p in model.info and p not in unavailable and p not in out:
                out.append(p)
                added += 1

    viewed = [v["product_id"] for v in reversed(recent_views or []) if isinstance(v, dict) and "product_id" in v]
    known = list(dict.fromkeys(p for p in viewed if p in model.info))  # unique, newest first

    add(known, MAX_VIEWED)                                   # 1. what they just viewed
    score: Counter[str] = Counter()                          # 2. co-viewed with it
    for k, p in enumerate(known[:3]):  
        for rank, q in enumerate(model.cov.get(p, [])):
            score[q] += DECAY ** k / (rank + 1)
    add([q for q, _ in score.most_common()], MAX_COVIEW)
    for p in known[:3]:                                      # 3. same subcategory / category
        add(model.by_sub[model.info[p][1]], PER_SUBCATEGORY)
    for p in known[:3]:
        add(model.by_cat[model.info[p][0]], PER_CATEGORY)
    add(model.ranked, 10)                                    # 4. overall popularity
    return out