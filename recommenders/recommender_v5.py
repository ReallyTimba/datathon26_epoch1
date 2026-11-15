"""v5: personalisation from the shopper's own history.

Offline (shop.history_events, 3000 buyers): 49% of purchases were a product the same user had
viewed / carted in an EARLIER session, 37% a product from a subcategory they had viewed
earlier. Live, ~70% of requests come from returning users, and all of them are in the history.

For every user we load once their 10 most interesting products from the history
(add_to_cart = 3, view = 1, products they already bought are left out), then:

recommend(..., history_first=True):  user's history items -> 2 viewed in this session -> v2
recommend(..., history_first=False): 2 viewed in this session -> user's history items -> v2
New users have no history, so they get v2 with 2 viewed products first (v2b).
"""
from __future__ import annotations

import recommender_v2 as v2

MAX_HISTORY = 4
MAX_SESSION = 2

USER_ITEMS = """
SELECT u, p
FROM (
    SELECT user_id::String AS u, product_id::String AS p,
           sum(multiIf(event_type::String = 'add_to_cart', 3, event_type::String = 'view', 1, 0)) AS w,
           max(event_type::String = 'purchase') AS bought,
           max(ts::DateTime64(3)) AS last_ts
    FROM shop.history_events
    GROUP BY u, p
    HAVING bought = 0 AND w > 0
)
ORDER BY u, w DESC, last_ts DESC
LIMIT 10 BY u"""
"""For every user, their 10 most interacted-with products they have not bought."""


def load_user_items() -> dict[str, list[str]]:
    r = v2.http.post(f"{v2.API}/v1/query", json={"sql": USER_ITEMS}, timeout=600)
    r.raise_for_status()
    items: dict[str, list[str]] = {}
    for row in r.json()["data"]:  # already ordered by interest inside each user
        items.setdefault(row["u"], []).append(row["p"])
    return items


class Model(v2.Model):
    def __init__(self) -> None:
        try:
            self.user_items = load_user_items()
            print(f"history loaded for {len(self.user_items)} users")
        except Exception as e:  # without it v5 behaves like v2b
            print(f"user history failed: {e}")
            self.user_items = {}
        super().__init__()


def recommend(req: dict, model: Model, unavailable: set[str], history_first: bool = True) -> list[str]:
    out: list[str] = []

    def add(products, limit: int) -> None:
        added = 0
        for p in products:
            if len(out) >= 10 or added >= limit:
                return
            if p in model.info and p not in unavailable and p not in out:
                out.append(p)
                added += 1

    recent_views = req.get("recent_views") or []
    viewed = [v["product_id"] for v in reversed(recent_views) if isinstance(v, dict) and "product_id" in v]
    session = list(dict.fromkeys(p for p in viewed if p in model.info))   # newest first
    history = model.user_items.get(str(req.get("user_id", "")), [])

    if history_first:
        add(history, MAX_HISTORY)
        add(session, MAX_SESSION)
    else:
        add(session, MAX_SESSION)
        add(history, MAX_HISTORY)
    for p in v2.recommend(recent_views, model, unavailable, max_viewed=MAX_SESSION):  # fill like v2b
        add([p], 1)
    return out