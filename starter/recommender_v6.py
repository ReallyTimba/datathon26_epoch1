"""v6: v5 + today's activity of the same user + the user's favourite subcategories.

Live A/B: v5 (personal history) ~1.11% vs v2 ~0.95% vs baseline ~0.71%. v6 uses the
personal signal more fully:

  1. 2 products viewed in this session (newest first)
  2. up to 3 products this user viewed / carted TODAY (shop.events, live, refreshed every
     30 s in a background thread so it never delays the answers)
  3. up to 4 products from the user's 60-day history (as v5)
  4. the most popular product of each of the user's 3 favourite subcategories (history)
  5. fill like v2b

Products the user bought today are never recommended (README: buyers rarely rebuy), and
products bought in the history were already left out by v5's query.
New users have no history, but they do have live events once they start browsing, so
step 2 also helps the cold-start score.
"""
from __future__ import annotations

import threading
import time

import recommender_v2 as v2
import recommender_v5 as v5

MAX_SESSION = 2
MAX_LIVE = 3
MAX_HISTORY = 4
LIVE_SECONDS = 30

USER_SUBCATEGORIES = """
SELECT u, sub
FROM (
    SELECT e.user_id::String AS u, c.subcategory::String AS sub,
           sum(multiIf(e.event_type::String = 'add_to_cart', 3, e.event_type::String = 'purchase', 2, 1)) AS w
    FROM shop.history_events AS e
    INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
    GROUP BY u, sub
)
ORDER BY u, w DESC
LIMIT 3 BY u"""
"""For every user, their 3 favourite subcategories in the history."""

LIVE_USER_ITEMS = """
SELECT user_id::String AS u, product_id::String AS p,
       sum(multiIf(event_type::String = 'add_to_cart', 3, event_type::String = 'view', 1, 0)) AS w,
       max(event_type::String = 'purchase') AS bought,
       max(ts::DateTime64(3)) AS last_ts
FROM shop.events
WHERE ts > now() - INTERVAL 3 HOUR
GROUP BY u, p
ORDER BY u, bought DESC, w DESC, last_ts DESC
LIMIT 15 BY u"""
"""What every user did in the store in the last 3 hours (bought products first, so they are never cut)."""


def load_user_subcategories() -> dict[str, list[str]]:
    r = v2.http.post(f"{v2.API}/v1/query", json={"sql": USER_SUBCATEGORIES}, timeout=600)
    r.raise_for_status()
    subs: dict[str, list[str]] = {}
    for row in r.json()["data"]:
        subs.setdefault(row["u"], []).append(row["sub"])
    return subs


def load_live() -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    items: dict[str, list[str]] = {}
    bought: dict[str, set[str]] = {}
    for row in v2.query(LIVE_USER_ITEMS):
        if row["bought"] in (True, 1, "1", "true"):
            bought.setdefault(row["u"], set()).add(row["p"])
        elif int(row["w"]) > 0:
            items.setdefault(row["u"], []).append(row["p"])
    return items, bought


class Model(v5.Model):
    def __init__(self) -> None:
        try:
            self.user_subs = load_user_subcategories()
            print(f"favourite subcategories loaded for {len(self.user_subs)} users")
        except Exception as e:
            print(f"user subcategories failed: {e}")
            self.user_subs = {}
        self.live_items: dict[str, list[str]] = {}
        self.live_bought: dict[str, set[str]] = {}
        super().__init__()
        threading.Thread(target=self._live_loop, daemon=True).start()

    def _live_loop(self) -> None:
        """Background thread: refresh today's activity without blocking the answers."""
        while True:
            try:
                items, bought = load_live()
                self.live_items, self.live_bought = items, bought   # swap in one go
                print(f"live activity: {len(items)} users, {sum(map(len, bought.values()))} purchases")
            except Exception as e:
                print(f"live activity failed: {e}")
            time.sleep(LIVE_SECONDS)


def recommend(req: dict, model: Model, unavailable: set[str]) -> list[str]:
    uid = str(req.get("user_id", ""))
    blocked = unavailable | model.live_bought.get(uid, set())
    out: list[str] = []

    def add(products, limit: int) -> None:
        added = 0
        for p in products:
            if len(out) >= 10 or added >= limit:
                return
            if p in model.info and p not in blocked and p not in out:
                out.append(p)
                added += 1

    recent_views = req.get("recent_views") or []
    viewed = [v["product_id"] for v in reversed(recent_views) if isinstance(v, dict) and "product_id" in v]
    session = list(dict.fromkeys(p for p in viewed if p in model.info))

    add(session, MAX_SESSION)                                   # 1. this session
    add(model.live_items.get(uid, []), MAX_LIVE)                # 2. today, outside the carousel
    add(model.user_items.get(uid, []), MAX_HISTORY)             # 3. 60-day history
    for sub in model.user_subs.get(uid, []):                    # 4. favourite subcategories
        add(model.by_sub.get(sub, []), 1)
    for p in v2.recommend(recent_views, model, blocked, max_viewed=MAX_SESSION):  # 5. fill like v2b
        add([p], 1)
    return out
