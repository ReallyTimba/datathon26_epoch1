"""v3: rank products by position-corrected lift from the history.

Offline findings (team feedback + shop.history_sessions):
  * the slot matters most: slot 1 sells ~2.5x slot 2 and ~8x slots 6-10;
  * comparing products is only fair on the same slot, so for every product we compute
        lift = (purchases + 5) / (purchases an average product would get on the same slots + 5)
    and put the products with the highest lift first.

recommend_v3:  the 10 active, in-stock products with the highest lift.
recommend_v3b: the same, but the last viewed product is put in slot 2.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import recommender_v2 as v2

SMOOTHING = 5

LIFT_SQL = """
SELECT x.1 AS p, x.2 AS slot, count() AS shows, countIf(b = x.1) AS buys
FROM (
    SELECT arrayJoin(arrayZip(shown::Array(String), arrayEnumerate(shown::Array(String)))) AS x,
           ifNull(purchased::Nullable(String), '') AS b
    FROM shop.history_sessions
)
GROUP BY p, slot"""


def compute_lift() -> dict[str, float]:
    rows = v2.query(LIFT_SQL)
    shows_by_slot: Counter[int] = Counter()
    buys_by_slot: Counter[int] = Counter()
    for r in rows:
        shows_by_slot[int(r["slot"])] += int(r["shows"])
        buys_by_slot[int(r["slot"])] += int(r["buys"])
    slot_rate = {s: buys_by_slot[s] / shows_by_slot[s] for s in shows_by_slot if shows_by_slot[s]}

    buys: Counter[str] = Counter()
    expected: defaultdict[str, float] = defaultdict(float)
    for r in rows:
        buys[r["p"]] += int(r["buys"])
        expected[r["p"]] += int(r["shows"]) * slot_rate.get(int(r["slot"]), 0.0)
    return {p: (buys[p] + SMOOTHING) / (expected[p] + SMOOTHING) for p in expected}


class Model(v2.Model):
    """v2's catalogue / popularity model plus a lift ranking (computed once: the history does not change)."""

    def __init__(self) -> None:
        self.lift = compute_lift()
        print(f"lift computed for {len(self.lift)} products")
        super().__init__()

    def rebuild(self) -> None:
        super().rebuild()
        # unknown products (e.g. new during the event) get lift 1.0 = "average"
        self.by_lift = sorted(self.info, key=lambda p: (self.lift.get(p, 1.0), self.score[p]), reverse=True)


def recommend_v3(recent_views, model: Model, unavailable: set[str]) -> list[str]:
    return [p for p in model.by_lift if p not in unavailable][:10]


def recommend_v3b(recent_views, model: Model, unavailable: set[str]) -> list[str]:
    top = recommend_v3(recent_views, model, unavailable)
    viewed = [v["product_id"] for v in reversed(recent_views or []) if isinstance(v, dict) and "product_id" in v]
    last = next((p for p in viewed if p in model.info and p not in unavailable), None)
    if last is None:
        return top
    rest = [p for p in top if p != last]
    return (rest[:1] + [last] + rest[1:])[:10]