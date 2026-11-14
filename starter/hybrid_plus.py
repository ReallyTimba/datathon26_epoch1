"""Adapter: the teammate's HybridTransformerXGB (recommenderd.py) inside our A/B test.

Two ways to call it:
  * plain: exactly the teammate's model;
  * plus:  the same model, but its candidate pool also gets v6's personal products
           (this session's views, today's activity, the 60-day history without purchases)
           and the products the user bought today are blocked.

The model files (meta.json, sasrec_ctx.pt, xgb_ranker.json) are looked for in HYBRID_DIR
(environment variable, default: the folder you run the program from).
"""
from __future__ import annotations

import os
from pathlib import Path

import recommenderd as rd

HYBRID_DIR = Path(os.environ.get("HYBRID_DIR", "."))
PLUS_LIMIT = 55  # candidates for "plus": the teammate's 45 + room for the personal products


class HybridPlus(rd.HybridTransformerXGB):
    extra_items: list[str] = []

    def get_candidates(self, row_probs, u_prods, unavailable, limit=45):
        if not self.extra_items:  # plain: unchanged
            return super().get_candidates(row_probs, u_prods, unavailable, limit)
        # personal products first among the user products, in our order of interest
        personal = list(dict.fromkeys(list(self.extra_items) + list(u_prods)))
        return super().get_candidates(row_probs, personal, unavailable, PLUS_LIMIT)


def load() -> HybridPlus:
    h = HybridPlus(str(HYBRID_DIR / "xgb_ranker.json"), str(HYBRID_DIR / "sasrec_ctx.pt"),
                   str(HYBRID_DIR / "meta.json"))
    h.init_static_data()
    h.refresh_live_trends()
    return h


def recommend(h: HybridPlus, req: dict, unavailable: set[str],
              extra_items=(), extra_blocked=frozenset()) -> list[str]:
    h.extra_items = list(extra_items)
    try:
        return h.predict_batch([req], set(unavailable) | set(extra_blocked))[0]["items"]
    finally:
        h.extra_items = []


def personal_items(req: dict, model) -> tuple[list[str], set[str]]:
    """v6's personal products for this request (newest / most interesting first) and today's purchases."""
    uid = str(req.get("user_id", ""))
    viewed = [v["product_id"] for v in reversed(req.get("recent_views") or [])
              if isinstance(v, dict) and "product_id" in v]
    items = viewed + model.live_items.get(uid, []) + model.user_items.get(uid, [])
    return list(dict.fromkeys(items)), model.live_bought.get(uid, set())