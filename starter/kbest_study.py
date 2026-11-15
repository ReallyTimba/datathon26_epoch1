"""SelectKBest Feature Selection & Ablation Curve (kbest_study.py)"""
from __future__ import annotations

import math
import lightgbm as lgb
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

try:
    from loco4 import Loco4Recommender as RecommenderEngine
    from loco4 import FEATURE_NAMES, TRAIN_SESSIONS_SQL, query
    MODEL_NAME = "loco4"
except ImportError:
    from loco3 import Loco3Recommender as RecommenderEngine
    from loco3 import FEATURE_NAMES, TRAIN_SESSIONS_SQL, query
    MODEL_NAME = "loco3"


def eval_hr1_ndcg3(probs: np.ndarray, groups: list[int], targets: list[int]) -> tuple[float, float]:
    hr1, ndcg3, offset = [], [], 0
    for g, t_idx in zip(groups, targets):
        s_slice = probs[offset : offset + g]
        offset += g
        order = np.argsort(-s_slice)
        hr1.append(1.0 if order[0] == t_idx else 0.0)
        ndcg = 0.0
        for rank, idx in enumerate(order[:3]):
            if idx == t_idx:
                ndcg = 1.0 / math.log2(rank + 2)
                break
        ndcg3.append(ndcg)
    return float(np.mean(hr1)), float(np.mean(ndcg3))


def build_dataset(engine, sessions):
    X_blocks, y_blocks, groups, targets = [], [], [], []
    for s in sessions:
        target = s["purchased"]
        rv = s.get("recent_views") or []
        ctx = s.get("context") if isinstance(s.get("context"), dict) else {}

        out = engine._get_candidates(rv, engine.inactive, ctx)
        cands, cov_m, txt_m, t_cat, l_sub, avg_p, v_info = out
        if target not in cands:
            cands = [target] + cands[:49]

        tr_s = engine._score_transformer_batch([rv], [cands])[0]
        X_i = engine._build_lgb_features(
            cands, tr_s, cov_m, txt_m, t_cat, l_sub, avg_p, v_info, len(rv), ctx
        )
        t_idx = cands.index(target)
        y_i = np.zeros(len(cands), dtype=np.int32)
        y_i[t_idx] = 1

        X_blocks.append(X_i)
        y_blocks.append(y_i)
        groups.append(len(cands))
        targets.append(t_idx)
    return np.vstack(X_blocks), np.concatenate(y_blocks), groups, targets


def main():
    print(f"[kbest] Cargando caché de {MODEL_NAME}...")
    engine = RecommenderEngine(until=None)
    raw = query(TRAIN_SESSIONS_SQL.format(cut=""), timeout=120)
    purchases = [s for s in raw if (s.get("purchased") or "") in engine.prod2idx]

    train_sess = purchases[2500:7500]
    val_sess = purchases[:2500]

    X_tr, y_tr, _, _ = build_dataset(engine, train_sess)
    X_val, y_val, val_groups, val_targets = build_dataset(engine, val_sess)

    # 1. Puntuaciones SelectKBest (Mutual Info + ANOVA F)
    skb_mi = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k="all")
    skb_mi.fit(X_tr, y_tr)

    skb_f = SelectKBest(score_func=f_classif, k="all")
    skb_f.fit(X_tr, y_tr)

    print(f"\n{'Feature':22s} | {'Mutual Info':>11s} | {'ANOVA F-Score':>13s}")
    print("-" * 54)
    order_mi = np.argsort(-skb_mi.scores_)
    for idx in order_mi:
        f_val = 0.0 if np.isnan(skb_f.scores_[idx]) else skb_f.scores_[idx]
        print(f"{FEATURE_NAMES[idx]:22s} | {skb_mi.scores_[idx]:11.5f} | {f_val:13.2f}")

    # 2. Curva de rendimiento real al quedarnos con las K mejores features
    print(f"\n{'K Features':12s} | {'Val HR@1':>10s} | {'Val NDCG@3':>10s} | Features descartadas")
    print("-" * 78)
    for k in [4, 6, 8, 10, 12, 14, len(FEATURE_NAMES)]:
        if k > len(FEATURE_NAMES):
            continue
        selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k=k)
        X_tr_k = selector.fit_transform(X_tr, y_tr)
        X_val_k = selector.transform(X_val)

        mask = selector.get_support()
        dropped = [FEATURE_NAMES[i] for i, keep in enumerate(mask) if not keep]

        clf = lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.05, num_leaves=15,
            min_child_samples=80, reg_lambda=3.0, random_state=42, n_jobs=-1, verbose=-1
        )
        clf.fit(X_tr_k, y_tr)
        probs = clf.booster_.predict(X_val_k)
        hr1, ndcg3 = eval_hr1_ndcg3(probs, val_groups, val_targets)
        dropped_str = ", ".join(dropped) if dropped else "Ninguna (Modelo completo)"
        print(f"k = {k:<8d} | {hr1:9.4%} | {ndcg3:9.4%} | {dropped_str}")


if __name__ == "__main__":
    main()