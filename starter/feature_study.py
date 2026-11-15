"""Offline Feature Importance, SHAP & Ablation Study (feature_study.py)
Evaluates loco3 (or loco2) cached models on real purchase sessions (`shop.history_sessions`):
1. Univariate ROC-AUC per feature (positive purchased item vs non-purchased candidates).
2. LightGBM Split & Gain Feature Importances.
3. Permutation Importance directly on business ranking metrics:
   - HitRate@1 (Slot 1 conversion accuracy) & NDCG@3.
   - Broken down separately for Active Users (`history_len > 0`) vs Cold-Start Users (`history_len == 0`).
4. SHAP TreeExplainer directional analysis (positive vs negative push on purchase probability),
   saving a clean summary report (`feature_study_report.txt`) and optional PNG plots (`shap_summary.png`, `perm_importance.png`) if matplotlib/shap are installed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

# Import the active champion engine (falls back cleanly to loco2 if loco3 isn't present)
try:
    from loco3 import Loco3Recommender as RecommenderEngine
    from loco3 import FEATURE_NAMES, TRAIN_SESSIONS_SQL, query
    MODEL_NAME = "loco3"
except ImportError:
    from loco2 import Loco2Recommender as RecommenderEngine
    from loco2 import FEATURE_NAMES, TRAIN_SESSIONS_SQL, query
    MODEL_NAME = "loco2"


def compute_ndcg_at_k(scores: np.ndarray, target_idx: int, k: int = 3) -> float:
    order = np.argsort(-scores)[:k]
    for rank, idx in enumerate(order):
        if idx == target_idx:
            return 1.0 / math.log2(rank + 2)
    return 0.0


def evaluate_ranking(
    probs: np.ndarray,
    groups: list[int],
    targets: list[int],
    is_cold_mask: list[bool],
) -> dict[str, float]:
    hr1_all, ndcg3_all = [], []
    hr1_act, ndcg3_act = [], []
    hr1_cold, ndcg3_cold = [], []

    offset = 0
    for g, t_idx, is_cold in zip(groups, targets, is_cold_mask):
        s_slice = probs[offset : offset + g]
        offset += g

        top1 = int(np.argmax(s_slice))
        hit1 = 1.0 if top1 == t_idx else 0.0
        ndcg3 = compute_ndcg_at_k(s_slice, t_idx, k=3)

        hr1_all.append(hit1)
        ndcg3_all.append(ndcg3)
        if is_cold:
            hr1_cold.append(hit1)
            ndcg3_cold.append(ndcg3)
        else:
            hr1_act.append(hit1)
            ndcg3_act.append(ndcg3)

    return {
        "hr1_all": float(np.mean(hr1_all)) if hr1_all else 0.0,
        "ndcg3_all": float(np.mean(ndcg3_all)) if ndcg3_all else 0.0,
        "hr1_act": float(np.mean(hr1_act)) if hr1_act else 0.0,
        "ndcg3_act": float(np.mean(ndcg3_act)) if ndcg3_act else 0.0,
        "hr1_cold": float(np.mean(hr1_cold)) if hr1_cold else 0.0,
        "ndcg3_cold": float(np.mean(ndcg3_cold)) if ndcg3_cold else 0.0,
    }


def run_study(max_sessions: int = 3500) -> None:
    print(f"[study] Loading cached {MODEL_NAME} model...")
    engine = RecommenderEngine(until=None)
    if engine.lgb_model is None:
        raise RuntimeError("LightGBM model is not trained/loaded in cache.")

    print(f"[study] Fetching recent real purchase sessions for validation...")
    raw_sessions = query(TRAIN_SESSIONS_SQL.format(cut=""), timeout=120)
    purchase_sessions = [s for s in raw_sessions if (s.get("purchased") or "") in engine.prod2idx]
    val_sessions = purchase_sessions[:max_sessions]
    print(f"[study] Evaluating on {len(val_sessions)} verified purchase sessions...")

    X_blocks, y_blocks, groups, targets, is_cold_mask = [], [], [], [], []
    blocked = engine.inactive

    for s in val_sessions:
        target = s["purchased"]
        rv = s.get("recent_views") or []
        ctx = s.get("context") if isinstance(s.get("context"), dict) else {}

        cands, cov_m, txt_m, t_cat, l_sub, avg_p, v_set = engine._get_candidates(rv, blocked, ctx)
        if target not in cands:
            cands = [target] + cands[:49]

        tr_s = engine._score_transformer_batch([rv], [cands])[0]
        X_i = engine._build_lgb_features(
            cands, tr_s, cov_m, txt_m, t_cat, l_sub, avg_p, v_set, len(rv), ctx
        )
        t_idx = cands.index(target)
        y_i = np.zeros(len(cands), dtype=np.int32)
        y_i[t_idx] = 1

        X_blocks.append(X_i)
        y_blocks.append(y_i)
        groups.append(len(cands))
        targets.append(t_idx)
        is_cold_mask.append(len(rv) == 0)

    X_val = np.vstack(X_blocks)
    y_val = np.concatenate(y_blocks)
    n_cold = sum(is_cold_mask)
    n_act = len(is_cold_mask) - n_cold

    # -----------------------------------------------------------------------
    # 1. BASELINE RANKING METRICS
    # -----------------------------------------------------------------------
    base_probs = engine.lgb_model.booster_.predict(X_val)
    base_metrics = evaluate_ranking(base_probs, groups, targets, is_cold_mask)

    report_lines = [
        "=" * 78,
        f"FEATURE IMPORTANCE & RANKING DIAGNOSTIC STUDY ({MODEL_NAME})",
        f"Validation Sample: {len(val_sessions)} purchases ({n_act} Active | {n_cold} Cold-Start)",
        "=" * 78,
        f"Baseline HitRate@1 (All)       : {base_metrics['hr1_all']:.4%}",
        f"Baseline NDCG@3    (All)       : {base_metrics['ndcg3_all']:.4%}",
        f"Baseline HitRate@1 (Active)    : {base_metrics['hr1_act']:.4%}",
        f"Baseline HitRate@1 (Cold-Start): {base_metrics['hr1_cold']:.4%}",
        "",
        "-" * 78,
        f"{'Feature':18s} | {'Univar AUC':>10s} | {'Drop HR@1':>10s} | {'Drop Act':>10s} | {'Drop Cold':>10s} | {'LGB Gain':>9s}",
        "-" * 78,
    ]

    # -----------------------------------------------------------------------
    # 2. UNIVARIATE AUC + PERMUTATION IMPORTANCE (ALL / ACTIVE / COLD-START)
    # -----------------------------------------------------------------------
    gains = engine.lgb_model.booster_.feature_importance(importance_type="gain")
    gain_pct = 100.0 * gains / max(np.sum(gains), 1e-9)

    rng = np.random.default_rng(42)
    rows_summary = []

    for j, name in enumerate(FEATURE_NAMES):
        col = X_val[:, j]
        try:
            auc = roc_auc_score(y_val, col) if len(np.unique(col)) > 1 else 0.5
        except Exception:
            auc = 0.5

        # Average over 3 permutations to eliminate random noise
        d_all, d_act, d_cold = [], [], []
        for _ in range(3):
            X_perm = X_val.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            p_perm = engine.lgb_model.booster_.predict(X_perm)
            m_perm = evaluate_ranking(p_perm, groups, targets, is_cold_mask)
            d_all.append(base_metrics["hr1_all"] - m_perm["hr1_all"])
            d_act.append(base_metrics["hr1_act"] - m_perm["hr1_act"])
            d_cold.append(base_metrics["hr1_cold"] - m_perm["hr1_cold"])

        rows_summary.append({
            "name": name,
            "auc": auc,
            "drop_all": float(np.mean(d_all)),
            "drop_act": float(np.mean(d_act)),
            "drop_cold": float(np.mean(d_cold)),
            "gain_pct": float(gain_pct[j]),
        })

    rows_summary.sort(key=lambda r: r["drop_all"], reverse=True)
    for r in rows_summary:
        report_lines.append(
            f"{r['name']:18s} | {r['auc']:10.4f} | {r['drop_all']:+9.4%} | {r['drop_act']:+9.4%} | {r['drop_cold']:+9.4%} | {r['gain_pct']:8.1f}%"
        )

    # -----------------------------------------------------------------------
    # 3. OPTIONAL SHAP DIRECTIONAL ANALYSIS & PLOTS
    # -----------------------------------------------------------------------
    try:
        import shap

        sub_idx = rng.choice(len(X_val), size=min(2500, len(X_val)), replace=False)
        X_sub = X_val[sub_idx]
        explainer = shap.TreeExplainer(engine.lgb_model)
        shap_vals = explainer.shap_values(X_sub)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        report_lines.extend([
            "",
            "-" * 78,
            "SHAP DIRECTIONAL CORRELATION (Positive = High feature value boosts purchase)",
            "-" * 78,
        ])
        for j, name in enumerate(FEATURE_NAMES):
            col_sub = X_sub[:, j]
            sv_col = shap_vals[:, j]
            mean_abs = float(np.mean(np.abs(sv_col)))
            corr = float(np.corrcoef(col_sub, sv_col)[0, 1]) if np.std(col_sub) > 1e-9 else 0.0
            direction = "BOOSTS purchase (+)" if corr > 0.1 else ("LOWERS purchase (-)" if corr < -0.1 else "Non-linear / Neutral")
            report_lines.append(f"{name:18s} | Mean |SHAP|: {mean_abs:.4f} | Corr: {corr:+.3f} -> {direction}")

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sub, feature_names=FEATURE_NAMES, show=False)
            plt.tight_layout()
            plt.savefig("shap_summary.png", dpi=150)
            plt.close()
            report_lines.append("[plot] Saved SHAP summary plot to shap_summary.png")
        except Exception as plot_err:
            report_lines.append(f"[warn] Could not save SHAP plot: {plot_err}")

    except ImportError:
        report_lines.append("[info] Install 'shap' (`pip install shap matplotlib`) to add directional SHAP plots.")

    # Save Permutation Importance Bar Chart
    try:
        import matplotlib.pyplot as plt

        names_plot = [r["name"] for r in reversed(rows_summary)]
        drops_plot = [r["drop_all"] * 100.0 for r in reversed(rows_summary)]
        plt.figure(figsize=(9, 5.5))
        colors = ["#2b7bba" if v > 0 else "#c44e52" for v in drops_plot]
        plt.barh(names_plot, drops_plot, color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("HitRate@1 Drop when Permuted (percentage points)")
        plt.title(f"{MODEL_NAME} — Feature Permutation Importance on Slot-1 Purchase Accuracy")
        plt.tight_layout()
        plt.savefig("perm_importance.png", dpi=150)
        plt.close()
        report_lines.append("[plot] Saved Permutation Importance chart to perm_importance.png")
    except Exception:
        pass

    report_text = "".join(report_lines)
    print("" + report_text)
    Path("feature_study_report.txt").write_text(report_text, encoding="utf-8")
    print("[done] Full report saved to feature_study_report.txt")


if __name__ == "__main__":
    run_study(max_sessions=3500)
