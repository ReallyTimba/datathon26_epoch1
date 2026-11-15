from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import xgboost as xgb

from train_transformer import ContextualTransformer, DEVICE, EMBED_DIM, MAX_LEN

os.chdir(Path(__file__).resolve().parent)  # model files live next to this script (models/)
for line in Path("../.env").read_text().splitlines() if Path("../.env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"


def query(sql: str) -> list[dict[str, Any]]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=90)
    r.raise_for_status()
    return r.json()["data"]


def load_all():
    print("1. Cargando catalogo, popularidad y afinidad region-categoria...")
    cat_rows = query("""
        SELECT product_id,
               argMax(category, v)    AS category,
               argMax(subcategory, v) AS subcategory,
               argMax(price_eur, v)   AS price_eur
        FROM (
            SELECT product_id::String AS product_id, category::String AS category,
                   subcategory::String AS subcategory, price_eur::Float64 AS price_eur, '' AS v
            FROM shop.catalog
            UNION ALL
            SELECT product_id::String, category::String, subcategory::String,
                   price_eur::Float64, toString(ts)
            FROM shop.catalog_updates
        )
        GROUP BY product_id
    """)
    catalog = {r["product_id"]: r for r in cat_rows}

    pop_rows = query("""
        SELECT product_id::String AS product_id,
               countIf(event_type = 'view') AS views,
               countIf(event_type = 'purchase') AS purchases
        FROM shop.history_events
        GROUP BY product_id
    """)
    pop = {r["product_id"]: (int(r["views"]), int(r["purchases"])) for r in pop_rows}

    reg_rows = query("""
        SELECT e.region::String AS region, c.category::String AS category, count() AS cnt
        FROM shop.history_events AS e
        INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
        WHERE e.event_type = 'purchase'
        GROUP BY region, category
    """)
    reg_tot: dict[str, int] = {}
    reg_cat_cnt: dict[tuple[str, str], int] = {}
    for r in reg_rows:
        reg, cat, c = r["region"], r["category"], int(r["cnt"])
        reg_tot[reg] = reg_tot.get(reg, 0) + c
        reg_cat_cnt[(reg, cat)] = c
    reg_cat_share = {k: v / max(reg_tot[k[0]], 1) for k, v in reg_cat_cnt.items()}

    user_rows = query("""
        SELECT user_id::String AS uid,
               groupUniqArray(20)(product_id::String) AS prods
        FROM shop.history_events
        GROUP BY uid
    """)
    user_hist = {r["uid"]: set(r["prods"]) for r in user_rows}

    print("2. Descargando sesiones historicas para XGBRanker...")
    sessions = query("""
        SELECT user_id, context, recent_views, shown, clicked, purchased, propensity
        FROM shop.history_sessions
        WHERE clicked IS NOT NULL AND clicked != ''
        ORDER BY rand()
        LIMIT 80000
    """)
    return catalog, pop, reg_cat_share, user_hist, sessions


def main():
    meta = json.loads(Path("meta.json").read_text())
    item_vocab = meta["item_vocab"]
    page_vocab = meta["page_vocab"]
    reg_vocab = meta["reg_vocab"]
    dev_vocab = meta["dev_vocab"]
    cat_vocab = meta["cat_vocab"]
    item_to_cat = torch.tensor(meta["item_to_cat"], dtype=torch.long, device=DEVICE)
    embed_dim = int(meta.get("embed_dim", EMBED_DIM))

    trans_model = ContextualTransformer(
        len(item_vocab),
        len(page_vocab),
        len(reg_vocab),
        len(dev_vocab),
        item_to_cat,
        len(cat_vocab),
        embed_dim,
    ).to(DEVICE)
    trans_model.load_state_dict(torch.load("sasrec_ctx.pt", map_location=DEVICE))
    trans_model.eval()

    catalog, pop, reg_cat_share, user_hist, sessions = load_all()

    use_amp = DEVICE.type == "cuda"
    print(f"3. Calculando predicciones del Transformer BPR en {DEVICE} (bfloat16={use_amp})...")
    all_probs = []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, len(sessions), 1024):
            batch = sessions[i : i + 1024]
            seqs, pages, regs, devs = [], [], [], []
            for r in batch:
                ctx = r.get("context") or {}
                pages.append(page_vocab.get(str(ctx.get("page", "")), 0))
                regs.append(reg_vocab.get(str(ctx.get("region", "")), 0))
                devs.append(dev_vocab.get(str(ctx.get("device", "")), 0))

                rv = r.get("recent_views") or []
                ids = [item_vocab.get(x["product_id"] if isinstance(x, dict) else str(x), 0) for x in rv]
                ids = [x for x in ids if x > 0][-MAX_LEN:]
                seqs.append([0] * (MAX_LEN - len(ids)) + ids)

            t_seq = torch.tensor(seqs, dtype=torch.long, device=DEVICE)
            t_p = torch.tensor(pages, dtype=torch.long, device=DEVICE)
            t_r = torch.tensor(regs, dtype=torch.long, device=DEVICE)
            t_d = torch.tensor(devs, dtype=torch.long, device=DEVICE)
            logits = trans_model(t_seq, t_p, t_r, t_d)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs_matrix = np.vstack(all_probs)

    print("4. Construyendo dataset de 13 features para XGBRanker...")
    X_list, y_list, qid_list, w_list = [], [], [], []
    page_map = {"home": 1.0, "category": 2.0, "product": 3.0}

    for q_idx, r in enumerate(sessions):
        shown = r.get("shown") or []
        if len(shown) < 2:
            continue
        clicked = r.get("clicked")
        purchased = r.get("purchased")
        props = r.get("propensity") or [0.1] * len(shown)
        ctx = r.get("context") or {}
        region = str(ctx.get("region", ""))
        page_num = page_map.get(str(ctx.get("page", "")), 0.0)
        uid = str(r.get("user_id", ""))
        u_prods = user_hist.get(uid, set())

        rv_raw = r.get("recent_views") or []
        rv_ids = [x["product_id"] if isinstance(x, dict) else str(x) for x in rv_raw]
        last_p = rv_ids[-1] if rv_ids else ""
        last_info = catalog.get(last_p, {})
        viewed_prices = [float(catalog[p]["price_eur"]) for p in rv_ids if p in catalog and catalog[p]["price_eur"] > 0]
        avg_v_price = float(np.mean(viewed_prices)) if viewed_prices else 0.0
        rv_set = set(rv_ids)

        row_probs = probs_matrix[q_idx]
        has_pos = False
        q_X, q_y = [], []
        for idx, cand in enumerate(shown):
            rel = 3 if (purchased and cand == purchased) else (1 if (clicked and cand == clicked) else 0)
            if rel > 0:
                has_pos = True

            c_info = catalog.get(cand, {})
            c_cat = c_info.get("category", "")
            c_sub = c_info.get("subcategory", "")
            c_price = float(c_info.get("price_eur", 0.0))
            c_views, c_purch = pop.get(cand, (0, 0))
            c_conv = c_purch / max(c_views, 1)

            t_idx = item_vocab.get(cand, 0)
            t_prob = float(row_probs[t_idx]) if t_idx > 0 else 0.0

            same_cat = 1.0 if (last_p and c_cat == last_info.get("category")) else 0.0
            same_sub = 1.0 if (last_p and c_sub == last_info.get("subcategory")) else 0.0
            p_ratio = c_price / max(avg_v_price, 1.0) if avg_v_price > 0 else 1.0
            reg_aff = reg_cat_share.get((region, c_cat), 0.05)
            in_u_hist = 1.0 if cand in u_prods else 0.0
            in_rv = 1.0 if cand in rv_set else 0.0

            q_X.append([
                t_prob,
                np.log1p(t_prob * 10000),
                c_price,
                np.log1p(c_views),
                np.log1p(c_purch),
                c_conv,
                same_cat,
                same_sub,
                p_ratio,
                reg_aff,
                page_num,
                in_u_hist,
                in_rv,
            ])
            q_y.append(rel)

        if not has_pos:
            continue

        click_slot = shown.index(clicked) if clicked in shown else 0
        p_val = max(float(props[click_slot]), 0.02) if click_slot < len(props) else 0.1
        q_weight = min(1.0 / p_val, 8.0)

        X_list.extend(q_X)
        y_list.extend(q_y)
        qid_list.extend([q_idx] * len(q_X))
        w_list.append(q_weight)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    qid = np.array(qid_list, dtype=np.int32)
    weights = np.array(w_list, dtype=np.float32)

    print(f"5. Entrenando XGBRanker en GPU ({len(X)} filas, {len(weights)} consultas)...")
    ranker = xgb.XGBRanker(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="rank:ndcg",
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
    ranker.fit(X, y, qid=qid, sample_weight=weights, verbose=50)
    ranker.get_booster().save_model("xgb_ranker.json")
    print("Guardado 'xgb_ranker.json' con exito.")


if __name__ == "__main__":
    main()