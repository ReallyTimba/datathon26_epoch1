from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
import xgboost as xgb

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

Record = dict[str, Any]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 15
EMBED_DIM = 128

# --- SQL QUERIES ---
CATALOG_SQL = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, price_eur::Float64, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id"""

SALES_SQL = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'view') AS views,
       countIf(event_type = 'purchase') AS purchases
FROM shop.history_events
GROUP BY product_id"""

REG_CAT_SQL = """
SELECT e.region::String AS region, c.category::String AS category, count() AS cnt
FROM shop.history_events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
WHERE e.event_type = 'purchase'
GROUP BY region, category"""

USER_HIST_SQL = """
SELECT user_id::String AS uid,
       groupUniqArray(20)(product_id::String) AS prods
FROM shop.history_events
GROUP BY uid"""

LIVE_TRENDS_SQL = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'purchase') AS live_purchases
FROM shop.events
WHERE ts > now() - INTERVAL 1 HOUR
GROUP BY product_id
ORDER BY live_purchases DESC
LIMIT 200"""

NEW_REQUESTS_SQL = """
SELECT request_id, user_id, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 50"""

OUT_OF_STOCK_SQL = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""


class ContextualTransformer(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_pages: int,
        num_regs: int,
        num_devs: int,
        item_to_cat: torch.Tensor,
        num_cats: int,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(num_cats + 1, embed_dim, padding_idx=0)
        self.register_buffer("item_to_cat", item_to_cat)

        self.pos_emb = nn.Embedding(MAX_LEN + 1, embed_dim)
        self.page_emb = nn.Embedding(num_pages + 1, embed_dim)
        self.reg_emb = nn.Embedding(num_regs + 1, embed_dim)
        self.dev_emb = nn.Embedding(num_devs + 1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_bias = nn.Parameter(torch.zeros(num_items + 1))

    def get_item_representations(self, ids: torch.Tensor) -> torch.Tensor:
        return self.item_emb(ids) + self.cat_emb(self.item_to_cat[ids])

    def encode_session(self, seq: torch.Tensor, page: torch.Tensor, reg: torch.Tensor, dev: torch.Tensor) -> torch.Tensor:
        ctx_token = (self.page_emb(page) + self.reg_emb(reg) + self.dev_emb(dev)).unsqueeze(1)
        item_tokens = self.get_item_representations(seq)
        x = torch.cat([ctx_token, item_tokens], dim=1)

        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

        ctx_mask = torch.zeros((seq.size(0), 1), dtype=torch.bool, device=seq.device)
        pad_mask = torch.cat([ctx_mask, seq == 0], dim=1)

        h = self.transformer(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        return h[:, 0, :] + h[:, -1, :]

    def forward(self, seq: torch.Tensor, page: torch.Tensor, reg: torch.Tensor, dev: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_session(seq, page, reg, dev)
        all_ids = torch.arange(self.item_emb.num_embeddings, device=seq.device)
        all_vecs = self.get_item_representations(all_ids)
        return torch.matmul(pooled, all_vecs.t()) + self.output_bias


def query(sql: str) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=120)  # startup queries are heavy
    r.raise_for_status()
    return r.json()["data"]


def out_of_stock() -> set[str]:
    return {r["product_id"] for r in query(OUT_OF_STOCK_SQL)}


def pull(seen: set[str]) -> list[Record]:
    fresh = []
    for req in query(NEW_REQUESTS_SQL):
        if req["request_id"] not in seen:
            seen.add(req["request_id"])
            fresh.append(req)
    return fresh


def push(answers: list[Record]) -> None:
    http.post(f"{API}/v1/tables/submissions", params={"database": TEAM}, json=answers, timeout=10).raise_for_status()
    print(f"answered {len(answers)}")


class HybridTransformerXGB:
    def __init__(
        self,
        ranker_path: str = "xgb_ranker.json",
        trans_path: str = "sasrec_ctx.pt",
        meta_path: str = "meta.json",
    ):
        meta = json.loads(Path(meta_path).read_text())
        self.item_vocab: dict[str, int] = meta["item_vocab"]
        self.inv_item_vocab: dict[int, str] = {v: k for k, v in self.item_vocab.items()}
        self.page_vocab: dict[str, int] = meta["page_vocab"]
        self.reg_vocab: dict[str, int] = meta["reg_vocab"]
        self.dev_vocab: dict[str, int] = meta["dev_vocab"]
        self.cat_vocab: dict[str, int] = meta["cat_vocab"]
        item_to_cat = torch.tensor(meta["item_to_cat"], dtype=torch.long, device=DEVICE)
        embed_dim = int(meta.get("embed_dim", EMBED_DIM))

        # 1. Cargar Transformer Contextual con BPR + Categorias en RTX 4070
        self.trans_model = ContextualTransformer(
            len(self.item_vocab),
            len(self.page_vocab),
            len(self.reg_vocab),
            len(self.dev_vocab),
            item_to_cat,
            len(self.cat_vocab),
            embed_dim,
        ).to(DEVICE)
        self.trans_model.load_state_dict(torch.load(trans_path, map_location=DEVICE))
        self.trans_model.eval()

        # 2. Cargar XGBRanker en RTX 4070
        self.booster = xgb.Booster()
        self.booster.load_model(ranker_path)
        self.booster.set_param({"device": "cuda"})

        # Warm-up de ambos modelos en GPU (con Mixed Precision bfloat16)
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")
        ):
            z_seq = torch.zeros((4, MAX_LEN), dtype=torch.long, device=DEVICE)
            z_ctx = torch.zeros((4,), dtype=torch.long, device=DEVICE)
            _ = self.trans_model(z_seq, z_ctx, z_ctx, z_ctx)
        _ = self.booster.inplace_predict(np.zeros((10, 13), dtype=np.float32))

        self.catalog: dict[str, Record] = {}
        self.active_products: set[str] = set()
        self.pop: dict[str, tuple[int, int]] = {}
        self.reg_cat_share: dict[tuple[str, str], float] = {}
        self.user_hist: dict[str, set[str]] = {}
        self.best: list[str] = []
        self.live_hot: list[str] = []
        self.page_map = {"home": 1.0, "category": 2.0, "product": 3.0}

    @staticmethod
    def _extract_rv_ids(recent_views: list[Any] | None) -> list[str]:
        return [
            item["product_id"] if isinstance(item, dict) else str(item)
            for item in (recent_views or [])
            if (isinstance(item, dict) and "product_id" in item) or isinstance(item, str)
        ]

    def init_static_data(self) -> None:
        print("Cargando catalogo, popularidad, regiones e historial de usuarios...")
        for r in query(CATALOG_SQL):
            pid = r["product_id"]
            self.catalog[pid] = r
            if r.get("is_active", True):
                self.active_products.add(pid)

        sales_list = []
        for r in query(SALES_SQL):
            pid = r["product_id"]
            v, p = int(r["views"]), int(r["purchases"])
            self.pop[pid] = (v, p)
            sales_list.append((p, pid))

        sales_list.sort(reverse=True)
        self.best = [pid for _, pid in sales_list]

        reg_tot: dict[str, int] = {}
        reg_cat_cnt: dict[tuple[str, str], int] = {}
        for r in query(REG_CAT_SQL):
            reg, cat, c = r["region"], r["category"], int(r["cnt"])
            reg_tot[reg] = reg_tot.get(reg, 0) + c
            reg_cat_cnt[(reg, cat)] = c
        self.reg_cat_share = {k: v / max(reg_tot[k[0]], 1) for k, v in reg_cat_cnt.items()}

        for r in query(USER_HIST_SQL):
            self.user_hist[r["uid"]] = set(r["prods"])

        print(f"Listo: {len(self.active_products)} productos activos cargados en {DEVICE}.")

    def refresh_live_trends(self) -> None:
        self.live_hot = [r["product_id"] for r in query(LIVE_TRENDS_SQL)]

    def _run_transformer_batch(self, requests_list: list[Record]) -> np.ndarray:
        """Ejecuta el Transformer Contextual en Mixed Precision bfloat16 sobre RTX 4070."""
        seqs, pages, regs, devs = [], [], [], []
        for req in requests_list:
            ctx = req.get("context") or {}
            pages.append(self.page_vocab.get(str(ctx.get("page", "")), 0))
            regs.append(self.reg_vocab.get(str(ctx.get("region", "")), 0))
            devs.append(self.dev_vocab.get(str(ctx.get("device", "")), 0))

            rv_ids = self._extract_rv_ids(req.get("recent_views"))
            idx_seq = [self.item_vocab.get(p, 0) for p in rv_ids]
            idx_seq = [x for x in idx_seq if x > 0][-MAX_LEN:]
            seqs.append([0] * (MAX_LEN - len(idx_seq)) + idx_seq)

        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")
        ):
            t_seq = torch.tensor(seqs, dtype=torch.long, device=DEVICE)
            t_p = torch.tensor(pages, dtype=torch.long, device=DEVICE)
            t_r = torch.tensor(regs, dtype=torch.long, device=DEVICE)
            t_d = torch.tensor(devs, dtype=torch.long, device=DEVICE)
            logits = self.trans_model(t_seq, t_p, t_r, t_d)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs

    def get_candidates(
        self,
        row_probs: np.ndarray,
        u_prods: set[str],
        unavailable: set[str],
        limit: int = 45,
    ) -> list[str]:
        """Combina Top predicciones del Transformer + historial usuario + tendencias + best sellers."""
        cands: list[str] = []
        seen_cands: set[str] = set()

        def try_add(pid: str) -> bool:
            if pid not in seen_cands and pid in self.active_products and pid not in unavailable:
                seen_cands.add(pid)
                cands.append(pid)
                return len(cands) >= limit
            return False

        # 1. Top 30 productos sugeridos por el Transformer Contextual
        top_t_idx = np.argsort(-row_probs)[:35]
        for idx in top_t_idx:
            pid = self.inv_item_vocab.get(int(idx))
            if pid and try_add(pid):
                return cands

        # 2. Productos del historial previo del usuario
        for up in u_prods:
            if try_add(up):
                return cands

        # 3. Tendencias en vivo de la ultima hora
        for hot_p in self.live_hot[:15]:
            if try_add(hot_p):
                return cands

        # 4. Best sellers globales (garantiza el suelo fuerte del baseline)
        for b_p in self.best:
            if try_add(b_p):
                return cands

        return cands

    def build_features_for_request(
        self,
        req: Record,
        cands: list[str],
        row_probs: np.ndarray,
        u_prods: set[str],
    ) -> np.ndarray:
        """Construye las 13 features exactas para XGBRanker."""
        n = len(cands)
        feats = np.zeros((n, 13), dtype=np.float32)

        ctx = req.get("context") or {}
        region = str(ctx.get("region", ""))
        page_num = self.page_map.get(str(ctx.get("page", "")), 0.0)

        rv_ids = self._extract_rv_ids(req.get("recent_views"))
        has_rv = len(rv_ids) > 0
        last_p = rv_ids[-1] if has_rv else ""
        last_info = self.catalog.get(last_p, {}) if has_rv else {}
        last_cat = last_info.get("category", "")
        last_sub = last_info.get("subcategory", "")

        viewed_prices = [
            float(self.catalog[p]["price_eur"])
            for p in rv_ids
            if p in self.catalog and self.catalog[p]["price_eur"] > 0
        ]
        avg_v_price = float(np.mean(viewed_prices)) if viewed_prices else 0.0
        rv_set = set(rv_ids)

        for i, cand in enumerate(cands):
            c_info = self.catalog.get(cand, {})
            c_cat = c_info.get("category", "")
            c_sub = c_info.get("subcategory", "")
            c_price = float(c_info.get("price_eur", 0.0))
            c_views, c_purch = self.pop.get(cand, (0, 0))
            c_conv = c_purch / max(c_views, 1)

            t_idx = self.item_vocab.get(cand, 0)
            t_prob = float(row_probs[t_idx]) if t_idx > 0 else 0.0

            same_cat = 1.0 if (has_rv and last_cat and c_cat == last_cat) else 0.0
            same_sub = 1.0 if (has_rv and last_sub and c_sub == last_sub) else 0.0
            p_ratio = c_price / max(avg_v_price, 1.0) if (has_rv and avg_v_price > 0) else 1.0
            reg_aff = self.reg_cat_share.get((region, c_cat), 0.05)
            in_u_hist = 1.0 if cand in u_prods else 0.0
            in_rv = 1.0 if cand in rv_set else 0.0

            feats[i] = [
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
            ]
        return feats

    def predict_batch(self, requests_list: list[Record], unavailable: set[str]) -> list[Record]:
        if not requests_list:
            return []

        probs_matrix = self._run_transformer_batch(requests_list)

        all_features: list[np.ndarray] = []
        request_slices: list[tuple[str, list[str], int, int]] = []
        cursor = 0

        for idx, req in enumerate(requests_list):
            uid = str(req.get("user_id", ""))
            u_prods = self.user_hist.get(uid, set())
            row_probs = probs_matrix[idx]

            cands = self.get_candidates(row_probs, u_prods, unavailable, limit=45)
            feats = self.build_features_for_request(req, cands, row_probs, u_prods)
            all_features.append(feats)
            request_slices.append((req["request_id"], cands, cursor, cursor + len(cands)))
            cursor += len(cands)

        X_batch = np.ascontiguousarray(np.vstack(all_features), dtype=np.float32)
        scores = self.booster.inplace_predict(X_batch)

        submissions: list[Record] = []
        for req_id, cands, start, end in request_slices:
            req_scores = scores[start:end]
            top_idx = np.argsort(-req_scores)
            top10 = [cands[i] for i in top_idx if cands[i] not in unavailable][:10]

            if len(top10) < 10:
                chosen = set(top10)
                for p in self.best:
                    if p in self.active_products and p not in unavailable and p not in chosen:
                        top10.append(p)
                        if len(top10) == 10:
                            break

            submissions.append({
                "request_id": req_id,
                "items": top10,
                "model_version": "hybrid_bpr_xgb_v2",
            })
        return submissions


# --- Funciones de compatibilidad para offline_eval.ipynb ---
def best_sellers(until: str | None = None) -> list[str]:
    sales_cut = f" AND ts < '{until}'" if until else ""
    sql = f"""
    SELECT product_id::String AS product_id, count() AS n
    FROM shop.history_events
    WHERE event_type = 'purchase'{sales_cut}
    GROUP BY product_id"""
    sales = sorted(((int(r["n"]), r["product_id"]) for r in query(sql)), reverse=True)
    return [product for _, product in sales]


def recommend(recent_views: list[Record], best: list[str], unavailable: set[str] = frozenset()) -> list[str]:
    return [product for product in best if product not in unavailable][:10]


def main() -> None:
    rec = HybridTransformerXGB("xgb_ranker.json", "sasrec_ctx.pt", "meta.json")
    rec.init_static_data()

    seen: set[str] = set()
    unavailable: set[str] = set()
    stock_checked = 0.0
    trends_checked = 0.0

    while True:
        try:
            now = time.time()
            if now - stock_checked >= 1.0:
                stock_checked = now
                try:
                    unavailable = out_of_stock()
                except Exception as e:
                    print(f"stock check failed: {e}")

            if now - trends_checked >= 15.0:
                trends_checked = now
                try:
                    rec.refresh_live_trends()
                except Exception as e:
                    print(f"trends check failed: {e}")

            fresh_reqs = pull(seen)
            if fresh_reqs:
                answers = rec.predict_batch(fresh_reqs, unavailable)
                if answers:
                    push(answers)
        except Exception as e:
            print(f"error: {e}")
            time.sleep(2)
        time.sleep(0.3)


if __name__ == "__main__":
    main()