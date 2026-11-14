from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 15
EMBED_DIM = 128
NUM_NEGATIVES = 15
torch.manual_seed(42)
np.random.seed(42)

def query(sql: str) -> list[dict[str, Any]]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=90)
    r.raise_for_status()
    return r.json()["data"]


def load_data():
    print("1. Descargando catalogo y sesiones con clics/compras...")
    cat_rows = query("""
        SELECT product_id::String AS pid, category::String AS cat, subcategory::String AS sub
        FROM shop.catalog
    """)
    cat_map = {r["pid"]: (r["cat"], r["sub"]) for r in cat_rows}

    rows = query("""
        SELECT context, recent_views, shown, clicked, purchased
        FROM shop.history_sessions
        WHERE clicked IS NOT NULL AND clicked != ''
        ORDER BY rand()
        LIMIT 150000
    """)
    return cat_map, rows


def bpr_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking (BPR) Loss.
    Columna 0 = producto positivo (clic/compra), columnas 1..N = negativos.
    """
    pos_scores = logits[:, 0:1]
    neg_scores = logits[:, 1:]
    return -F.logsigmoid(pos_scores - neg_scores).mean()


class ContrastiveSessionDataset(Dataset):
    def __init__(self, samples, num_items: int):
        self.samples = samples
        self.num_items = num_items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, page_id, reg_id, dev_id, pos_id, hard_negs = self.samples[idx]
        negs = list(hard_negs[:NUM_NEGATIVES])
        while len(negs) < NUM_NEGATIVES:
            rnd = np.random.randint(1, self.num_items + 1)
            if rnd != pos_id:
                negs.append(rnd)

        candidates = [pos_id] + negs  # Indice 0 siempre es el positivo
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(page_id, dtype=torch.long),
            torch.tensor(reg_id, dtype=torch.long),
            torch.tensor(dev_id, dtype=torch.long),
            torch.tensor(candidates, dtype=torch.long),
        )


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

    def forward_candidates(self, seq, page, reg, dev, candidates):
        pooled = self.encode_session(seq, page, reg, dev).unsqueeze(1)
        cand_vecs = self.get_item_representations(candidates)
        cand_bias = self.output_bias[candidates]
        return (pooled * cand_vecs).sum(dim=-1) + cand_bias

    def forward(self, seq: torch.Tensor, page: torch.Tensor, reg: torch.Tensor, dev: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_session(seq, page, reg, dev)
        all_ids = torch.arange(self.item_emb.num_embeddings, device=seq.device)
        all_vecs = self.get_item_representations(all_ids)
        return torch.matmul(pooled, all_vecs.t()) + self.output_bias


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    use_amp = DEVICE.type == "cuda"
    print(f"Entrenando Transformer con BPR Loss en: {DEVICE} (bfloat16={use_amp})")
    cat_map, rows = load_data()

    item_vocab: dict[str, int] = {}
    cat_vocab: dict[str, int] = {"unk": 0}
    page_vocab: dict[str, int] = {"unk": 0}
    reg_vocab: dict[str, int] = {"unk": 0}
    dev_vocab: dict[str, int] = {"unk": 0}

    def get_id(vocab: dict[str, int], key: str) -> int:
        if not key:
            return 0
        if key not in vocab:
            vocab[key] = len(vocab) + 1
        return vocab[key]

    for pid, (cat, sub) in cat_map.items():
        get_id(item_vocab, pid)
        get_id(cat_vocab, f"{cat}_{sub}")

    item_to_cat = torch.zeros(len(item_vocab) + 1, dtype=torch.long)
    for pid, idx in item_vocab.items():
        cat, sub = cat_map.get(pid, ("", ""))
        item_to_cat[idx] = cat_vocab.get(f"{cat}_{sub}", 0)

    samples = []
    for r in rows:
        ctx = r.get("context") or {}
        p_id = get_id(page_vocab, str(ctx.get("page", "")))
        r_id = get_id(reg_vocab, str(ctx.get("region", "")))
        d_id = get_id(dev_vocab, str(ctx.get("device", "")))

        rv_raw = r.get("recent_views") or []
        rv_ids = [item_vocab.get(x["product_id"] if isinstance(x, dict) else str(x), 0) for x in rv_raw]
        rv_ids = [x for x in rv_ids if x > 0][-MAX_LEN:]
        padded = [0] * (MAX_LEN - len(rv_ids)) + rv_ids

        shown = r.get("shown") or []
        clicked = r.get("clicked")
        purchased = r.get("purchased")

        target_pid = purchased if (purchased and isinstance(purchased, str)) else clicked
        if not target_pid or target_pid not in item_vocab:
            continue

        t_id = item_vocab[target_pid]
        hard_negs = [item_vocab[p] for p in shown if p in item_vocab and p != target_pid and p != clicked]

        repeats = 2 if (purchased and isinstance(purchased, str)) else 1
        for _ in range(repeats):
            samples.append((padded, p_id, r_id, d_id, t_id, hard_negs))

    dataset = ContrastiveSessionDataset(samples, len(item_vocab))
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=torch.cuda.is_available())

    model = ContextualTransformer(
        len(item_vocab), len(page_vocab), len(reg_vocab), len(dev_vocab),
        item_to_cat.to(DEVICE), len(cat_vocab), EMBED_DIM
    ).to(DEVICE)

    epochs = 15
    optimizer = torch.optim.RAdam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        epochs=epochs,
        steps_per_epoch=len(loader),
        pct_start=0.1,
    )

    print(f"2. Entrenando con {len(samples)} muestras...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for b_seq, b_p, b_r, b_d, b_cands in loader:
            b_seq = b_seq.to(DEVICE, non_blocking=True)
            b_p = b_p.to(DEVICE, non_blocking=True)
            b_r = b_r.to(DEVICE, non_blocking=True)
            b_d = b_d.to(DEVICE, non_blocking=True)
            b_cands = b_cands.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model.forward_candidates(b_seq, b_p, b_r, b_d, b_cands)
                loss = bpr_loss(logits)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} - BPR Loss: {total_loss / len(loader):.4f}")

    meta = {
        "item_vocab": item_vocab,
        "page_vocab": page_vocab,
        "reg_vocab": reg_vocab,
        "dev_vocab": dev_vocab,
        "cat_vocab": cat_vocab,
        "item_to_cat": item_to_cat.tolist(),
        "embed_dim": EMBED_DIM,
    }
    Path("meta.json").write_text(json.dumps(meta))
    torch.save(model.state_dict(), "sasrec_ctx.pt")
    print("Modelo guardado en 'sasrec_ctx.pt' y 'meta.json'.")


if __name__ == "__main__":
    main()