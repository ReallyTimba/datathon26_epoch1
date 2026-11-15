"""Monitor en vivo del marcador (monitor.py) conectado a {TEAM}.feedback + {TEAM}.submissions.
Inspecciona automáticamente el esquema de {TEAM}.feedback al arrancar para evitar errores de columnas
Dynamic en ClickHouse y actualiza cada 15 segundos: Requests, Conversión %, Cold-Start % y €/req.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

for env_path in (Path(".env"), Path("../.env")):
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "team_15")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"


def query(sql: str, timeout: int = 20) -> list[dict[str, Any]]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json().get("data", [])


def inspect_feedback_schema() -> set[str]:
    try:
        cols = query(f"DESCRIBE TABLE {TEAM}.feedback", timeout=15)
        names = {str(c.get("name") or "") for c in cols if c.get("name")}
        print(f"[schema] Columnas en {TEAM}.feedback: {sorted(names)}")
        return names
    except Exception as e:
        print(f"[warn] No se pudo inspeccionar {TEAM}.feedback: {e}")
        return set()


def build_scoreboard_sql(fb_cols: set[str]) -> str:
    # Detección automática de columnas según el esquema real de {TEAM}.feedback
    has_purchased = "purchased" in fb_cols
    has_revenue = "revenue_eur" in fb_cols
    has_cold = "is_cold_start" in fb_cols or "is_new_user" in fb_cols
    cold_col = "is_cold_start" if "is_cold_start" in fb_cols else ("is_new_user" if "is_new_user" in fb_cols else "")

    if has_purchased:
        conv_expr = "f.purchased::String != ''"
    elif "event_type" in fb_cols:
        conv_expr = "f.event_type::String = 'purchase'"
    elif "converted" in fb_cols:
        conv_expr = "toInt32OrZero(toString(f.converted)) = 1"
    else:
        conv_expr = "toFloat64OrZero(toString(f.revenue_eur)) > 0"

    if has_cold:
        cold_cond = f"toInt32OrZero(toString(f.{cold_col})) = 1"
        req_join = ""
    else:
        cold_cond = "length(r.recent_views) = 0"
        req_join = "LEFT JOIN shop.requests AS r ON s.request_id::String = r.request_id::String"

    if has_revenue:
        rev_expr = "toFloat64OrZero(toString(f.revenue_eur))"
        cat_join = ""
    elif "price_eur" in fb_cols:
        rev_expr = f"if({conv_expr}, toFloat64OrZero(toString(f.price_eur)), 0.0)"
        cat_join = ""
    else:
        rev_expr = f"if({conv_expr}, c.price_eur, 0.0)"
        cat_join = """
        LEFT JOIN (
            SELECT product_id::String AS pid, argMax(price_eur::Float64, ts) AS price_eur
            FROM (
                SELECT product_id, price_eur, toDateTime64('1970-01-01 00:00:00', 3) AS ts FROM shop.catalog
                UNION ALL
                SELECT product_id, price_eur, ts::DateTime64(3) FROM shop.catalog_updates
            )
            GROUP BY pid
        ) AS c ON f.purchased::String = c.pid
        """

    return f"""
    SELECT
        s.model_version::String AS model,
        count() AS reqs,
        round(100.0 * countIf({conv_expr}) / greatest(count(), 1), 4) AS conv_pct,
        round(100.0 * countIf({cold_cond} AND ({conv_expr}))
              / greatest(countIf({cold_cond}), 1), 4) AS cold_pct,
        round(sum({rev_expr}) / greatest(count(), 1), 3) AS eur_per_req
    FROM (
        SELECT request_id::String AS request_id, argMax(model_version::String, _row_num) AS model_version
        FROM (
            SELECT request_id, model_version, rowNumberInAllBlocks() AS _row_num
            FROM {TEAM}.submissions
        )
        GROUP BY request_id
    ) AS s
    INNER JOIN {TEAM}.feedback AS f
        ON s.request_id = f.request_id::String
    {req_join}
    {cat_join}
    GROUP BY model
    ORDER BY reqs DESC
    """


def main() -> None:
    print(f"[init] Conectando a {API} (DB: {TEAM})...")
    fb_cols = inspect_feedback_schema()
    sql = build_scoreboard_sql(fb_cols)

    while True:
        try:
            rows = query(sql, timeout=20)
            os.system("cls" if os.name == "nt" else "clear")
            print(f"=== MARCADOR EN VIVO ({TEAM}.feedback) | {time.strftime('%H:%M:%S')} ===")
            if not rows:
                print(f"Aún no hay filas consolidadas en {TEAM}.feedback. Esperando feedback del simulador...")
            else:
                print(f"{'Modelo':12s} | {'Reqs':>7s} | {'Conversión':>10s} | {'Coldstart':>10s} | {'€/req':>8s}")
                print("-" * 60)
                for r in rows:
                    print(
                        f"{str(r['model']):12s} | {int(r['reqs']):7d} | "
                        f"{float(r['conv_pct']):9.4f}% | "
                        f"{float(r['cold_pct']):9.4f}% | "
                        f"€{float(r['eur_per_req']):7.3f}"
                    )
        except Exception as e:
            print(f"[error] {e}")
        time.sleep(15)


if __name__ == "__main__":
    main()
