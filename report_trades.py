#!/usr/bin/env python3
"""
One-shot report for the Trend strategy.

- Reads SQLite DB (default: ./market_data.db)
- Filters trades for strategy_name = 'trend_4h' (configurable)
- Date filter from --start (default: 2025-08-31 UTC) to optional --end (exclusive)
- Groups by position_id (master + partials), computes:
    gross_pnl, fees, net_pnl, spend, ROI%, hold-hours, partial count
- Prints a console summary and writes:
    - CSV: trend_report_<strategy>_<stamp>.csv
    - TXT: trend_report_<strategy>_<stamp>.txt
No external deps beyond Python stdlib.
"""

import argparse
import csv
import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_DB = "market_data.db"
DEFAULT_STRATEGY = "trend_4h"
DEFAULT_START = "2025-08-31"   # UTC, inclusive

def epoch_ms(date_str_yyyy_mm_dd: str) -> int:
    dt = datetime.strptime(date_str_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def safe_div(a, b):
    try:
        if b:
            return a / b
    except Exception:
        pass
    return 0.0

def read_distinct_strategies(cur):
    try:
        cur.execute("SELECT DISTINCT strategy_name FROM trades ORDER BY 1;")
        return [r[0] for r in cur.fetchall()]
    except Exception:
        return []

def ensure_columns(cur, table, needed_cols):
    cur.execute(f"PRAGMA table_info({table});")
    cols = {row[1] for row in cur.fetchall()}
    missing = [c for c in needed_cols if c not in cols]
    if missing:
        raise RuntimeError(
            f"Table '{table}' missing columns: {missing}\n"
            f"Found columns: {sorted(cols)}"
        )

def load_trades(conn, strategy, start_ms, end_ms=None):
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Required columns based on your screenshot
    need = [
        "id","timestamp","datetime_utc","symbol","side","price","amount",
        "position_id","position_type","status","pnl_eur","fees","trade_cost",
        "exchange","is_master","strategy_name"
    ]
    ensure_columns(cur, "trades", need)

    where = ["strategy_name = ?", "timestamp >= ?"]
    params = [strategy, start_ms]
    if end_ms is not None:
        where.append("timestamp < ?")
        params.append(end_ms)
    where_sql = " AND ".join(where)

    sql = f"""
      SELECT {", ".join(need)}
      FROM trades
      WHERE {where_sql}
      ORDER BY position_id, timestamp
    """
    cur.execute(sql, params)
    return cur.fetchall()

def group_by_position(rows):
    groups = {}
    for r in rows:
        pid = r["position_id"]
        groups.setdefault(pid, []).append(r)
    return groups

def summarize_positions(groups):
    report_rows = []
    totals = {"gross":0.0, "fees":0.0, "net":0.0, "wins":0, "losses":0, "count":0}
    for pid, gr in groups.items():
        # master fallback = first row if none flagged
        master = next((rr for rr in gr if rr["is_master"] == 1), gr[0])
        symbol = master["symbol"]
        side0  = master["side"]

        opened_ts = min(safe_float(rr["timestamp"]) for rr in gr)
        closed_ts = max(safe_float(rr["timestamp"]) for rr in gr)
        held_hours = (closed_ts - opened_ts) / 3_600_000.0

        gross = sum(safe_float(rr["pnl_eur"]) for rr in gr)
        fees  = sum(safe_float(rr["fees"]) for rr in gr)
        net   = gross - fees
        cost  = sum(safe_float(rr["trade_cost"]) for rr in gr)
        roi_pct = safe_div(net, cost) * 100.0
        partials = max(0, len(gr) - 1)

        totals["gross"]  += gross
        totals["fees"]   += fees
        totals["net"]    += net
        totals["count"]  += 1
        if net > 0:
            totals["wins"] += 1
        elif net < 0:
            totals["losses"] += 1

        report_rows.append({
            "master_id": master["id"],
            "position_id": pid,
            "symbol": symbol,
            "side": side0,
            "trades": len(gr),
            "partials": partials,
            "held_hours": round(held_hours, 2),
            "gross_pnl_eur": round(gross, 2),
            "fees_eur": round(fees, 2),
            "net_pnl_eur": round(net, 2),
            "spent_eur": round(cost, 2),
            "roi_pct": round(roi_pct, 2),
            "opened_utc": min(rr["datetime_utc"] for rr in gr),
            "closed_utc": max(rr["datetime_utc"] for rr in gr),
        })

    report_rows.sort(key=lambda x: x["closed_utc"])
    return report_rows, totals

def write_csv(out_path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def write_txt(out_path, summary_text):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

def render_summary(strategy, start, end, rows, totals):
    lines = []
    lines.append(f"=== Trend Report: strategy='{strategy}' from {start} to {end or 'now'} (UTC) ===")
    lines.append(f"Positions: {totals['count']} | Wins: {totals['wins']} | Losses: {totals['losses']}")
    lines.append(f"Gross PnL: €{totals['gross']:.2f} | Fees: €{totals['fees']:.2f} | Net: €{totals['net']:.2f}")
    if rows:
        avg_roi = sum(r["roi_pct"] for r in rows) / len(rows)
        avg_hold = sum(r["held_hours"] for r in rows) / len(rows)
        lines.append(f"Avg ROI/position: {avg_roi:.2f}% | Avg hold: {avg_hold:.2f}h")

        # Top/bottom by net
        top = sorted(rows, key=lambda x: x["net_pnl_eur"], reverse=True)[:5]
        bot = sorted(rows, key=lambda x: x["net_pnl_eur"])[:5]

        lines.append("\nTop 5 by net PnL:")
        for r in top:
            lines.append(f"  {r['symbol']}  net=€{r['net_pnl_eur']:.2f}  roi={r['roi_pct']:.2f}%  held={r['held_hours']}h  closed={r['closed_utc']}")

        lines.append("\nBottom 5 by net PnL:")
        for r in bot:
            lines.append(f"  {r['symbol']}  net=€{r['net_pnl_eur']:.2f}  roi={r['roi_pct']:.2f}%  held={r['held_hours']}h  closed={r['closed_utc']}")
    return "\n".join(lines)

def parse_args():
    ap = argparse.ArgumentParser(description="One-shot Trend strategy report")
    ap.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB (default: market_data.db)")
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY, help="strategy_name (default: trend_4h)")
    ap.add_argument("--start", default=DEFAULT_START, help="start date UTC YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", help="end date UTC YYYY-MM-DD (exclusive)")
    ap.add_argument("--outdir", default="logs", help="output directory for CSV/TXT (default: logs)")
    return ap.parse_args()

def main():
    args = parse_args()
    start_ms = epoch_ms(args.start)
    end_ms = epoch_ms(args.end) if args.end else None

    if not os.path.exists(args.db):
        print(f"❌ DB not found: {args.db}")
        return

    conn = sqlite3.connect(args.db)
    try:
        # sanity check: strategy exists
        strategies = read_distinct_strategies(conn.cursor())
        if args.strategy not in strategies:
            print(f"⚠️  strategy_name '{args.strategy}' not found in DB.")
            if strategies:
                print("   Available:", strategies)
            conn.close()
            return

        rows = load_trades(conn, args.strategy, start_ms, end_ms)
        if not rows:
            print("No trades matched your filter "
                  f"(strategy={args.strategy}, start={args.start}, end={args.end or 'now'}).")
            conn.close()
            return

        groups = group_by_position(rows)
        report_rows, totals = summarize_positions(groups)

        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
        base = f"trend_report_{args.strategy}_{stamp}"
        csv_path = os.path.join(args.outdir, base + ".csv")
        txt_path = os.path.join(args.outdir, base + ".txt")

        write_csv(csv_path, report_rows)
        summary_txt = render_summary(args.strategy, args.start, args.end, report_rows, totals)
        write_txt(txt_path, summary_txt)

        # Also print to console
        print(summary_txt)
        print(f"\nCSV: {csv_path}")
        print(f"TXT: {txt_path}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
