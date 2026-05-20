"""Read-only position lifecycle report.

This report checks whether master and child trade rows form a sane position
lifecycle. It does not change live trading behavior. The goal is to catch
bookkeeping issues before the operator app or future autonomy relies on the
position state.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "positions")
DEFAULT_LATEST_FILE = "latest_position_lifecycle_report.json"
DEFAULT_STRATEGY_NAME = "trend_4h"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class PositionLifecycleReport:
    def __init__(self, db_path: str = DB_FILE, strategy_name: str = DEFAULT_STRATEGY_NAME):
        self.db_path = db_path
        self.strategy_name = strategy_name

    def build_report(self, limit: Optional[int] = None) -> dict:
        masters, children, data_quality = self._load_trades(limit=limit)
        masters_by_position: dict[str, list[dict]] = defaultdict(list)
        children_by_position: dict[str, list[dict]] = defaultdict(list)
        for master in masters:
            masters_by_position[str(master.get("position_id") or "")].append(master)
        for child in children:
            children_by_position[str(child.get("position_id") or "")].append(child)

        issues = []
        for position_id, rows in masters_by_position.items():
            if not position_id:
                continue
            if len(rows) > 1:
                issues.append(self._issue(
                    severity="high",
                    code="duplicate_master_position_id",
                    finding=f"Position {position_id} has {len(rows)} master trades.",
                    position_id=position_id,
                    evidence={"master_trade_ids": [row.get("id") for row in rows]},
                ))

        master_ids = {str(master.get("position_id") or "") for master in masters if master.get("position_id")}
        for child in children:
            position_id = str(child.get("position_id") or "")
            if not position_id:
                issues.append(self._issue(
                    severity="medium",
                    code="child_without_position_id",
                    finding=f"Child trade {child.get('id')} has no position_id.",
                    position_id=None,
                    evidence={"trade_id": child.get("id"), "symbol": child.get("symbol")},
                ))
            elif position_id not in master_ids:
                issues.append(self._issue(
                    severity="high",
                    code="orphan_child_trade",
                    finding=f"Child trade {child.get('id')} references missing master position {position_id}.",
                    position_id=position_id,
                    evidence={"trade_id": child.get("id"), "symbol": child.get("symbol")},
                ))

        lifecycles = []
        for master in masters:
            position_id = str(master.get("position_id") or "")
            child_rows = sorted(children_by_position.get(position_id, []), key=lambda row: (_safe_int(row.get("timestamp")), _safe_int(row.get("id"))))
            lifecycle, row_issues = self._lifecycle(master, child_rows)
            lifecycles.append(lifecycle)
            issues.extend(row_issues)

        summary = self._summary(masters, children, lifecycles, issues, data_quality)
        return {
            "created_utc": _utc_now(),
            "status": summary.get("status"),
            "meta": {
                "db_path": self.db_path,
                "strategy_name": self.strategy_name,
                "limit": limit,
                "read_only": True,
                "live_effect": False,
            },
            "summary": summary,
            "issues": issues[:100],
            "lifecycles": lifecycles[:200],
            "data_quality": data_quality,
        }

    def _load_trades(self, limit: Optional[int]) -> tuple[list[dict], list[dict], dict]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            cols = self._columns(con)
            required = {"id", "timestamp", "symbol", "side", "price", "amount", "position_id", "status", "pnl_eur", "fees", "strategy_name", "is_master"}
            missing = sorted(required - set(cols))
            if missing:
                return [], [], {"missing_columns": missing}

            reason_select = ", exit_reason, exit_event_type" if {"exit_reason", "exit_event_type"} <= set(cols) else ""
            master_sql = f"""
                SELECT id, timestamp, datetime_utc, symbol, side, price, amount,
                       position_id, position_type, status, pnl_eur, fees,
                       trade_cost, exchange, strategy_name, is_master{reason_select}
                  FROM trades
                 WHERE strategy_name=?
                   AND is_master=1
                 ORDER BY timestamp DESC, id DESC
            """
            params: list[Any] = [self.strategy_name]
            if limit:
                master_sql += " LIMIT ?"
                params.append(int(limit))
            masters = [dict(row) for row in con.execute(master_sql, params).fetchall()]

            child_sql = f"""
                SELECT id, timestamp, datetime_utc, symbol, side, price, amount,
                       position_id, position_type, status, pnl_eur, fees,
                       trade_cost, exchange, strategy_name, is_master{reason_select}
                  FROM trades
                 WHERE strategy_name=?
                   AND is_master=0
                 ORDER BY timestamp ASC, id ASC
            """
            children = [dict(row) for row in con.execute(child_sql, (self.strategy_name,)).fetchall()]
            return masters, children, {
                "missing_columns": [],
                "reason_columns_available": "exit_reason" in cols and "exit_event_type" in cols,
                "masters_without_position_id": sum(1 for row in masters if not row.get("position_id")),
                "children_without_position_id": sum(1 for row in children if not row.get("position_id")),
            }
        finally:
            con.close()

    @staticmethod
    def _columns(con: sqlite3.Connection) -> set[str]:
        rows = con.execute("PRAGMA table_info(trades)").fetchall()
        return {str(row[1]) for row in rows}

    def _lifecycle(self, master: dict, children: list[dict]) -> tuple[dict, list[dict]]:
        issues = []
        position_id = str(master.get("position_id") or "")
        status = str(master.get("status") or "").lower()
        child_statuses = [str(row.get("status") or "").lower() for row in children]
        child_amount = sum(_safe_float(row.get("amount")) for row in children)
        master_amount = _safe_float(master.get("amount"))
        realized_pnl = _safe_float(master.get("pnl_eur")) or sum(_safe_float(row.get("pnl_eur")) for row in children)
        fees = _safe_float(master.get("fees")) or sum(_safe_float(row.get("fees")) for row in children)
        path = self._path(status, child_statuses)
        last_child = children[-1] if children else {}
        final_exit_reason = (
            str(master.get("exit_reason") or "").strip()
            or str(last_child.get("exit_reason") or "").strip()
            or None
        )

        if not position_id:
            issues.append(self._issue(
                severity="medium",
                code="master_without_position_id",
                finding=f"Master trade {master.get('id')} has no position_id.",
                position_id=None,
                evidence={"trade_id": master.get("id"), "symbol": master.get("symbol")},
            ))
        if master_amount < 0:
            issues.append(self._issue(
                severity="high",
                code="negative_master_amount",
                finding=f"Master trade {master.get('id')} has negative amount.",
                position_id=position_id,
                evidence={"amount": master_amount, "symbol": master.get("symbol")},
            ))
        for child in children:
            if _safe_float(child.get("amount")) <= 0:
                issues.append(self._issue(
                    severity="high",
                    code="non_positive_child_amount",
                    finding=f"Child trade {child.get('id')} has non-positive amount.",
                    position_id=position_id,
                    evidence={"amount": child.get("amount"), "symbol": child.get("symbol")},
                ))

        if status == "closed" and master_amount > 0 and child_amount > 0:
            issues.append(self._issue(
                severity="medium",
                code="closed_master_has_remaining_amount",
                finding=f"Closed position {position_id} still has master amount {master_amount}.",
                position_id=position_id,
                evidence={"master_amount": master_amount, "child_amount": child_amount},
            ))
        if status in {"open", "partial"} and "closed" in child_statuses:
            issues.append(self._issue(
                severity="high",
                code="open_master_has_closed_child",
                finding=f"Open/partial position {position_id} has a closed child trade.",
                position_id=position_id,
                evidence={"status": status, "child_statuses": child_statuses},
            ))
        if status == "partial" and "partial" not in child_statuses:
            issues.append(self._issue(
                severity="medium",
                code="partial_master_without_partial_child",
                finding=f"Partial position {position_id} has no partial child trade.",
                position_id=position_id,
                evidence={"status": status, "child_statuses": child_statuses},
            ))
        if status == "closed" and children and "closed" not in child_statuses:
            issues.append(self._issue(
                severity="low",
                code="closed_master_without_closed_child",
                finding=f"Closed position {position_id} has child rows but no closed child row.",
                position_id=position_id,
                evidence={"child_statuses": child_statuses},
            ))

        return {
            "position_id": position_id or None,
            "master_trade_id": master.get("id"),
            "symbol": master.get("symbol"),
            "side": master.get("side"),
            "status": master.get("status"),
            "path": path,
            "entry_ts": master.get("timestamp"),
            "entry_datetime_utc": master.get("datetime_utc"),
            "master_amount": _round(master_amount),
            "child_amount_sum": _round(child_amount),
            "child_trades": len(children),
            "partial_children": child_statuses.count("partial"),
            "closed_children": child_statuses.count("closed"),
            "realized_pnl_eur": _round(realized_pnl),
            "fees_eur": _round(fees),
            "final_exit_reason": final_exit_reason,
            "issue_count": len(issues),
        }, issues

    @staticmethod
    def _path(status: str, child_statuses: list[str]) -> str:
        if child_statuses.count("partial") and child_statuses.count("closed"):
            return "open_to_partial_to_closed"
        if child_statuses.count("partial") and status in {"open", "partial"}:
            return "open_to_partial_active"
        if child_statuses.count("closed"):
            return "open_to_closed"
        if status in {"open", "partial"}:
            return "active_no_exit_children"
        if status == "closed":
            return "closed_without_exit_children"
        return "unknown"

    @staticmethod
    def _issue(severity: str, code: str, finding: str, position_id: Optional[str], evidence: dict) -> dict:
        return {
            "severity": severity,
            "code": code,
            "finding": finding,
            "position_id": position_id,
            "evidence": evidence,
            "live_effect": False,
        }

    def _summary(self, masters: list[dict], children: list[dict], lifecycles: list[dict], issues: list[dict], data_quality: dict) -> dict:
        by_severity = Counter(str(issue.get("severity") or "unknown") for issue in issues)
        by_code = Counter(str(issue.get("code") or "unknown") for issue in issues)
        by_status = Counter(str(row.get("status") or "unknown").lower() for row in masters)
        by_path = Counter(str(row.get("path") or "unknown") for row in lifecycles)
        high = by_severity.get("high", 0)
        medium = by_severity.get("medium", 0)
        if high:
            status = "ACTION_NEEDED"
            verdict = "lifecycle_integrity_issues"
        elif medium:
            status = "REVIEW"
            verdict = "lifecycle_review_needed"
        else:
            status = "OK"
            verdict = "lifecycle_ok"
        return {
            "status": status,
            "verdict": verdict,
            "master_trades": len(masters),
            "child_trades": len(children),
            "open_masters": by_status.get("open", 0),
            "partial_masters": by_status.get("partial", 0),
            "closed_masters": by_status.get("closed", 0),
            "issue_count": len(issues),
            "high_issues": high,
            "medium_issues": medium,
            "low_issues": by_severity.get("low", 0),
            "by_master_status": dict(by_status),
            "by_lifecycle_path": dict(by_path),
            "by_issue_code": dict(by_code),
            "data_quality": data_quality,
            "top_issues": issues[:5],
            "live_effect": False,
        }


def run_position_lifecycle_report(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    db_path: str = DB_FILE,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    limit: Optional[int] = None,
) -> dict:
    report = PositionLifecycleReport(db_path=db_path, strategy_name=strategy_name).build_report(limit=limit)
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    _write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build read-only position lifecycle report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-path", type=str, default=DB_FILE)
    parser.add_argument("--strategy-name", type=str, default=DEFAULT_STRATEGY_NAME)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_position_lifecycle_report(
        output_dir=args.output_dir,
        db_path=args.db_path,
        strategy_name=args.strategy_name,
        limit=args.limit,
    )
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
