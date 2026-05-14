# ============================================================
# src/analysis/recommendation_registry.py
# ============================================================
"""
Recommendation Registry + Approval Gate.

Stores advisor recommendations with stable IDs and lifecycle status:
- proposed
- approved
- rejected
- auto_applied

This module does not change live trading behavior. It creates the control
surface needed before any future autonomous changes can be safely applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "recommendations")
DEFAULT_REGISTRY_FILE = "recommendation_registry.json"
DEFAULT_LATEST_FILE = "latest_recommendation_registry_summary.json"
ACTIVE_STATUSES = {"proposed", "approved"}
FINAL_STATUSES = {"rejected", "auto_applied"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _stable_id(rec: dict) -> str:
    payload = {
        "area": rec.get("area"),
        "finding": rec.get("finding"),
        "recommendation": rec.get("recommendation"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class RecommendationRegistry:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.registry_path = os.path.join(output_dir, DEFAULT_REGISTRY_FILE)
        self.summary_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)

    def sync_from_advice(self, advice: dict) -> dict:
        registry = self._load_registry()
        now_ms = int(time.time() * 1000)
        now_utc = _utc_now()
        seen_ids = set()
        created = 0
        updated = 0

        for rec in advice.get("recommendations", []) or []:
            rec_id = _stable_id(rec)
            seen_ids.add(rec_id)
            existing = registry["items"].get(rec_id)
            if existing:
                existing["last_seen_ts"] = now_ms
                existing["last_seen_utc"] = now_utc
                existing["seen_count"] = int(existing.get("seen_count", 0)) + 1
                existing["priority"] = rec.get("priority")
                existing["latest_evidence"] = rec.get("evidence", {})
                existing["latest_recommendation"] = rec.get("recommendation")
                existing["requires_human_approval"] = bool(rec.get("requires_human_approval"))
                updated += 1
            else:
                registry["items"][rec_id] = {
                    "id": rec_id,
                    "status": "proposed",
                    "created_ts": now_ms,
                    "created_utc": now_utc,
                    "last_seen_ts": now_ms,
                    "last_seen_utc": now_utc,
                    "seen_count": 1,
                    "priority": rec.get("priority"),
                    "area": rec.get("area"),
                    "finding": rec.get("finding"),
                    "latest_recommendation": rec.get("recommendation"),
                    "requires_human_approval": bool(rec.get("requires_human_approval")),
                    "latest_evidence": rec.get("evidence", {}),
                    "decision_history": [],
                }
                created += 1

        registry["meta"] = {
            "updated_ts": now_ms,
            "updated_utc": now_utc,
            "source_status": advice.get("status"),
            "source_summary": advice.get("summary", {}),
            "last_sync": {
                "seen": len(seen_ids),
                "created": created,
                "updated": updated,
            },
        }
        self._save_registry(registry)
        summary = self.summary(registry=registry)
        _write_json(self.summary_path, summary)
        return summary

    def approve(self, rec_id: str, note: str = "") -> dict:
        return self._set_status(rec_id=rec_id, status="approved", note=note)

    def reject(self, rec_id: str, note: str = "") -> dict:
        return self._set_status(rec_id=rec_id, status="rejected", note=note)

    def mark_auto_applied(self, rec_id: str, note: str = "") -> dict:
        return self._set_status(rec_id=rec_id, status="auto_applied", note=note)

    def summary(self, registry: Optional[dict] = None) -> dict:
        registry = registry or self._load_registry()
        items = list((registry.get("items") or {}).values())
        by_status = {}
        by_area = {}
        for item in items:
            by_status[item.get("status", "unknown")] = by_status.get(item.get("status", "unknown"), 0) + 1
            area = item.get("area", "unknown")
            by_area[area] = by_area.get(area, 0) + 1

        active = [
            item for item in items
            if item.get("status") in ACTIVE_STATUSES
        ]
        active.sort(key=lambda item: (self._priority_rank(item.get("priority")), item.get("last_seen_ts", 0)), reverse=True)

        return {
            "meta": registry.get("meta", {}),
            "registry_path": self.registry_path,
            "summary_path": self.summary_path,
            "total": len(items),
            "by_status": by_status,
            "by_area": by_area,
            "active": [
                self._compact_item(item)
                for item in active[:25]
            ],
        }

    def _set_status(self, rec_id: str, status: str, note: str = "") -> dict:
        registry = self._load_registry()
        item = (registry.get("items") or {}).get(rec_id)
        if not item:
            return {"ok": False, "error": f"unknown recommendation id: {rec_id}"}
        now_ms = int(time.time() * 1000)
        now_utc = _utc_now()
        item["status"] = status
        item.setdefault("decision_history", []).append({
            "ts": now_ms,
            "utc": now_utc,
            "status": status,
            "note": note,
        })
        registry["meta"] = {
            **(registry.get("meta") or {}),
            "updated_ts": now_ms,
            "updated_utc": now_utc,
        }
        self._save_registry(registry)
        summary = self.summary(registry=registry)
        _write_json(self.summary_path, summary)
        return {"ok": True, "id": rec_id, "status": status, "summary": summary}

    def _load_registry(self) -> dict:
        return _load_json(self.registry_path, {"meta": {}, "items": {}})

    def _save_registry(self, registry: dict) -> None:
        _write_json(self.registry_path, registry)

    def _compact_item(self, item: dict) -> dict:
        return {
            "id": item.get("id"),
            "status": item.get("status"),
            "priority": item.get("priority"),
            "area": item.get("area"),
            "finding": item.get("finding"),
            "recommendation": item.get("latest_recommendation"),
            "requires_human_approval": item.get("requires_human_approval"),
            "seen_count": item.get("seen_count"),
            "last_seen_utc": item.get("last_seen_utc"),
        }

    def _priority_rank(self, priority: Any) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(priority), 0)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Manage bot recommendation registry.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Registry output directory.")
    sub = parser.add_subparsers(dest="command")

    sync = sub.add_parser("sync", help="Sync recommendations from an advisor JSON file.")
    sync.add_argument("--advice", type=str, default=os.path.join("analysis", "bot_advisor", "latest_bot_advice.json"))

    sub.add_parser("summary", help="Show registry summary.")

    approve = sub.add_parser("approve", help="Approve a recommendation by id.")
    approve.add_argument("id")
    approve.add_argument("--note", type=str, default="")

    reject = sub.add_parser("reject", help="Reject a recommendation by id.")
    reject.add_argument("id")
    reject.add_argument("--note", type=str, default="")

    auto = sub.add_parser("auto-applied", help="Mark recommendation as auto_applied.")
    auto.add_argument("id")
    auto.add_argument("--note", type=str, default="")

    args = parser.parse_args(list(argv) if argv is not None else None)
    registry = RecommendationRegistry(output_dir=args.output_dir)

    if args.command == "sync":
        advice = _load_json(args.advice, {})
        result = registry.sync_from_advice(advice)
    elif args.command == "approve":
        result = registry.approve(args.id, note=args.note)
    elif args.command == "reject":
        result = registry.reject(args.id, note=args.note)
    elif args.command == "auto-applied":
        result = registry.mark_auto_applied(args.id, note=args.note)
    else:
        result = registry.summary()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
