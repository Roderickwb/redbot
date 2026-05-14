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
FINAL_STATUSES = {"rejected", "auto_applied", "archived"}
PROMOTABLE_SEEN_COUNT = 3
PROMOTABLE_MIN_DAYS = 2
DAY_MS = 24 * 60 * 60 * 1000


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
    evidence = rec.get("evidence") or {}
    hypotheses = evidence.get("hypotheses") or []
    if hypotheses and isinstance(hypotheses[0], dict) and hypotheses[0].get("rule_id"):
        payload = {
            "area": rec.get("area"),
            "hypothesis_rule_id": hypotheses[0].get("rule_id"),
        }
    elif evidence_key := _evidence_stable_key(rec, evidence):
        payload = {
            "area": rec.get("area"),
            "evidence_key": evidence_key,
        }
    else:
        payload = {
            "area": rec.get("area"),
            "finding": rec.get("finding"),
            "recommendation": rec.get("recommendation"),
        }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _evidence_stable_key(rec: dict, evidence: dict) -> Optional[str]:
    area = str(rec.get("area") or "")
    recommendation = str(rec.get("recommendation") or "")

    if "missing_scores_pct" in evidence:
        return "gpt_missing_structured_scores"
    if "readiness" in evidence and area == "ml_edge_model":
        status = (evidence.get("readiness") or {}).get("status", "unknown")
        return f"ml_edge_readiness_{status}"
    if "dataset_summary" in evidence and area == "ml_edge_model":
        return "ml_edge_dataset_summary"
    if "zero_conf_pct" in evidence:
        return "gpt_zero_confidence_rate"
    if "scored_events" in evidence:
        return "gpt_scored_sample_size"
    if "cf_avg_r" in evidence and area == "gpt_decision":
        return "gpt_counterfactual_avg_r"
    if "symbols" in evidence:
        return f"{area}_symbols_{_hash_key(evidence.get('symbols'))}"
    if "loaded_events" in evidence and area == "chart_vision":
        return "chart_vision_sample_size"
    if "loaded_candidates" in evidence and area == "opportunities":
        return "opportunity_sample_size"
    if "hold_rate_pct" in evidence and "cf_avg_r" in evidence:
        return "opportunity_hold_rate_positive_cf"
    if "held_large_positive_opportunities" in evidence:
        return "opportunity_held_large_positive"
    if "patterns" in evidence:
        return f"{area}_patterns_{_pattern_key(evidence.get('patterns'))}"
    if "top_patterns" in evidence:
        return f"{area}_top_patterns_{_pattern_key(evidence.get('top_patterns'))}"
    if "pattern" in evidence and "numeric" in evidence:
        return f"{area}_feature_contrast_{_hash_key(evidence.get('pattern'))}"
    if "risk_mode" in evidence and "breadth" in evidence:
        return f"market_regime_{area}_{_hash_key(rec.get('finding'))}"
    if "attention_case" in evidence:
        return f"{area}_{evidence.get('attention_case')}"
    if "verdict" in evidence and area == "shadow_experiments":
        return f"shadow_experiment_verdict_{evidence.get('verdict')}"
    if "overlap_count" in evidence and area == "shadow_experiments":
        return "shadow_experiment_overlap_groups"
    if "promotion_status" in evidence and area == "promotion_gate":
        return f"promotion_gate_{evidence.get('promotion_status')}"
    if recommendation.startswith("Treat this as transition noise"):
        return f"{area}_transition_noise"
    return None


def _hash_key(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _pattern_key(patterns: Any) -> str:
    if not isinstance(patterns, list) or not patterns:
        return "none"
    first = patterns[0] if isinstance(patterns[0], dict) else {}
    payload = {
        "group": first.get("group"),
        "pattern": first.get("pattern") or first.get("name"),
        "interpretation": first.get("interpretation"),
    }
    return _hash_key(payload)


def _stable_hypothesis_id(hypothesis: dict) -> str:
    rule_id = hypothesis.get("rule_id")
    if rule_id:
        return str(rule_id)
    payload = {
        "group": hypothesis.get("group"),
        "pattern": hypothesis.get("pattern"),
        "hypothesis_type": hypothesis.get("hypothesis_type"),
        "proposed_action": hypothesis.get("proposed_action"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _age_days(first_ts: int, last_ts: int) -> int:
    if not first_ts or not last_ts:
        return 1
    return max(1, int((last_ts - first_ts) // DAY_MS) + 1)


def _extract_hypotheses(rec: dict) -> list[dict]:
    evidence = rec.get("evidence") or {}
    hypotheses = evidence.get("hypotheses") or []
    return [
        hypothesis
        for hypothesis in hypotheses
        if isinstance(hypothesis, dict)
    ]


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
                existing["missing_count"] = 0
                self._sync_hypotheses(existing, rec, now_ms=now_ms, now_utc=now_utc)
                updated += 1
            else:
                item = {
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
                    "missing_count": 0,
                }
                self._sync_hypotheses(item, rec, now_ms=now_ms, now_utc=now_utc)
                registry["items"][rec_id] = item
                created += 1

        missing = self._mark_missing_items(registry, seen_ids, now_ms=now_ms, now_utc=now_utc)

        registry["meta"] = {
            "updated_ts": now_ms,
            "updated_utc": now_utc,
            "source_status": advice.get("status"),
            "source_summary": advice.get("summary", {}),
            "last_sync": {
                "seen": len(seen_ids),
                "created": created,
                "updated": updated,
                "missing_active": missing,
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

    def cleanup(
        self,
        apply: bool = False,
        missing_count: int = 2,
        stale_days: int = 14,
    ) -> dict:
        registry = self._load_registry()
        now_ms = int(time.time() * 1000)
        now_utc = _utc_now()
        candidates = []

        for item in (registry.get("items") or {}).values():
            if item.get("status") != "proposed":
                continue
            reasons = []
            item_missing_count = int(item.get("missing_count", 0) or 0)
            last_seen_ts = int(item.get("last_seen_ts", 0) or 0)
            days_since_seen = _age_days(last_seen_ts, now_ms) if last_seen_ts else 999

            if item_missing_count >= missing_count:
                reasons.append(f"missing_count>={missing_count}")
            if days_since_seen >= stale_days:
                reasons.append(f"stale_days>={stale_days}")

            if not reasons:
                continue
            candidates.append({
                "id": item.get("id"),
                "area": item.get("area"),
                "priority": item.get("priority"),
                "finding": item.get("finding"),
                "missing_count": item_missing_count,
                "days_since_seen": days_since_seen,
                "reasons": reasons,
            })

            if apply:
                item["status"] = "archived"
                item.setdefault("decision_history", []).append({
                    "ts": now_ms,
                    "utc": now_utc,
                    "status": "archived",
                    "note": "registry cleanup: " + ", ".join(reasons),
                })

        if apply:
            registry["meta"] = {
                **(registry.get("meta") or {}),
                "updated_ts": now_ms,
                "updated_utc": now_utc,
                "last_cleanup": {
                    "applied": True,
                    "archived": len(candidates),
                    "missing_count": missing_count,
                    "stale_days": stale_days,
                },
            }
            self._save_registry(registry)
            summary = self.summary(registry=registry)
            _write_json(self.summary_path, summary)
        else:
            summary = self.summary(registry=registry)

        return {
            "ok": True,
            "applied": apply,
            "candidates": len(candidates),
            "missing_count": missing_count,
            "stale_days": stale_days,
            "items": candidates[:50],
            "summary": {
                "total": summary.get("total"),
                "by_status": summary.get("by_status", {}),
                "hypothesis_summary": summary.get("hypothesis_summary", {}),
            },
        }

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
            "hypothesis_summary": self._hypothesis_summary(items),
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

    def _mark_missing_items(self, registry: dict, seen_ids: set[str], now_ms: int, now_utc: str) -> int:
        missing = 0
        for rec_id, item in (registry.get("items") or {}).items():
            if rec_id in seen_ids:
                continue
            if item.get("status") not in ACTIVE_STATUSES:
                continue
            item["missing_count"] = int(item.get("missing_count", 0) or 0) + 1
            item["last_missing_ts"] = now_ms
            item["last_missing_utc"] = now_utc
            missing += 1
        return missing

    def _compact_item(self, item: dict) -> dict:
        compact = {
            "id": item.get("id"),
            "status": item.get("status"),
            "priority": item.get("priority"),
            "area": item.get("area"),
            "finding": item.get("finding"),
            "recommendation": item.get("latest_recommendation"),
            "requires_human_approval": item.get("requires_human_approval"),
            "seen_count": item.get("seen_count"),
            "missing_count": item.get("missing_count", 0),
            "last_seen_utc": item.get("last_seen_utc"),
        }
        hypotheses = self._compact_hypotheses(item)
        if hypotheses:
            compact["hypotheses"] = hypotheses
        return compact

    def _priority_rank(self, priority: Any) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(priority), 0)

    def _sync_hypotheses(self, item: dict, rec: dict, now_ms: int, now_utc: str) -> None:
        hypotheses = _extract_hypotheses(rec)
        if not hypotheses:
            return

        tracked = item.setdefault("hypotheses", {})
        for hypothesis in hypotheses:
            hyp_id = _stable_hypothesis_id(hypothesis)
            existing = tracked.get(hyp_id)
            if existing:
                existing["last_seen_ts"] = now_ms
                existing["last_seen_utc"] = now_utc
                existing["seen_count"] = int(existing.get("seen_count", 0)) + 1
            else:
                existing = {
                    "id": hyp_id,
                    "first_seen_ts": now_ms,
                    "first_seen_utc": now_utc,
                    "last_seen_ts": now_ms,
                    "last_seen_utc": now_utc,
                    "seen_count": 1,
                }
                tracked[hyp_id] = existing

            existing.update({
                "hypothesis_type": hypothesis.get("hypothesis_type"),
                "group": hypothesis.get("group"),
                "pattern": hypothesis.get("pattern"),
                "proposed_action": hypothesis.get("proposed_action"),
                "confidence": hypothesis.get("confidence"),
                "matches": hypothesis.get("matches"),
                "cf_avg_r": hypothesis.get("cf_avg_r"),
                "cf_positive_rate_pct": hypothesis.get("cf_positive_rate_pct"),
                "cf_loss_rate_pct": hypothesis.get("cf_loss_rate_pct"),
                "requires_human_approval": hypothesis.get("requires_human_approval", True),
                "guardrails": hypothesis.get("guardrails", []),
                "sample_cases": hypothesis.get("sample_cases", [])[:5],
            })
            self._refresh_hypothesis_state(existing)

    def _refresh_hypothesis_state(self, hypothesis: dict) -> None:
        age_days = _age_days(
            int(hypothesis.get("first_seen_ts", 0) or 0),
            int(hypothesis.get("last_seen_ts", 0) or 0),
        )
        seen_count = int(hypothesis.get("seen_count", 0) or 0)
        confidence = str(hypothesis.get("confidence", "low"))
        promotable = (
            seen_count >= PROMOTABLE_SEEN_COUNT
            and age_days >= PROMOTABLE_MIN_DAYS
            and confidence in {"medium", "high"}
        )

        if promotable:
            stability = "promotable"
        elif seen_count >= PROMOTABLE_SEEN_COUNT:
            stability = "stable_same_day"
        elif seen_count >= 2:
            stability = "repeat"
        else:
            stability = "new"

        hypothesis["age_days"] = age_days
        hypothesis["stability"] = stability
        hypothesis["promotable"] = promotable
        hypothesis["promotion_requirements"] = {
            "min_seen_count": PROMOTABLE_SEEN_COUNT,
            "min_age_days": PROMOTABLE_MIN_DAYS,
            "requires_confidence": ["medium", "high"],
        }

    def _compact_hypotheses(self, item: dict) -> list[dict]:
        hypotheses = list((item.get("hypotheses") or {}).values())
        hypotheses.sort(
            key=lambda hyp: (
                1 if hyp.get("promotable") else 0,
                int(hyp.get("seen_count", 0) or 0),
                float(hyp.get("cf_avg_r", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return [
            {
                "id": hyp.get("id"),
                "stability": hyp.get("stability"),
                "promotable": hyp.get("promotable"),
                "seen_count": hyp.get("seen_count"),
                "age_days": hyp.get("age_days"),
                "confidence": hyp.get("confidence"),
                "hypothesis_type": hyp.get("hypothesis_type"),
                "pattern": hyp.get("pattern"),
                "matches": hyp.get("matches"),
                "cf_avg_r": hyp.get("cf_avg_r"),
                "cf_loss_rate_pct": hyp.get("cf_loss_rate_pct"),
            }
            for hyp in hypotheses[:5]
        ]

    def _hypothesis_summary(self, items: list[dict]) -> dict:
        hypotheses = []
        for item in items:
            for hypothesis in (item.get("hypotheses") or {}).values():
                self._refresh_hypothesis_state(hypothesis)
                hypotheses.append({
                    **hypothesis,
                    "recommendation_id": item.get("id"),
                    "area": item.get("area"),
                    "priority": item.get("priority"),
                })

        by_stability = {}
        for hypothesis in hypotheses:
            stability = hypothesis.get("stability", "unknown")
            by_stability[stability] = by_stability.get(stability, 0) + 1

        promotable = [
            hypothesis for hypothesis in hypotheses
            if hypothesis.get("promotable")
        ]
        promotable.sort(
            key=lambda hyp: (
                self._priority_rank(hyp.get("priority")),
                float(hyp.get("cf_avg_r", 0.0) or 0.0),
                int(hyp.get("seen_count", 0) or 0),
            ),
            reverse=True,
        )

        return {
            "total": len(hypotheses),
            "by_stability": by_stability,
            "promotable": len(promotable),
            "top_promotable": [
                {
                    "recommendation_id": hyp.get("recommendation_id"),
                    "hypothesis_id": hyp.get("id"),
                    "area": hyp.get("area"),
                    "confidence": hyp.get("confidence"),
                    "hypothesis_type": hyp.get("hypothesis_type"),
                    "pattern": hyp.get("pattern"),
                    "seen_count": hyp.get("seen_count"),
                    "age_days": hyp.get("age_days"),
                    "cf_avg_r": hyp.get("cf_avg_r"),
                    "cf_loss_rate_pct": hyp.get("cf_loss_rate_pct"),
                }
                for hyp in promotable[:10]
            ],
        }


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

    cleanup = sub.add_parser("cleanup", help="Find or archive stale proposed recommendations.")
    cleanup.add_argument("--apply", action="store_true", help="Archive cleanup candidates.")
    cleanup.add_argument("--missing-count", type=int, default=2, help="Archive proposals missing from this many syncs.")
    cleanup.add_argument("--stale-days", type=int, default=14, help="Archive proposals not seen for this many days.")

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
    elif args.command == "cleanup":
        result = registry.cleanup(
            apply=args.apply,
            missing_count=args.missing_count,
            stale_days=args.stale_days,
        )
    else:
        result = registry.summary()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
