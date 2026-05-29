"""Build active paper-mode restrictions from approved operator recommendations.

This is the first closed-loop bridge:
operator approve -> adaptive restriction state -> strategy can consume it.

The output is intentionally conservative:
- no live effect is enabled here;
- rejected/waiting/frozen items are ignored;
- cooldown means dynamic cooldown with reopen criteria, not a permanent ban.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_RECOMMENDATIONS_PATH = os.path.join("analysis", "recommendations", "latest_recommendation_aggregator.json")
DEFAULT_OUTPUT_DIR = os.path.join("analysis", "adaptive_restrictions")
DEFAULT_LATEST_FILE = "latest_adaptive_restrictions.json"

APPROVED_STATUSES = {
    "approved_pending_live_gate",
    "approved_shadow",
    "approved_context_live",
}
SUPPORTED_CANDIDATES = {
    "entry_rule_candidate",
    "per_coin_learning_candidate",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _approved(item: dict) -> bool:
    resolution = item.get("operator_resolution") or {}
    return (
        str(item.get("status") or "") in APPROVED_STATUSES
        or str(resolution.get("status") or "") in APPROVED_STATUSES
        or str(resolution.get("action") or "").lower() == "approve"
    )


def _action_state(action_type: str) -> str:
    action_type = str(action_type or "").lower()
    if action_type in {"reduced_risk", "strict_confirmation", "conditional_cooldown"}:
        return action_type
    if action_type == "cooldown":
        return "conditional_cooldown"
    return "strict_confirmation"


def _restriction_id(item: dict, payload: dict) -> str:
    return f"arest_{_stable_hash({'item': item.get('id'), 'payload': payload})}"


class AdaptiveRestrictionBuilder:
    def __init__(
        self,
        recommendations_path: str = DEFAULT_RECOMMENDATIONS_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.recommendations_path = recommendations_path
        self.output_dir = output_dir

    def build(self) -> dict:
        recommendations = _load_json(self.recommendations_path, {"items": [], "resolved_items": []})
        items = self._candidate_items(recommendations)
        restrictions = [r for item in items for r in self._restrictions_from_item(item)]
        restrictions.sort(key=lambda r: (str(r.get("scope")), str(r.get("symbol")), str(r.get("rule_id"))))

        summary = {
            "approved_items": len(items),
            "active_restrictions": len(restrictions),
            "coin_restrictions": sum(1 for r in restrictions if r.get("scope") == "coin"),
            "cluster_restrictions": sum(1 for r in restrictions if r.get("scope") == "cluster"),
            "reduced_risk": sum(1 for r in restrictions if r.get("state") == "reduced_risk"),
            "strict_confirmation": sum(1 for r in restrictions if r.get("state") == "strict_confirmation"),
            "conditional_cooldown": sum(1 for r in restrictions if r.get("state") == "conditional_cooldown"),
            "paper_effect": bool(restrictions),
            "live_effect": False,
        }
        report = {
            "created_utc": _utc_now(),
            "status": "ACTIVE" if restrictions else "WATCH",
            "summary": summary,
            "restrictions": restrictions,
            "source_path": self.recommendations_path,
            "output_path": os.path.join(self.output_dir, DEFAULT_LATEST_FILE),
            "live_effect": False,
        }
        _write_json(report["output_path"], report)
        return report

    def _candidate_items(self, report: dict) -> list[dict]:
        seen = set()
        result = []
        for item in (report.get("resolved_items") or []) + (report.get("items") or []):
            item_id = item.get("id")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            if item.get("candidate_type") not in SUPPORTED_CANDIDATES:
                continue
            if not _approved(item):
                continue
            result.append(item)
        return result

    def _restrictions_from_item(self, item: dict) -> list[dict]:
        ctype = item.get("candidate_type")
        if ctype == "per_coin_learning_candidate":
            return self._coin_restriction(item)
        if ctype == "entry_rule_candidate":
            return self._entry_cluster_restriction(item)
        return []

    def _base_payload(self, item: dict, best: dict) -> dict:
        action_type = str(best.get("action_type") or "")
        return {
            "source_item_id": item.get("id"),
            "source_candidate_type": item.get("candidate_type"),
            "source_title": item.get("operator_title") or item.get("title"),
            "operator_decision": item.get("operator_decision") or {},
            "state": _action_state(action_type),
            "action_type": action_type or "strict_confirmation",
            "rule_id": best.get("rule_id"),
            "risk_multiplier": _safe_float(best.get("multiplier"), 1.0) if _action_state(action_type) == "reduced_risk" else None,
            "review_after": best.get("review_after") or "na 10 nieuwe paper/shadow signalen of 48 uur, wat eerder komt",
            "reopen_criteria": best.get("reopen_criteria") or [
                "blijft elk uur in analyse",
                "heropen als nieuwe paper/shadow signalen positief worden",
                "heropen als marktregime of 1h/4h structuur duidelijk verbetert",
            ],
            "paper_effect": True,
            "live_effect": False,
        }

    def _coin_restriction(self, item: dict) -> list[dict]:
        evidence = item.get("evidence") or {}
        coin = evidence.get("coin") or {}
        best = coin.get("best_coin_rule_candidate") or {}
        symbol = coin.get("symbol") or item.get("subject")
        if not symbol or not best:
            return []
        payload = self._base_payload(item, best)
        payload.update({
            "scope": "coin",
            "symbol": symbol,
            "match": {"symbol": symbol},
            "reason": f"operator-approved coin learning for {symbol}",
            "evidence": {
                "performance": coin.get("performance") or {},
                "risk_advice": coin.get("risk_advice") or {},
                "entry_feature_context": coin.get("entry_feature_context") or {},
                "best_coin_rule_candidate": best,
            },
        })
        payload["restriction_id"] = _restriction_id(item, payload)
        return [payload]

    def _entry_cluster_restriction(self, item: dict) -> list[dict]:
        evidence = item.get("evidence") or {}
        best = evidence.get("best_candidate") or {}
        cluster = evidence.get("source_cluster") or {}
        dimension = cluster.get("dimension") or (evidence.get("summary") or {}).get("dimension")
        value = cluster.get("value") or (evidence.get("summary") or {}).get("value")
        if not best or not dimension or value is None:
            return []
        payload = self._base_payload(item, best)
        payload.update({
            "scope": "cluster",
            "symbol": None,
            "match": {"dimension": dimension, "value": value},
            "reason": f"operator-approved entry rule for {dimension}={value}",
            "evidence": {
                "source_cluster": cluster,
                "best_candidate": best,
            },
        })
        payload["restriction_id"] = _restriction_id(item, payload)
        return [payload]


def run_adaptive_restrictions(
    recommendations_path: str = DEFAULT_RECOMMENDATIONS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    return AdaptiveRestrictionBuilder(
        recommendations_path=recommendations_path,
        output_dir=output_dir,
    ).build()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build active adaptive paper restrictions from operator approvals.")
    parser.add_argument("--recommendations-path", type=str, default=DEFAULT_RECOMMENDATIONS_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_adaptive_restrictions(
        recommendations_path=args.recommendations_path,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
