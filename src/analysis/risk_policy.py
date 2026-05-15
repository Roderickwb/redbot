# ============================================================
# src/analysis/risk_policy.py
# ============================================================
"""
Read-only risk policy layer.

Combines market regime, learning coin profiles and current safety reports into
one explicit per-symbol risk policy. This module does not change live sizing,
block trades, or write coin profiles. It only creates an auditable report that
can later be wired into live execution behind a separate approval gate.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "risk")
DEFAULT_LATEST_FILE = "latest_risk_policy_report.json"
DEFAULT_MARKET_REGIME = os.path.join("analysis", "market_regime", "latest_market_regime.json")
DEFAULT_PROFILE_PROPOSALS = os.path.join("analysis", "strategy_events", "latest_strategy_profile_proposals.json")
DEFAULT_PROMOTION_GATE = os.path.join("analysis", "promotion_gate", "latest_promotion_gate_report.json")
DEFAULT_APPROVAL_INBOX = os.path.join("analysis", "approvals", "latest_approval_inbox.json")
DEFAULT_ML_EDGE_REPORT = os.path.join("analysis", "ml_models", "latest_edge_model_report.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except Exception:
        return default


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


def _parse_json(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class RiskPolicyBuilder:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        market_regime_path: str = DEFAULT_MARKET_REGIME,
        profile_proposals_path: str = DEFAULT_PROFILE_PROPOSALS,
        promotion_gate_path: str = DEFAULT_PROMOTION_GATE,
        approval_inbox_path: str = DEFAULT_APPROVAL_INBOX,
        ml_edge_path: str = DEFAULT_ML_EDGE_REPORT,
    ):
        self.db = db or DatabaseManager(db_path=DB_FILE)
        self.market_regime_path = market_regime_path
        self.profile_proposals_path = profile_proposals_path
        self.promotion_gate_path = promotion_gate_path
        self.approval_inbox_path = approval_inbox_path
        self.ml_edge_path = ml_edge_path

    def build_report(self, strategy_name: str = "trend_4h", limit: int = 1000) -> dict:
        market = _load_json(self.market_regime_path, {})
        proposals = _load_json(self.profile_proposals_path, {})
        promotion = _load_json(self.promotion_gate_path, {})
        approval = _load_json(self.approval_inbox_path, {})
        ml_edge = _load_json(self.ml_edge_path, {})

        profiles = self._load_coin_profiles(strategy_name=strategy_name)
        proposal_map = proposals.get("proposals", {}) or {}
        symbols = sorted(set(profiles.keys()) | set(proposal_map.keys()))
        policies = [
            self._policy_for_symbol(
                symbol=symbol,
                live_profile=profiles.get(symbol, {}),
                proposal=proposal_map.get(symbol, {}),
                market=market,
                promotion=promotion,
                approval=approval,
                ml_edge=ml_edge,
            )
            for symbol in symbols[:limit]
        ]

        return {
            "meta": {
                "created_utc": _utc_now(),
                "strategy_name": strategy_name,
                "read_only": True,
                "live_enforcement": False,
                "sources": {
                    "market_regime": self.market_regime_path,
                    "profile_proposals": self.profile_proposals_path,
                    "promotion_gate": self.promotion_gate_path,
                    "approval_inbox": self.approval_inbox_path,
                    "ml_edge_model": self.ml_edge_path,
                    "coin_profiles_table": "coin_profiles",
                },
            },
            "guardrails": {
                "risk_down_can_be_automated_later": True,
                "risk_up_requires_human_approval": True,
                "entry_rule_changes_require_promotion_gate": True,
                "live_wiring_required_before_enforcement": True,
            },
            "market_context": self._market_context(market),
            "summary": self._summary(policies, market, promotion, approval, ml_edge),
            "policies": policies,
        }

    def _load_coin_profiles(self, strategy_name: str) -> dict[str, dict]:
        rows = self.db.execute_query(
            """
            SELECT symbol, risk_multiplier, bias, n_trades, expectancy_r, source, updated_ts, profile_json
            FROM coin_profiles
            WHERE strategy_name=?
            ORDER BY symbol
            """,
            (strategy_name,),
        )
        profiles: dict[str, dict] = {}
        for row in rows or []:
            symbol, risk_mult, bias, n_trades, expectancy_r, source, updated_ts, profile_json = row
            profile = _parse_json(profile_json)
            profile.setdefault("symbol", symbol)
            profile.setdefault("risk_multiplier", _safe_float(risk_mult, 1.0))
            profile.setdefault("bias", bias or "neutral")
            profile.setdefault("n_trades", _safe_int(n_trades))
            profile.setdefault("expectancy_R", _safe_float(expectancy_r))
            profile.setdefault("source", source or "coin_profiles")
            profile.setdefault("updated_ts", _safe_int(updated_ts))
            profiles[str(symbol)] = profile
        return profiles

    def _policy_for_symbol(
        self,
        symbol: str,
        live_profile: dict,
        proposal: dict,
        market: dict,
        promotion: dict,
        approval: dict,
        ml_edge: dict,
    ) -> dict:
        source_profile = live_profile or self._proposal_as_profile(symbol, proposal)
        flags = set(source_profile.get("flags") or proposal.get("flags") or [])
        market_mult = _safe_float(market.get("risk_multiplier"), 1.0)
        profile_mult = _safe_float(source_profile.get("risk_multiplier"), 1.0)
        flag_mult = self._flag_multiplier(flags, source_profile)
        final_mult = self._clamp_multiplier(market_mult * profile_mult * flag_mult)
        risk_mode = self._risk_mode(final_mult)
        directional_policy = self._directional_policy(
            base_multiplier=final_mult,
            profile=source_profile,
            flags=flags,
            market=market,
        )
        actions = self._actions(
            symbol,
            final_mult,
            directional_policy,
            source_profile,
            flags,
            market,
            promotion,
            approval,
            ml_edge,
        )

        return {
            "symbol": symbol,
            "policy_mode": risk_mode,
            "risk_multiplier": final_mult,
            "directional_policy": directional_policy,
            "components": {
                "market_multiplier": round(market_mult, 2),
                "profile_multiplier": round(profile_mult, 2),
                "flag_multiplier": round(flag_mult, 2),
            },
            "bias": source_profile.get("bias", "neutral"),
            "learning_confidence": source_profile.get("learning_confidence") or proposal.get("confidence"),
            "n_trades": _safe_int(source_profile.get("n_trades")),
            "expectancy_R": round(_safe_float(source_profile.get("expectancy_R")), 4),
            "flags": sorted(flags),
            "actions": actions,
            "reasons": self._reasons(source_profile, flags, market, promotion, approval, ml_edge),
            "source": source_profile.get("source", "profile_proposals"),
        }

    def _proposal_as_profile(self, symbol: str, proposal: dict) -> dict:
        metrics = proposal.get("metrics", {}) or {}
        flags = self._profile_style_flags(proposal.get("flags", []) or [])
        return {
            "symbol": symbol,
            "risk_multiplier": _safe_float(proposal.get("risk_multiplier"), 1.0),
            "bias": proposal.get("bias", "neutral"),
            "learning_confidence": proposal.get("confidence"),
            "n_trades": _safe_int(metrics.get("trade_open")),
            "expectancy_R": _safe_float(metrics.get("cf_avg_r")),
            "flags": flags,
            "source": "profile_proposals",
        }

    def _profile_style_flags(self, flags: list[Any]) -> list[str]:
        result = [str(flag) for flag in flags]
        raw = {str(flag).lower() for flag in flags}
        mappings = {
            "trade_quality_negative": "DRAWDOWN_RISK",
            "range_breakout_candidate": "RANGE_BREAKOUT_CANDIDATE",
            "filters_may_be_too_strict": "FILTER_REVIEW",
            "counterfactual_edge_positive": "COUNTERFACTUAL_EDGE_POSITIVE",
            "counterfactual_edge_negative": "COUNTERFACTUAL_EDGE_NEGATIVE",
            "sample_low": "SAMPLE_LOW",
        }
        for source, target in mappings.items():
            if source in raw and target not in result:
                result.append(target)
        return result

    def _flag_multiplier(self, flags: set[str], profile: dict) -> float:
        multiplier = 1.0
        normalized = {str(flag).upper() for flag in flags}
        if "DRAWDOWN_RISK" in normalized:
            multiplier *= 0.75
        if "COUNTERFACTUAL_EDGE_NEGATIVE" in normalized:
            multiplier *= 0.75
        if "SAMPLE_LOW" in normalized or profile.get("learning_confidence") == "low":
            multiplier *= 0.85
        return multiplier

    def _clamp_multiplier(self, value: float) -> float:
        if value <= 0:
            return 0.0
        return round(min(1.0, max(0.25, value)), 2)

    def _risk_mode(self, multiplier: float) -> str:
        if multiplier <= 0.35:
            return "minimal"
        if multiplier <= 0.55:
            return "defensive"
        if multiplier < 0.85:
            return "cautious"
        return "normal"

    def _directional_policy(
        self,
        base_multiplier: float,
        profile: dict,
        flags: set[str],
        market: dict,
    ) -> dict:
        long_multiplier = base_multiplier
        short_multiplier = base_multiplier
        long_reasons = []
        short_reasons = []
        regime = market.get("regime")
        directional_bias = market.get("directional_bias")
        profile_bias = str(profile.get("bias") or "neutral")
        normalized_flags = {str(flag).upper() for flag in flags}

        if regime == "risk_off":
            long_multiplier = min(long_multiplier, 0.5)
            long_reasons.append("market_risk_off_caps_longs")
            if directional_bias == "short_or_cash":
                short_reasons.append("market_risk_off_allows_short_or_cash_only")
        elif regime == "risk_on":
            short_multiplier = min(short_multiplier, 0.7)
            short_reasons.append("market_risk_on_caps_shorts")
        elif regime == "chop":
            long_multiplier = min(long_multiplier, 0.7)
            short_multiplier = min(short_multiplier, 0.7)
            long_reasons.append("market_chop_caps_directional_entries")
            short_reasons.append("market_chop_caps_directional_entries")

        if profile_bias in {"long_edge", "long_bias"}:
            short_multiplier = min(short_multiplier, 0.75)
            short_reasons.append("profile_bias_favors_long")
        elif profile_bias in {"short_edge", "short_bias"}:
            long_multiplier = min(long_multiplier, 0.75)
            long_reasons.append("profile_bias_favors_short")

        if "DRAWDOWN_RISK" in normalized_flags or "COUNTERFACTUAL_EDGE_NEGATIVE" in normalized_flags:
            long_reasons.append("profile_negative_edge_or_drawdown")
            short_reasons.append("profile_negative_edge_or_drawdown")

        return {
            "long": {
                "risk_multiplier": self._clamp_multiplier(long_multiplier),
                "policy_mode": self._risk_mode(self._clamp_multiplier(long_multiplier)),
                "can_increase_risk": False,
                "reasons": long_reasons or ["base_policy"],
            },
            "short": {
                "risk_multiplier": self._clamp_multiplier(short_multiplier),
                "policy_mode": self._risk_mode(self._clamp_multiplier(short_multiplier)),
                "can_increase_risk": False,
                "reasons": short_reasons or ["base_policy"],
            },
        }

    def _actions(
        self,
        symbol: str,
        multiplier: float,
        directional_policy: dict,
        profile: dict,
        flags: set[str],
        market: dict,
        promotion: dict,
        approval: dict,
        ml_edge: dict,
    ) -> list[dict]:
        actions = []
        regime = market.get("regime")
        directional_bias = market.get("directional_bias")
        if regime == "risk_off":
            actions.append({
                "action": "cap_new_long_risk",
                "mode": "read_only",
                "suggested_multiplier": (directional_policy.get("long") or {}).get("risk_multiplier", min(multiplier, 0.5)),
            })
            if directional_bias == "short_or_cash":
                actions.append({
                    "action": "prefer_short_or_cash",
                    "mode": "read_only",
                    "short_multiplier": (directional_policy.get("short") or {}).get("risk_multiplier", multiplier),
                })
        if multiplier < 1.0:
            actions.append({
                "action": "risk_down",
                "mode": "read_only",
                "suggested_multiplier": multiplier,
            })
        if "FILTER_REVIEW" in {str(flag).upper() for flag in flags}:
            actions.append({"action": "inspect_filter_strictness", "mode": "review_only"})
        if self._approval_reject_candidates(approval) > 0:
            actions.append({"action": "do_not_relax_entries_from_blocked_experiments", "mode": "guardrail"})
        if self._promotion_blocked(promotion) > 0:
            actions.append({"action": "respect_promotion_gate_blocks", "mode": "guardrail"})
        if (ml_edge.get("readiness") or {}).get("status") != "ready":
            actions.append({"action": "ignore_ml_for_live_risk", "mode": "guardrail"})
        return actions

    def _reasons(
        self,
        profile: dict,
        flags: set[str],
        market: dict,
        promotion: dict,
        approval: dict,
        ml_edge: dict,
    ) -> list[str]:
        reasons = []
        if market.get("regime"):
            reasons.append(
                f"market_regime={market.get('regime')} risk_mode={market.get('risk_mode')} multiplier={market.get('risk_multiplier')}"
            )
        if profile.get("risk_multiplier") is not None:
            reasons.append(f"profile_multiplier={profile.get('risk_multiplier')}")
        for flag in sorted(flags):
            reasons.append(f"profile_flag={flag}")
        if self._promotion_blocked(promotion):
            reasons.append(f"promotion_gate_blocked={self._promotion_blocked(promotion)}")
        if self._approval_reject_candidates(approval):
            reasons.append(f"approval_reject_candidates={self._approval_reject_candidates(approval)}")
        readiness = ml_edge.get("readiness") or {}
        if readiness.get("status"):
            reasons.append(f"ml_edge_status={readiness.get('status')}")
        return reasons

    def _market_context(self, market: dict) -> dict:
        return {
            "regime": market.get("regime"),
            "risk_mode": market.get("risk_mode"),
            "directional_bias": market.get("directional_bias"),
            "risk_multiplier": market.get("risk_multiplier"),
            "breadth": market.get("breadth", {}),
            "flags": market.get("flags", []),
        }

    def _summary(
        self,
        policies: list[dict],
        market: dict,
        promotion: dict,
        approval: dict,
        ml_edge: dict,
    ) -> dict:
        by_mode = Counter(policy.get("policy_mode") for policy in policies)
        by_long_mode = Counter(((policy.get("directional_policy") or {}).get("long") or {}).get("policy_mode") for policy in policies)
        by_short_mode = Counter(((policy.get("directional_policy") or {}).get("short") or {}).get("policy_mode") for policy in policies)
        risk_down = [p for p in policies if _safe_float(p.get("risk_multiplier"), 1.0) < 1.0]
        cap_longs = [
            p for p in policies
            if any(action.get("action") == "cap_new_long_risk" for action in p.get("actions", []))
        ]
        long_risk_down = [
            p for p in policies
            if _safe_float(((p.get("directional_policy") or {}).get("long") or {}).get("risk_multiplier"), 1.0) < 1.0
        ]
        short_risk_down = [
            p for p in policies
            if _safe_float(((p.get("directional_policy") or {}).get("short") or {}).get("risk_multiplier"), 1.0) < 1.0
        ]
        short_not_increased = [
            p for p in policies
            if not bool(((p.get("directional_policy") or {}).get("short") or {}).get("can_increase_risk"))
        ]
        avg_mult = (
            round(sum(_safe_float(p.get("risk_multiplier"), 1.0) for p in policies) / len(policies), 3)
            if policies else 1.0
        )
        avg_long_mult = (
            round(sum(_safe_float(((p.get("directional_policy") or {}).get("long") or {}).get("risk_multiplier"), 1.0) for p in policies) / len(policies), 3)
            if policies else 1.0
        )
        avg_short_mult = (
            round(sum(_safe_float(((p.get("directional_policy") or {}).get("short") or {}).get("risk_multiplier"), 1.0) for p in policies) / len(policies), 3)
            if policies else 1.0
        )
        return {
            "total_symbols": len(policies),
            "by_policy_mode": dict(by_mode),
            "by_long_policy_mode": dict(by_long_mode),
            "by_short_policy_mode": dict(by_short_mode),
            "average_risk_multiplier": avg_mult,
            "average_long_risk_multiplier": avg_long_mult,
            "average_short_risk_multiplier": avg_short_mult,
            "risk_down_symbols": [
                {"symbol": p.get("symbol"), "risk_multiplier": p.get("risk_multiplier"), "policy_mode": p.get("policy_mode")}
                for p in risk_down[:20]
            ],
            "long_risk_down_symbols": [
                {
                    "symbol": p.get("symbol"),
                    "risk_multiplier": ((p.get("directional_policy") or {}).get("long") or {}).get("risk_multiplier"),
                    "policy_mode": ((p.get("directional_policy") or {}).get("long") or {}).get("policy_mode"),
                }
                for p in long_risk_down[:20]
            ],
            "short_risk_down_symbols": [
                {
                    "symbol": p.get("symbol"),
                    "risk_multiplier": ((p.get("directional_policy") or {}).get("short") or {}).get("risk_multiplier"),
                    "policy_mode": ((p.get("directional_policy") or {}).get("short") or {}).get("policy_mode"),
                }
                for p in short_risk_down[:20]
            ],
            "cap_new_long_symbols": [
                {
                    "symbol": p.get("symbol"),
                    "risk_multiplier": ((p.get("directional_policy") or {}).get("long") or {}).get("risk_multiplier", p.get("risk_multiplier")),
                }
                for p in cap_longs[:20]
            ],
            "short_risk_not_increased_symbols": [
                {
                    "symbol": p.get("symbol"),
                    "risk_multiplier": ((p.get("directional_policy") or {}).get("short") or {}).get("risk_multiplier"),
                }
                for p in short_not_increased[:20]
            ],
            "market_regime": market.get("regime"),
            "promotion_blocked": self._promotion_blocked(promotion),
            "approval_reject_candidates": self._approval_reject_candidates(approval),
            "ml_edge_status": (ml_edge.get("readiness") or {}).get("status"),
        }

    def _promotion_blocked(self, promotion: dict) -> int:
        return _safe_int((promotion.get("summary") or {}).get("blocked"))

    def _approval_reject_candidates(self, approval: dict) -> int:
        return _safe_int((approval.get("summary") or {}).get("reject_candidate"))


def run_risk_policy(
    strategy_name: str = "trend_4h",
    limit: int = 1000,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = RiskPolicyBuilder(db=db).build_report(strategy_name=strategy_name, limit=limit)
    finally:
        db.close_connection()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build read-only risk policy report.")
    parser.add_argument("--strategy-name", type=str, default="trend_4h", help="Coin profile strategy name.")
    parser.add_argument("--limit", type=int, default=1000, help="Max symbols to include.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_risk_policy(
        strategy_name=args.strategy_name,
        limit=args.limit,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "summary": report.get("summary", {}),
        "guardrails": report.get("guardrails", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
