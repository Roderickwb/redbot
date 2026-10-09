import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.analysis.trade_accounting import position_accounting
from src.analysis.adaptive_restriction_outcome_tracker import AdaptiveRestrictionOutcomeTracker
from src.analysis.strategy_profile_proposer import StrategyProfileProposer
from src.analysis.strategy_event_outcome_labeler import StrategyEventOutcomeLabeler


class LearningRepairs(unittest.TestCase):
    def test_partial_exits_include_entry_fee_once(self):
        master = {"fees": 0.30, "pnl_eur": 0.20}
        children = [{"status": "partial", "fees": 0.10, "pnl_eur": 0.30},
                    {"status": "closed", "fees": 0.10, "pnl_eur": -0.10}]
        result = position_accounting(master, children)
        self.assertAlmostEqual(result["pnl_eur"], 0.10)
        self.assertAlmostEqual(result["entry_fees_eur"], 0.10)
        self.assertAlmostEqual(result["fees_eur"], 0.30)

    def test_opening_fee_can_turn_reported_winner_into_loss(self):
        result = position_accounting({"fees": 0.18},
                                    [{"status": "closed", "fees": 0.08, "pnl_eur": 0.05}])
        self.assertAlmostEqual(result["pnl_eur"], -0.05)

    def test_reject_inconsistent_fees(self):
        with self.assertRaises(ValueError):
            position_accounting({"fees": 0.01}, [{"status": "closed", "fees": 0.1}])

    def test_history_survives_new_unrelated_events_and_small_batches(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = str(Path(root) / "events.db")
            con = sqlite3.connect(db_path)
            con.execute("CREATE TABLE strategy_events(id INTEGER, timestamp INTEGER, symbol TEXT, event_type TEXT, decision_stage TEXT, skip_reason TEXT, gpt_action TEXT, trade_id INTEGER, features_json TEXT, outcome_status TEXT, outcome_json TEXT)")
            features = json.dumps({"adaptive_restriction_sizing": {"restrictions": [{"restriction_id": "test", "before": 1, "after": 0.5}]}})
            outcome = json.dumps({"counterfactual_trade": {"r_multiple": -1}})
            for i in range(12):
                con.execute("INSERT INTO strategy_events VALUES(?,?, 'XBT-EUR','trade_open','open',NULL,NULL,NULL,?,'labeled',?)", (i, i, features, outcome))
            for i in range(12, 50):
                con.execute("INSERT INTO strategy_events VALUES(?,?, 'XBT-EUR','skip','pre',NULL,NULL,NULL,'{}','labeled','{}')", (i, i))
            con.commit()
            con.close()
            restrictions = Path(root) / "restrictions.json"
            restrictions.write_text(json.dumps({"restrictions": [{"restriction_id": "test", "risk_multiplier": 0.5}]}))
            tracker = AdaptiveRestrictionOutcomeTracker(db_path, str(restrictions), root)
            for _ in range(2):
                result = tracker.build(limit=2)
                self.assertEqual(result["summary"]["labeled_events"], 12)
                self.assertEqual(result["summary"]["delta_r"], 6)

    def test_context_refresh_preserves_behavior_and_other_context(self):
        db = Mock()
        db.execute_query.return_value = [(json.dumps({"risk_multiplier": 0.7, "bias": "short", "hold_behavior": "keep", "learned_context": {"edge": 1}, "generated_at_utc": "old"}),)]
        proposer = StrategyProfileProposer()
        with patch.object(proposer, "build_coin_profiles", return_value={"XBT-EUR": {"risk_multiplier": 0.25, "bias": "neutral", "generated_at_utc": "new", "n_trades": 30}}):
            proposer.write_coin_profiles_to_db({}, db=db, context_only=True)
        profile = db.upsert_coin_profile.call_args.kwargs["profile"]
        self.assertEqual(profile["risk_multiplier"], 0.7)
        self.assertEqual(profile["bias"], "short")
        self.assertEqual(profile["generated_at_utc"], "new")
        self.assertEqual(profile["n_trades"], 30)
        self.assertEqual(profile["learned_context"], {"edge": 1})

    def test_historical_label_migration_preserves_counterfactual(self):
        labeler = StrategyEventOutcomeLabeler(db=Mock())
        labeler.db.execute_query.return_value = [(1, 7, json.dumps({"counterfactual_trade": {"r_multiple": 2}, "realized_trade": {"pnl_eur": 1}}))]
        with patch.object(labeler, "_load_realized_trade_outcome", return_value={"pnl_eur": -0.1, "accounting_version": "net_entry_exit_v1"}), patch.object(labeler, "_update_event_outcome") as update:
            self.assertEqual(labeler.refresh_realized_outcomes(apply=True), 1)
        self.assertEqual(update.call_args.kwargs["outcome"]["counterfactual_trade"]["r_multiple"], 2)
        self.assertEqual(update.call_args.kwargs["outcome"]["realized_trade"]["pnl_eur"], -0.1)


if __name__ == "__main__":
    unittest.main()
