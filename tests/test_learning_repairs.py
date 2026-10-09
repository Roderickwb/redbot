import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.analysis.trade_accounting import position_accounting
from src.analysis.adaptive_restriction_outcome_tracker import AdaptiveRestrictionOutcomeTracker
from src.analysis.strategy_profile_proposer import StrategyProfileProposer
from src.analysis.strategy_event_outcome_labeler import StrategyEventOutcomeLabeler, OutcomeConfig
from src.analysis.simulation_costs import apply_simulation_costs
from src.operator_app.backend.data import mobile_bundle
from src.analysis.adaptive_restrictions import AdaptiveRestrictionBuilder


class LearningRepairs(unittest.TestCase):
    def test_experiment_survives_missing_proposal_and_freezes_rule(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "recommendations.json"
            decisions = Path(root) / "decisions.jsonl"
            item = {"id": "coin-test", "candidate_type": "per_coin_learning_candidate",
                    "status": "approved_shadow", "evidence": {"coin": {"symbol": "ALGO-EUR",
                    "best_coin_rule_candidate": {"rule_id": "half", "action_type": "reduced_risk", "multiplier": 0.5}}}}
            path.write_text(json.dumps({"items": [item]}))
            builder = AdaptiveRestrictionBuilder(str(path), str(Path(root)/"outcomes.json"), root, str(decisions))
            first = builder.build()["restrictions"][0]
            item["evidence"]["coin"]["best_coin_rule_candidate"]["multiplier"] = 0.75
            path.write_text(json.dumps({"items": [item]}))
            self.assertEqual(builder.build()["restrictions"][0]["risk_multiplier"], 0.5)
            path.write_text(json.dumps({"items": []}))
            self.assertEqual(builder.build()["restrictions"][0]["restriction_id"], first["restriction_id"])
            decisions.write_text(json.dumps({"source_id": "coin-test", "action": "freeze"})+"\n")
            paused = builder.build()
            self.assertEqual(paused["restrictions"], [])
            self.assertEqual(len(paused["suspended_restrictions"]), 1)

    def test_historical_approval_recovers_paused_not_active(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "recommendations.json"
            decisions = Path(root) / "decisions.jsonl"
            item = {"id": "old-test", "candidate_type": "entry_rule_candidate", "evidence": {
                "source_cluster": {"dimension": "direction", "value": "short"},
                "best_candidate": {"rule_id": "half", "action_type": "reduced_risk", "multiplier": 0.5}}}
            path.write_text('{"items": []}')
            decisions.write_text(json.dumps({"source_id": "old-test", "action": "approve", "decision_id": "old",
                                             "source_snapshot": {"items": [item]}})+"\n")
            builder = AdaptiveRestrictionBuilder(str(path), str(Path(root)/"outcomes.json"), root, str(decisions))
            report = builder.build()
            self.assertFalse(report["restrictions"])
            self.assertEqual(report["suspended_restrictions"][0]["lifecycle_status"], "recovered_pending_review")
            self.assertFalse(builder.build()["restrictions"])

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
            outcome = json.dumps({"counterfactual_trade": {"r_multiple": -1, "entry_price": 100, "exit_price": 90, "risk_per_unit": 10}})
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
                self.assertAlmostEqual(result["summary"]["delta_r"], 6.399)

    def test_simulated_costs_long_short_partial_and_idempotence(self):
        for direction in ('long', 'short'):
            trade = {'direction': direction, 'r_multiple': 0.05, 'entry_price': 100,
                     'exit_price': 100.5, 'risk_per_unit': 10, 'exit_reason': 'OPEN_END'}
            result = apply_simulation_costs(trade, 0.0035)
            self.assertAlmostEqual(result['fees_r'], 0.0702)
            self.assertLess(result['r_multiple'], 0)
            self.assertEqual(result['label'], 'cf_loss')
            self.assertEqual(apply_simulation_costs(result, 0.0035), result)
        partial = apply_simulation_costs({'r_multiple': 1.5, 'entry_price': 100,
                                        'exit_price': 110, 'tp1_price': 120, 'tp1_hit': True,
                                        'tp1_portion_pct': 0.5, 'risk_per_unit': 10}, 0.0035)
        self.assertAlmostEqual(partial['fees_r'], 0.0753)

    def test_invalid_cost_inputs_are_not_claimed_as_net(self):
        self.assertEqual(apply_simulation_costs({'r_multiple': 1}, 0.0035)['cost_status'], 'missing_prices_or_risk')

    def test_simulator_charges_real_exit_notional_for_long_and_short(self):
        labeler = StrategyEventOutcomeLabeler(db=Mock(), config=OutcomeConfig(sl_atr_mult=1, tp1_atr_mult=1, trailing_atr_mult=1))
        long = labeler._simulate_counterfactual_trade('long', 100, 1, [{'timestamp': 1, 'high': 100.5, 'low': 98.9, 'close': 99}])
        short = labeler._simulate_counterfactual_trade('short', 100, 1, [{'timestamp': 1, 'high': 101.1, 'low': 99.5, 'close': 101}])
        self.assertAlmostEqual(long['r_multiple'], -1.6965)
        self.assertAlmostEqual(short['r_multiple'], -1.7035)
        partial = labeler._simulate_counterfactual_trade('long', 100, 1, [
            {'timestamp': 1, 'high': 102, 'low': 100, 'close': 101.5},
            {'timestamp': 2, 'high': 102, 'low': 101, 'close': 101.5}])
        self.assertAlmostEqual(partial['gross_r_multiple'], 1)
        self.assertAlmostEqual(partial['r_multiple'], 0.2965)
        self.assertTrue(partial['tp1_hit'])

    def test_migration_changes_only_simulated_cost_fields(self):
        labeler = StrategyEventOutcomeLabeler(db=Mock())
        raw = json.dumps({'counterfactual_trade': {'r_multiple': 0.05, 'entry_price': 100, 'exit_price': 100.5, 'risk_per_unit': 10}, 'realized_trade': {'pnl_eur': 0.123}, 'label': 'existing_move_label'})
        labeler.db.execute_query.return_value = [(1, raw)]
        with patch.object(labeler, '_update_event_outcome') as update:
            stats = labeler.refresh_simulation_costs(apply=True)
        result = update.call_args.kwargs['outcome']
        self.assertEqual(result['realized_trade']['pnl_eur'], 0.123)
        self.assertEqual(result['label'], 'existing_move_label')
        self.assertLess(result['counterfactual_trade']['r_multiple'], 0)
        self.assertEqual(stats['updated'], 1)

    def test_mobile_bundle_does_not_copy_raw_evidence_or_execution_snapshots(self):
        reports = {'recommendations': {'items': [{'id': 'abc', 'title': 'Test', 'operator_evidence': [{'label': 'Aantal', 'value': '12'}], 'evidence': {'large': 'x'*1000000}}]},
                   'operator_decisions': {'recent': [{'action': 'approve', 'source_snapshot': {'large': 'x'*1000000}}]}}
        with patch('src.operator_app.backend.data.report', side_effect=lambda name: reports.get(name, {'status': 'OK'})), patch('src.operator_app.backend.data.recent_positions', return_value={'rows': [], 'status': 'OK'}):
            bundle = mobile_bundle()
        self.assertLess(len(json.dumps(bundle)), 10000)
        self.assertNotIn('evidence', bundle['recommendations']['items'][0])
        self.assertNotIn('source_snapshot', bundle['operator_decisions']['recent'][0])
        self.assertEqual(bundle['recommendations']['items'][0]['operator_evidence'][0]['value'], '12')

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
