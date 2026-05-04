# Known Issues

## Executor restart around hourly candle

After a bot restart around an hourly boundary, `Executor.last_closed_ts` is empty because it is in-memory state.

This can cause the executor to treat the latest candle in the DB as "new", even if that candle is from the previous hour and was already processed before the restart.

Current impact:
- Low risk.
- The quarterly fallback can still pick up the real new candle later.
- `TrendStrategy4H.last_processed_1h_ts` and GPT state reduce duplicate GPT/trade risk.

Decision:
- Accepted for now.
- Do not change candle trigger logic unless this causes duplicate GPT calls, duplicate trades, or consistently late entries.

## Coin profile sample size

Coin profiles are active and loaded by `trend_strategy_4h`, but current conclusions are based on a small number of trades because historical data was partially lost after the old SD-card setup failed.

Current impact:
- Profiles are usable for context and risk hints.
- Confidence is limited while `LOW_SAMPLE` flags remain common.

Decision:
- Keep coin profiles enabled.
- Re-evaluate after more dryrun/live trade history has accumulated.


## Daily snapshot trigger

`TrendStrategy4H._daily_snapshot_if_due()` only sends between 17:30 and 17:34, but it is currently tied to strategy execution. If no symbol triggers `execute_strategy()` during that window, no snapshot is sent.

Decision:
- Keep daily stats.
- Later call snapshot check from the executor loop so it runs independently of new candle triggers.

## Strategy event outcomes vs actual trade outcomes

`strategy_event_outcome_labeler` labels what happened after an event over a fixed lookahead window, currently 8h on 5m candles. This is useful for skip/HOLD learning. For opened trades, the event outcome also includes realized master/child trade result when a `trade_id` is available.

Current impact:
- `open_went_against`, `open_no_followthrough`, and `open_followed_through` describe post-event price action.
- `realized_trade.label` describes the actual managed trade outcome when available.

Decision:
- Keep forward-window labels for setup quality.
- `trade_open` events are enriched with realized master/child trade outcome when a `trade_id` is available.
- Still needed later: reporting/aggregation that compares forward-window setup quality with realized trade result per coin.
