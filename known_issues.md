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

## ADX regime is currently strict

The current `trend_4h` strategy uses a hard 4h ADX threshold around 22. This is safe, but may be too strict for crypto trend setups where trend structure is already improving before ADX fully confirms.

Current behavior:
- 4h ADX below threshold causes a hard skip.
- Some potentially valid early/borderline trend setups may be filtered out.

Preferred future behavior:
- ADX < 16: hard skip.
- ADX 16-22: borderline regime; allow only if EMA structure, DI direction, ADX slope, 1h entry quality, candle direction, and sideways filter are supportive.
- ADX >= 22: normal trend mode.

Decision:
- Keep current strict ADX behavior during cleanup.
- Revisit after `trend_strategy_4h.py` cleanup and improved signal/skip logging.
- Implement first in dryrun, optionally with lower risk multiplier for borderline setups.

## Daily snapshot trigger

`TrendStrategy4H._daily_snapshot_if_due()` only sends between 17:30 and 17:34, but it is currently tied to strategy execution. If no symbol triggers `execute_strategy()` during that window, no snapshot is sent.

Decision:
- Keep daily stats.
- Later call snapshot check from the executor loop so it runs independently of new candle triggers.
