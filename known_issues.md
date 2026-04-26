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

