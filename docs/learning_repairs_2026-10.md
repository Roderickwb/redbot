# Learning repairs, October 2026

Position reports and realized trade labels now subtract both entry and exit
fees. Child PnL already includes exit fees; opening fees are derived from
master accumulated fees minus child exit fees. Historical trades are not
rewritten. The next learning run refreshes legacy realized labels without
rebuilding their candle counterfactuals. The reporter also reads current
realized results, including trades that closed after their first event label.

The regular learning job refreshes coin profile evidence and timestamps while
preserving existing risk multiplier, bias and hold behavior. Indicator/ML
context is preserved and refreshed later by the existing context integrator.
The explicit behavioral profile-write option retains its existing semantics.
Fresh context can influence GPT decisions; it does not automatically apply
proposed sizing changes.

Adaptive outcome tracking reads all retained events containing restriction
metadata. The limit argument is now a read-batch size, not a history cutoff.
The source of truth remains strategy_events: do not delete experiment events
without archiving them. Results still measure counterfactual R, not reconciled
account profit. Simulated R and stored historical candle labels are not
recomputed by this accounting repair.

Verification: six regression tests plus read-only validation on Pi history.
661 closed positions reconstruct to EUR -101.729376 after recorded fees.
The current coin restriction recovers 12 labeled events rather than 2; the
current cluster restriction has no matching events even in the full history.
This is evidence for review, not automatic live promotion.

After commit and push, run scripts/pi_update.sh on the Pi. Its smoke check
runs the regression tests and the regular analysis. No service installation
or manual database migration is required. Profile context and outcome
conclusions will change on that first cycle, so inspect the resulting cards.
