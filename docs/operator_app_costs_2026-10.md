# Operator app and simulation costs

The default operator page is now frontend/operator.html, served by FastAPI.
It requires no Node build or external asset requests. The existing React source
and full report endpoints remain available for development and diagnostics.
The mobile endpoint is a compact projection of those reports, not a replacement
for the stored evidence. The Pi fixture shrinks from roughly 6.9 MB to 42 KB.
Gzip is also enabled.

The page has Overview, Decisions, Tests and Positions, with evidence in
disclosures. Position history displays master positions once, with entry and
exit fees, rather than mixing aggregate masters with their executions.
Open-position PnL is realized only, including paid entry fees. Freshness,
missing reports, failed requests and invalid tokens are visible. A failed
refresh preserves previously loaded data. The operator token is kept in
sessionStorage for the current browser session only.

Simulation R now includes configured entry and exit fees. TP1 and the remainder
are charged using their corresponding exit prices and fractions. OPEN_END is
valued assuming liquidation at the horizon; that assumption is explicit in the
label. Gross R, fees in R and net R are retained separately. This does not add
slippage, spread or financing modeling.

The learning job upgrades up to report_limit historical simulation labels per
cycle, newest first, without rebuilding candles or altering realized PnL.
The default batch is 5,000. Subsequent cycles continue the migration.
Restriction outcomes also apply cost accounting on read over all their history,
so older events need not wait for migration. Missing price/risk inputs are not
used as cost-aware evidence. Results and model metrics can change substantially
after costs are included. A sizing reduction that reduces losses is not proof
of a profitable strategy or grounds for automatic live activation.

Validation: eleven regression tests, read-only Pi data checks, mobile and desktop
browser checks, filters, a mock decision submission, token errors and connection
errors. Preview interactions do not record real operator decisions.

Deployment after user commit and push:

```bash
./scripts/pi_update.sh
sudo systemctl restart redbot-operator-app.service
```

The second command is needed because pi_update restarts the trading service,
not the existing operator app process. Reload the page after the service restart.
The static page is served with Cache-Control: no-cache.

Refresh and settings icons are from lucide-static 0.468.0 (ISC); the accompanying
LICENSE-lucide.txt contains the license.
