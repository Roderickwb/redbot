# Bot Roadmap

Deze notitie legt het huidige plan vast voor de volgende fase van de bot.

De eerste grote uitdaging is afgevinkt: de bot draait technisch, heeft live gedraaid, schrijft data weg, maakt GPT-beslissingen, logt strategy events, labelt uitkomsten en bouwt coin profiles op basis van learning.

De volgende fase gaat over het hart van de bot: beter zien, beter meten, beter risico nemen en daarna pas slimmer automatiseren.

## 1. Chart Vision Layer

Doel: GPT beter laten "zien" wat er in de markt gebeurt.

Uitbreiden van de chart/context-features:
- pullback depth;
- afstand tot EMA20/EMA50;
- trend age;
- candle quality score;
- wick rejection score;
- volatility state / ATR-percentiel;
- afstand tot support/resistance;
- higher-low / lower-high structuur;
- volume/volatility expansion;
- entry timing: early, clean, late, noisy.

Waarom: GPT ziet geen plaatje zoals TradingView, maar alleen de data die wij meesturen. Betere features betekent betere beslissingen.

## 2. GPT Decision Analytics

Doel: GPT-beslissingen meetbaar maken.

We loggen inmiddels:
- scores.trend;
- scores.entry;
- scores.risk;
- scores.learning;
- scores.sentiment;
- primary_veto;
- learning_effect;
- risk_notes;
- journal_tags.

Volgende stap:
- rapportage bouwen op deze velden;
- meten welke veto's terecht waren;
- meten wanneer GPT te streng of te soepel was;
- vergelijken met latere outcomes en counterfactual trade-resultaten.

Voorbeelden:
- HOLD door `weak_entry`, maar counterfactual later positief;
- OPEN ondanks `COUNTERFACTUAL_EDGE_NEGATIVE`, daarna verlies;
- `FILTER_REVIEW` cases die toch goede kansen bleken.

## 3. Market Regime Layer

Doel: marktbreed bepalen of de bot agressief, normaal of defensief moet zijn.

Te bouwen signalen:
- risk-on / risk-off;
- BTC/ETH trend als marktanker;
- alt strength / weakness;
- trendmarkt versus chopmarkt;
- volatility regime;
- meltdown / recovery context.

Waarom: een goede coin setup is minder betrouwbaar als de brede markt risk-off is.

## 4. Risk Engine

Doel: de bot beschermen en sizing slimmer maken.

Te verbeteren:
- daily loss limit;
- weekly drawdown guard;
- cooldown na verliesreeks;
- max aantal open trades;
- max exposure per coin/sector;
- max correlated alt exposure;
- volatility-adjusted sizing;
- risk multiplier per coin plus market regime.

Waarom: winstgevendheid komt niet alleen van entries. Risk management bepaalt of de bot slechte periodes overleeft.

## 5. Real Sentiment & News Layer

Doel: echte externe context toevoegen, maar pas nadat chart/risk goed staat.

Mogelijke bronnen:
- Fear & Greed voor macro sentiment;
- BTC/ETH chain/market sentiment;
- coin relative strength;
- headlines/news API;
- macro risk flags;
- event-risk filters.

Regel: sentiment mag een schone setup ondersteunen of een borderline setup blokkeren, maar mag nooit een rommelige chart redden.

## 6. Shadow / Backtest Prompt Versions

Doel: prompt- en strategywijzigingen objectief vergelijken voordat ze live gedrag sturen.

Te bouwen:
- replay van oude strategy_events;
- prompt V2 versus V3 vergelijken;
- oude beslissing versus nieuwe beslissing;
- counterfactual outcome per versie;
- rapport met verbetering/verslechtering per coin en per veto.

Waarom: zonder shadow/backtest voelt elke promptverbetering als hoop. Met shadow/backtest kunnen we meten of een nieuwe brain echt beter is.

## Belangrijk Principe

De bot moet niet te vroeg autonoom zijn in het aanpassen van regels. Eerst moeten observatie, labeling, rapportage en risk controls betrouwbaar zijn.

## Operationele Monitoring

Naast de learning-roadmap moet de bot dagelijks melden of de basis gezond is.

Actieve/bedoelde checks:
- GPT timeout / zero-confidence percentage;
- ontbrekende structured GPT-output zoals scores of primary_veto;
- stale candles;
- learning profile count;
- pending/labeled strategy events;
- databasegrootte, WAL-bestand en vrije schijfruimte.

Module:
- `src.analysis.bot_alerts_reporter`

Doel:
- dagelijks Telegram-bericht met health digest;
- expliciete waarschuwing als GPT timeouts of fallback-HOLDs te vaak voorkomen;
- voorkomen dat technische problemen worden verward met inhoudelijke HOLD-beslissingen.

## Advisor Layer

Stap 2 is de advieslaag: de bot leest zijn eigen rapporten en maakt daar conclusies uit.

Module:
- `src.analysis.bot_advisor`
- `src.analysis.daily_analysis_job` als centrale dagrun die learning, GPT-decision report, chart-vision QA, health alerts en advisor achter elkaar ververst.

Inputs:
- learning report;
- profile proposals;
- GPT decision report;
- chart vision QA report;
- bot alerts report.

Output:
- `analysis/bot_advisor/latest_bot_advice.json`
- `analysis/daily/latest_daily_analysis_job.json`

Principe:
- advisor mag conclusies trekken en aanbevelingen doen;
- advisor past nog niets automatisch aan;
- adviezen met `requires_human_approval=true` moeten eerst handmatig beoordeeld worden.

De volgorde blijft:
1. beter zien;
2. beter meten;
3. beter regime herkennen;
4. beter risico beheren;
5. sentiment/nieuws uitbreiden;
6. prompt/strategy versies objectief vergelijken;
7. pas daarna gecontroleerde autonome verbeteradviezen of rule updates.
