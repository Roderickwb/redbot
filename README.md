# aanzetten van de bot
cd ~/redbot
source venv/bin/activate
python -m src.main

# Alleen readm logs
cd ~/redbot/logs
tail -f naammodule.log

# streamlit starten
cd ~/redbot
source venv/bin/activate
streamlit run src/dashboard/dashboard.py

# Gebruik van Linux‐tools: grep, less, tail -f
tail -f pullback_strategy.log toont je live de laatste logregels.
grep -i "OPEN SHORT" pullback_strategy.log filtert direct alleen de regels met dat keyword.
less +F pullback_strategy.log opent het log in “scrollbare” modus, en Shift+F gaat naar de live tail.

# check of bot nog draait
ps aux | grep main.pycd 

Kill pip

pkill -9 python

# git pull Pi
cd ~/redbot
ls -a
git pull origin master

# toegang tot .env in Pi
cd ~/redbot
nano .env

# inhoud .env checken
cat .env


# Bot kill dan moet je twee regels krijgen waarbij 1234 fictuef is
redbot   5678  0.0  0.1  ... grep python # betekend dat die al gestopt is
kill 1234 # pip staat hierboen achter de 0.1 op de ....

# in SQlite verwijder alle trades en of trade_signals
DELETE FROM trade_signals;
DELETE FROM trades;
DELETE FROM trades WHERE id = ?;

BEGIN TRANSACTION;
DELETE FROM trade_signals;
DELETE FROM trades;
COMMIT;

# prullenbak in pi leegmaken
rm -rf ~/.local/share/Trash/*
# comtroleren of er voldoende ruimt is op de pi
df -h


# We gaan drie smaken onderscheiden:

1. ENVIRONMENT=production

USE_WEBSOCKET = True
PAPER_TRADING = False (echte orders)

2. ENVIRONMENT=paper

USE_WEBSOCKET = True (live data)
PAPER_TRADING = True (fake orders)

3. ENVIRONMENT=development

USE_WEBSOCKET = False (geen live data)
PAPER_TRADING = True (fake orders)

# bestanden checken die in repo zitten
cd /pad/naar/jouw/project
git ls-files

# als je coder wijzig o nieuwe files aanmaakt, gebruik je nu het standaard Git-werkproces
git add .
git commit -m "Mijn aanpassing"
git push

# symbol_id
1. XBT/EUR 
2. ETH/EUR 
3. XRP/EUR 
4. XDG/EUR 
5. ADA/EUR 
6. SOL/EUR 
7. DOT/EUR 
8. MATIC/EUR 
9. TRX/EUR 
10. LTC/EUR 
11. LINK/EUR 
12. XLM/EUR 
13. UNI/EUR 
14. ATOM/EUR 
15. ALGO/EUR 
16. AAVE/EUR 
17. ETC/EUR 
18. SAND/EUR 
19. AVAX/EUR 
20. BCH/EUR




flowchart TB
    A([Start]) --> B{Check Fail-safes?}                                 
    B -- Ja --> C([Stop trading<br>(Fail-safe)]) 
    B -- Nee --> D[Haal Data op: <br/>Daily & 4H Candles]
    D --> E{Data beschikbaar?}
    E -- Nee --> F([Stop trading<br>(Geen data)])
    E -- Ja --> G[Bereken RSI daily & 4H + Bepaal Trend]
    G --> H[Haal H1-data op + Bereken ATR]
    H --> I{H1-data voldoende?}
    I -- Nee --> J([Stop trading<br>(Geen H1 data)])
    I -- Ja --> K[Haal 15m-data & <br/>Check Pullback (RSI, MACD,...)]
    K --> L[Analyse Depth Trend <br/>(Orderbook) + ML-signaal]
    L --> M{Hebben we al<br/>een positie?}
    M -- Ja --> N([Manage Open Position<br/>(Partial TP, Trailing...)])
    M -- Nee --> O{Pullback detectie + <br/>Voldoet aan RSI/MACD/Depth/ML?}
    O -- Nee --> P([Geen actie])
    O -- Ja --> Q[[Open Position<br/>(BUY/SELL)]]
    Q --> R([Evt. +25% invest?])
    R --> S([Einde van Run])
    N --> S([Einde van Run])

A. Start
De strategie begint met het uitvoeren van execute_strategy(symbol).
B. Check Fail-safes?
Logica: Voer _check_fail_safes(symbol) uit.
Indicatoren:
Balance-check (max_daily_loss_pct): Kijk of de dagelijkse verlieslimiet is overschreden.
Flash Crash-check (flash_crash_timeframe, flash_crash_drop_pct): Check of er een extreme daling is.
Beslissing:
Ja → Ga naar C: Stop met traden voor deze run.
Nee → Ga verder naar D.
C. Stop trading (Fail-safe)
De strategie keert terug (geen orders).
D. Haal Data op: Daily & 4H Candles
Logica: _fetch_and_indicator(symbol, "1d") en _fetch_and_indicator(symbol, "4h")
Hier worden de dagelijkse en 4-uurs-candles opgehaald en indicatoren (zoals RSI, MACD) berekend.
E. Data beschikbaar?
Check of de DataFrames niet leeg zijn.
Nee → F: Stop (geen data).
Ja → G.
F. Stop trading (Geen data)
Er is niet voldoende candle-data. De strategie staakt voor nu.
G. Bereken RSI daily & 4H + Bepaal Trend
Logica: _check_trend_direction(df_daily, df_h4)
Indicator: RSI van daily & RSI van 4H.
Drempels:
daily_bull_rsi, daily_bear_rsi
h4_bull_rsi, h4_bear_rsi
Als RSI op daily + 4H boven de “bull drempel” is, beschouwt de strategie het als bullish trend.
Als RSI op daily + 4H onder de “bear drempel” is, beschouwt de strategie het als bearish trend.
Anders → range.
H. Haal H1-data op + Bereken ATR
Logica: _fetch_and_indicator(symbol, "1h") → _calculate_atr()
Indicator: ATR op H1-candles.
Dit helpt bij het vaststellen van stop-loss, take-profit en trailing.
I. H1-data voldoende?
Check of er voldoende candles zijn voor ATR (bijv. 14).
Nee → J: Stop.
Ja → K.
J. Stop trading (Geen H1 data)
De strategie keert terug.
K. Haal 15m-data & Check Pullback
Logica: _fetch_and_indicator(symbol, "15m"), daarna _detect_pullback().
Indicatoren: RSI, MACD op 15m.
Drempel:
pullback_threshold_pct, pullback_rolling_window
Hier wordt gekeken of de koers “x%” onder de recente high is (bull) of “x%” boven de recente low (bear).
Ook kijk je naar RSI- en MACD-signalen (rsi_val en macd_signal_score).
L. Analyse Depth Trend (Orderbook) + ML-signaal
Logica:
_analyze_depth_trend_instant(symbol) → depth_score
Bied- vs. vraag-volume in het orderboek.
Drempels: depth_threshold_bull, depth_threshold_bear.
_ml_predict_signal(df_daily) → ml_signal
Gebaseerd op RSI/MACD/volume op daily.
Deze scores worden meegenomen in de beslissing.
M. Hebben we al een positie?
Ja → N (Manage Open Position).
Nee → O.
N. Manage Open Position
Logica: _manage_open_position(...)
Indicatoren:
ATR (H1) → Take-Profit (TP1, TP2) + TrailingStop.
PnL-berekening → partial closes (bijv. 25% op TP1).
Voert SELL- of BUY-porties uit (afhankelijk van long/short).
O. Pullback detectie + Voldoet aan RSI/MACD/Depth/ML?
Wordt er inderdaad een pullback gedetecteerd?
Is rsi_val >= rsi_bull_threshold (of <= rsi_bear_threshold) en
macd_signal_score >= (of <=) macd_bull_threshold (of macd_bear_threshold)?
depth_score >= depth_threshold_bull (of <= depth_threshold_bear)?
ml_signal >= 0 (of <= 0)?
Nee → P: Geen actie.
Ja → Q: Open positie.
P. Geen actie
De strategie doet niets en eindigt hier.
Q. Open Position (BUY/SELL)
Logica: _open_position(...)
Berekent hoeveel er gekocht of verkocht kan worden (afhankelijk van de balance en position_size_pct).
Slaat “open” trade op in de database en voegt toe aan open_positions.
R. Eventueel +25% invest?
Check of accumulate_threshold is behaald (bijv. accumulate_threshold=1.25).
Zo ja, investeer extra (b.v. 10% i.p.v. 5% position_size) bij de volgende pullback.
S. Einde van Run
De strategie keert terug en zal bij de volgende iteratie de hele cyclus herhalen.
3. Kernindicatoren en hun Rollen
RSI (Relative Strength Index)

Op Daily & 4H: om de globale trend (bull/bear/range) te bepalen.
Op 15m: om de pullback te verfijnen (combinatie met MACD).
MACD (Moving Average Convergence Divergence)

Op 15m: signaal om te bepalen of we bullish (MACD > MACD-signaal) of bearish (MACD < MACD-signaal) zijn.
ATR (Average True Range)

Op H1: voor het inschatten van stop-loss, take-profit en trailing.
Depth Trend

Orderbook-analyse: volume aan de bids vs. de asks.
depth_score > 0 → “bullish orderbook”, depth_score < 0 → “bearish orderbook”.
ML-signaal

Op Daily: extra signaal (bijvoorbeeld +1 = bull, -1 = bear).
Pullback Threshold

Bepaalt hoe ver de koers moet terugvallen tegenover de recent high (bull) of recent low (bear) om te spreken van “pullback”.
Fail-safes

Dagelijks verlieslimiet (max_daily_loss_pct) en flash-crash-drempel.


# Config.py 
- databasepad 
streamlit run src/dashboard/dashboard.py
- cd C:\Users\My ACER\PycharmProjects\PythonProject4\data

ta==0.10.2

# to do: 
1. df staat voor dataframe (tabel) en kan beter per class worden omgenaamd naar bv candelstick_df etc.
2. 10 min candles zelf genereren voor je strategy


# alle log moet via logger.info en niet logging.info of logging.debug.

# PVA: Mijn advies: gebruik een Market object + statische IndicatorAnalysis. Dat maakt je code schoon en duidelijk, en is vrij standaard in veel bots.
Laten we daarna “stap voor stap” elke dag 1 deel fixen en testen, in plaats van alles in één keer.
Zo eindig je met een stabiele codebase die duidelijk is ingedeeld:

Market (OOP) → data ophalen per markt,
IndicatorAnalysis (statisch) → pure berekening,
Strategy / ML (OOP) → beslissingen.

# Samenvatting
Je log toont dat de code feilloos draait en indicatoren correct berekent – goed nieuws!
Het feit dat alles ‘Neutraal’ is, betekent dat je beslissingsvoorwaarden (in je strategie) nooit “waar” worden voor kopen/verkopen. Dat is logisch als je thresholds niet geraakt worden.
Wil je ‘Kopen’ of ‘Verkopen’ zien, moet je je conditionele logica aanscherpen (RSI < 30 => “Kopen”, RSI > 70 => “Verkopen”, etc.).
Zo krijg je ook in de logs andere beslissingen dan ‘Neutraal’.

# regelmatig prunen, db_manager.prune_old_candles(days=30)  # interval=None, dus alles 
# db_manager.prune_old_candles(days=60, interval='1m')


