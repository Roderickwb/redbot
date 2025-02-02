# dashboard.py

import os
import sys
from datetime import datetime, timezone, timedelta
import sqlite3
import pandas as pd
import yaml
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Lokale imports
from src.config.config import DB_FILE, DASHBOARD_LOG_FILE
from src.logger.logger import setup_logger
from src.indicator_analysis.indicators import IndicatorAnalysis, Market
from src.main import db_manager  # Zorg dat db_manager in main.py correct is geïnitialiseerd

# =========================================
# 1) Streamlit basisconfig + log
# =========================================
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

logger = setup_logger('dashboard', DASHBOARD_LOG_FILE)
logger.info("Dashboard gestart.")

def get_current_local_timestamp():
    """
    CET=UTC+1 in milliseconden.
    (LET OP: Eigenlijk werken we liever met UTC in de DB,
     maar deze functie blijft voor debug.)
    """
    cet = timezone(timedelta(hours=1))  # CET is UTC+1
    return int(datetime.now(cet).timestamp() * 1000)

load_dotenv()

st.write(f"🔍 **Database pad:** {DB_FILE}")
st.write(f"🔍 **Huidige werkdirectory:** {os.getcwd()}")
st.write(f"🔍 **sys.path:** {sys.path}")

logger.info("DatabaseManager succesvol geïmporteerd! (debug)")
logger.info(f"CWD: {os.getcwd()}")
logger.info(f"sys.path: {sys.path}")

# =========================================
# 2) YAML-config laden en pairs selecteren
# =========================================
@st.cache_resource
def load_full_config(config_path: str):
    """
    Probeer de YAML config te laden.
    Geeft een lege dict terug bij error of als het bestand niet gevonden wordt.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or {}
        return config
    except FileNotFoundError:
        st.warning(f"config.yaml niet gevonden op pad='{config_path}'. Check je pad.")
        return {}
    except Exception as error:
        st.error(f"Fout bij laden config: {error}")
        return {}

# Pas het pad aan naar waar jouw config.yaml werkelijk staat
CONFIG_FILE_PATH = "src/config/config.yaml"
full_config = load_full_config(CONFIG_FILE_PATH)

# Als je in config.yaml secties "bitvavo" en "kraken" hebt:
bitvavo_cfg = full_config.get("bitvavo", {})
kraken_cfg  = full_config.get("kraken", {})

# pairs-lijsten
bitvavo_pairs = bitvavo_cfg.get("pairs", [])  # list van strings
kraken_pairs  = kraken_cfg.get("pairs", [])

# Voor intervals
intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]

# =========================================
# 3) Hulpfuncties voor data
# =========================================
def add_utc_datetime_column(df, ms_col="timestamp"):
    """
    Voeg in df een kolom 'datetime' toe, gebaseerd op ms_col (in milliseconden).
    Zo krijg je een leesbare UTC-tijd (YYYY-MM-DD HH:MM:SS).
    """
    if ms_col in df.columns:
        df["datetime"] = df[ms_col].apply(
            lambda x: datetime.fromtimestamp(x / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notnull(x) else None
        )
    return df

def interval_to_ms(interval_str):
    """
    Converteer een interval-string (bv. "4h", "15m", "1d") naar milliseconden.
    """
    if interval_str.endswith("m"):
        return int(interval_str[:-1]) * 60 * 1000
    elif interval_str.endswith("h"):
        return int(interval_str[:-1]) * 60 * 60 * 1000
    elif interval_str.endswith("d"):
        return int(interval_str[:-1]) * 24 * 60 * 60 * 1000
    else:
        return 0

def filter_closed_candles(df, interval_str):
    """
    Filter de DataFrame zodat alleen candles overblijven waarvan de afsluiting
    (starttijd + interval) in het verleden ligt (dus volledig afgesloten).
    """
    duration_ms = interval_to_ms(interval_str)
    current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df_closed = df[df.index.to_series().apply(lambda t: (t.timestamp() * 1000) + duration_ms <= current_time_ms)]
    return df_closed

def localize_to_amsterdam(ts_series):
    """
    Als ts_series tz-naive is, localize to UTC en convert naar Europe/Amsterdam.
    Als ts_series al tz-aware is, wordt alleen tz_convert toegepast.
    """
    if ts_series.dt.tz is None:
        return ts_series.dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam")
    else:
        return ts_series.dt.tz_convert("Europe/Amsterdam")

@st.cache_data
def fetch_tables():
    """Haal alle tabellen uit de DB."""
    try:
        conn = sqlite3.connect(DB_FILE)
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(query, conn)
        conn.close()
        return tables
    except Exception as e:
        logger.error(f"Fout bij ophalen tabellen: {e}")
        return pd.DataFrame()

def show_tables():
    st.header("📋 Tabellen in de Database")
    tables = fetch_tables()
    if not tables.empty:
        st.write("Gevonden tabellen:", tables)
    else:
        st.warning("⚠️ Geen tabellen gevonden in de database.")

@st.cache_data
def fetch_data_cached(table_name, market=None, interval=None, limit=100, exchange=None):
    """
    Haal data op via db_manager.fetch_data(...), voeg een extra 'datetime' (UTC) kolom toe.
    Filter (optioneel) op 'exchange' (Bitvavo / Kraken).
    """
    logger.info(f"Ophalen data: Tabel={table_name}, Markt={market}, Interval={interval}, Limit={limit}, Exchange={exchange}")
    try:
        df = db_manager.fetch_data(
            table_name=table_name,
            limit=limit,
            market=market,
            interval=interval,
            exchange=exchange
        )
        if df is not None and not df.empty:
            df = add_utc_datetime_column(df, ms_col="timestamp")
            # Zet 'timestamp' als datetimeindex (UTC)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            df["timestamp"] = localize_to_amsterdam(df["timestamp"])
            df.set_index("timestamp", inplace=True, drop=True)
            # Optioneel: als het om candle-data gaat, filter op volledig afgesloten candles
            if table_name.lower() == "candles" and interval is not None:
                df = filter_closed_candles(df, interval)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Fout bij ophalen {table_name}: {e}")
        st.error(f"Databasefout: {e}")
        return pd.DataFrame()

def fetch_and_calculate_indicators(market="XRP-EUR", interval="1m", limit=100, exchange=None) -> pd.DataFrame:
    """
    Haal candles op via Market(...) en bereken indicatoren met IndicatorAnalysis.
    Zet 'timestamp' als datetimeindex in Europe/Amsterdam.
    """
    try:
        my_market = Market(symbol=market, db_manager=db_manager)
        df_candles = my_market.fetch_candles(interval, limit=limit)
        if df_candles.empty:
            return pd.DataFrame()

        df_indic = IndicatorAnalysis.calculate_indicators(df_candles)
        if df_indic.empty:
            return pd.DataFrame()

        df_indic["timestamp"] = pd.to_datetime(df_indic["timestamp"], unit="ms", errors="coerce")
        df_indic["timestamp"] = localize_to_amsterdam(df_indic["timestamp"])
        df_indic.set_index("timestamp", inplace=True, drop=True)
        return df_indic
    except Exception as err:
        logger.error(f"Fout in fetch_and_calculate_indicators: {err}")
        st.error(f"Fout in fetch_and_calculate_indicators: {err}")
        return pd.DataFrame()

# =========================================
# 4) Streamlit layout
# =========================================

st.subheader("Kies Exchange + Pairs")

# 1) Exchange-selectie (Bitvavo, Kraken, etc.)
exchange_list = ["Bitvavo", "Kraken"]
selected_exchange = st.selectbox("Selecteer Exchange", exchange_list, index=0)

# 2) Dynamische pairs-lijst
if selected_exchange == "Kraken":
    markets = kraken_pairs
else:
    markets = bitvavo_pairs

st.write(f"Je hebt nu exchange={selected_exchange} gekozen.")
st.write(f"Beschikbare markten: {len(markets)} stuks.")

def pivot_calculation(df):
    """
    Berekent de pivot, R1 en S1 op basis van de hoogste, laagste en laatste closing-prijs.
    """
    if df.empty:
        return None
    high_ = df['high'].max()
    low_  = df['low'].min()
    close_ = df['close'].iloc[-1]
    pivot = (high_ + low_ + close_) / 3
    r1 = 2 * pivot - low_
    s1 = 2 * pivot - high_
    return {"pivot": pivot, "R1": r1, "S1": s1}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Tabellen",
    "📊 Candle Data",
    "📈 Ticker Data",
    "📚 Orderbook Data",
    "📉 Indicatoren",
    "💱 Trades"
])

# =========== TAB 1: SHOW TABLES ===========
with tab1:
    show_tables()

# =========== TAB 2: Candle Data & Weerstandslijnen + Trades ===========
with tab2:
    st.header("📊 Candle Data & Weerstandslijnen + Trades")
    if not markets:
        st.warning("Geen markten in config.yaml? Check je 'bitvavo.pairs' of 'kraken.pairs'.")
        st.stop()

    selected_market = st.selectbox("Selecteer Markt (Candles)", markets, index=0)
    selected_interval = st.selectbox("Selecteer Interval (Candles)", intervals, index=0)
    record_limit = st.slider("Aantal Records (Candles)", min_value=10, max_value=500, step=10, value=100)
    chart_type = st.selectbox("Grafiektype", ["Candles", "Line"], index=0)

    df_candles_raw = fetch_data_cached(
        table_name="candles",
        market=selected_market,
        interval=selected_interval,
        limit=record_limit,
        exchange=selected_exchange
    )
    if not df_candles_raw.empty:
        df_candles_raw = df_candles_raw.sort_index()
        df_indic = IndicatorAnalysis.calculate_indicators(df_candles_raw.reset_index())
        # Zet timestamp opnieuw als datetimeindex en converteer naar Europe/Amsterdam
        df_indic["timestamp"] = pd.to_datetime(df_indic["timestamp"], unit="ms", errors="coerce")
        df_indic["timestamp"] = localize_to_amsterdam(df_indic["timestamp"])
        df_indic.set_index("timestamp", inplace=True, drop=True)
    else:
        st.warning("⚠️ Geen candle-data gevonden.")
        df_indic = pd.DataFrame()

    if df_indic.empty:
        st.stop()
    else:
        st.write("📅 **Candles + indicatoren (head):**", df_indic.head())

        # Pivots: haal daily, 4h en 1h candles op voor pivot berekening
        daily = fetch_data_cached("candles", market=selected_market, interval="1d", limit=50, exchange=selected_exchange)
        pivot_daily = pivot_calculation(daily)
        h4 = fetch_data_cached("candles", market=selected_market, interval="4h", limit=50, exchange=selected_exchange)
        pivot_4h = pivot_calculation(h4)
        h1 = fetch_data_cached("candles", market=selected_market, interval="1h", limit=50, exchange=selected_exchange)
        pivot_1h = pivot_calculation(h1)

        pivot_lines = []
        if pivot_daily: pivot_lines.append({"info": pivot_daily, "label": "1d"})
        if pivot_4h: pivot_lines.append({"info": pivot_4h, "label": "4h"})
        if pivot_1h: pivot_lines.append({"info": pivot_1h, "label": "1h"})
        df_trades = fetch_data_cached("trades", market=selected_market, limit=200, exchange=selected_exchange)
        if not df_trades.empty:
            df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], unit="ms", errors="coerce")
            df_trades["timestamp"] = df_trades["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam")
            buys = df_trades[df_trades["side"].str.upper() == "BUY"]
            sells = df_trades[df_trades["side"].str.upper() == "SELL"]
            st.write("**[DEBUG] Candles time range:**", df_indic.index.min(), "->", df_indic.index.max())
            st.write("**[DEBUG] Trades time range:**", df_trades["timestamp"].min(), "->", df_trades["timestamp"].max())
        else:
            buys = pd.DataFrame()
            sells = pd.DataFrame()

        # Plotting met Plotly
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.02, row_heights=[0.5, 0.2, 0.3],
            specs=[
                [{"type": "scatter"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}]
            ]
        )

        # Chart type: Candlestick of line-grafiek
        if chart_type == "Candles":
            fig.add_trace(go.Candlestick(
                x=df_indic.index,
                open=df_indic['open'],
                high=df_indic['high'],
                low=df_indic['low'],
                close=df_indic['close'],
                name='Candlesticks',
                increasing=dict(line=dict(color='green')),
                decreasing=dict(line=dict(color='red'))
            ), row=1, col=1)
        else:
            # Lijngrafiek van de closing-prijs
            fig.add_trace(go.Scatter(
                x=df_indic.index,
                y=df_indic['close'],
                mode='lines',
                line=dict(color='green', width=2),
                name='Line'
            ), row=1, col=1)

        # Voeg Bollinger Bands en MA toe
        if "bollinger_upper" in df_indic.columns:
            fig.add_trace(go.Scatter(
                x=df_indic.index,
                y=df_indic["bollinger_upper"],
                line=dict(color='blue', width=1),
                name='Boll.Upper'
            ), row=1, col=1)
        if "bollinger_lower" in df_indic.columns:
            fig.add_trace(go.Scatter(
                x=df_indic.index,
                y=df_indic["bollinger_lower"],
                line=dict(color='blue', width=1),
                name='Boll.Lower'
            ), row=1, col=1)
        if "moving_average" in df_indic.columns:
            fig.add_trace(go.Scatter(
                x=df_indic.index,
                y=df_indic["moving_average"],
                line=dict(color='orange', width=1),
                name='MA'
            ), row=1, col=1)

        # RSI op tweede rij
        if "rsi" in df_indic.columns:
            fig.add_trace(go.Scatter(
                x=df_indic.index,
                y=df_indic["rsi"],
                line=dict(color='purple', width=1),
                name='RSI'
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD op derde rij
        if "macd" in df_indic.columns:
            fig.add_trace(go.Scatter(
                x=df_indic.index,
                y=df_indic["macd"],
                line=dict(color='cyan', width=1),
                name='MACD'
            ), row=3, col=1)
        if "macd_signal" in df_indic.columns:
            fig.add_trace(go.Scatter(
                x=df_indic.index,
                y=df_indic["macd_signal"],
                line=dict(color='magenta', width=1),
                name='MACD Sig'
            ), row=3, col=1)

        # Voeg Pivot-lijnen toe (inclusief R1 en S1)
        for piv in pivot_lines:
            label = piv["label"]
            dat = piv["info"]
            pivot = dat["pivot"]
            r1 = dat["R1"]
            s1 = dat["S1"]
            fig.add_hline(y=pivot, line_dash="dash", line_color="yellow",
                          annotation_text=f"{label} Pivot", annotation_position="top right",
                          row=1, col=1)
            fig.add_hline(y=r1, line_dash="dash", line_color="red",
                          annotation_text=f"{label} R1", annotation_position="top right",
                          row=1, col=1)
            fig.add_hline(y=s1, line_dash="dash", line_color="green",
                          annotation_text=f"{label} S1", annotation_position="top right",
                          row=1, col=1)

        # Trades toevoegen als markers
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["price"],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=12),
                    name='BUY'
                ), row=1, col=1
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["price"],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=12),
                    name='SELL'
                ), row=1, col=1
            )

        fig.update_layout(
            title=f'Candlestick + Weerstand + Trades [{selected_exchange}] {selected_market} ({selected_interval})',
            yaxis_title='Prijs',
            xaxis_title='Tijd',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=700,
            width=900,
            margin=dict(l=40, r=40, t=60, b=30)
        )
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)

        st.plotly_chart(fig, use_container_width=False)
        st.success("✅ Candlestick + Weerstand + Trades weergegeven.")

# =========== TAB 3: Ticker Data ===========
with tab3:
    st.header("📈 Ticker Data")
    if not markets:
        st.warning("Geen markten geladen (check config.yaml).")
        st.stop()
    selected_market_ticker = st.selectbox("Selecteer Markt (Ticker)", markets, index=0)
    df_ticker = fetch_data_cached("ticker",
                                  market=selected_market_ticker,
                                  limit=50,
                                  exchange=selected_exchange)
    if not df_ticker.empty:
        st.write("**Ticker Data (head)**", df_ticker.head())
    else:
        st.warning(f"⚠️ Geen ticker data voor {selected_market_ticker} / {selected_exchange}.")

# =========== TAB 4: Orderbook Data ===========
with tab4:
    st.header("📚 Orderbook Data")
    if not markets:
        st.warning("Geen markten geladen (check config.yaml).")
        st.stop()
    selected_market_ob = st.selectbox("Selecteer Markt (Orderbook)", markets, index=0)
    df_bids = fetch_data_cached("orderbook_bids",
                                market=selected_market_ob,
                                limit=50,
                                exchange=selected_exchange)
    df_asks = fetch_data_cached("orderbook_asks",
                                market=selected_market_ob,
                                limit=50,
                                exchange=selected_exchange)
    st.subheader("Bids:")
    if not df_bids.empty:
        st.write(df_bids.head())
    else:
        st.warning("Geen Bids data.")
    st.subheader("Asks:")
    if not df_asks.empty:
        st.write(df_asks.head())
    else:
        st.warning("Geen Asks data.")

# =========== TAB 5: Indicatoren (losse tab) ===========
with tab5:
    st.header("📉 Indicatoren (losse tab)")
    if not markets:
        st.warning("Geen markten geladen (check config.yaml).")
        st.stop()
    sel_market_5 = st.selectbox("Markt (Indicators)", markets, index=0)
    sel_interval_5 = st.selectbox("Interval (Indicators)", intervals, index=0)
    sel_limit_5 = st.slider("Records (Indicators):", 10, 500, 100)
    df_5 = fetch_data_cached("candles",
                             market=sel_market_5,
                             interval=sel_interval_5,
                             limit=sel_limit_5,
                             exchange=selected_exchange)
    if not df_5.empty:
        df_5 = df_5.sort_index()
        df_5 = IndicatorAnalysis.calculate_indicators(df_5.reset_index())
        df_5["timestamp"] = pd.to_datetime(df_5["timestamp"], unit="ms", errors="coerce")
        df_5["timestamp"] = localize_to_amsterdam(df_5["timestamp"])
        df_5.set_index("timestamp", inplace=True, drop=True)
        st.write("**Indicatoren (head)**", df_5.head())
        if "close" in df_5.columns:
            st.line_chart(df_5["close"], use_container_width=True)
        if {"macd", "macd_signal"} <= set(df_5.columns):
            st.line_chart(df_5[["macd", "macd_signal"]], use_container_width=True)
    else:
        st.warning(f"⚠️ Geen candles => geen indicators voor {sel_market_5} / {selected_exchange}.")

# =========== TAB 6: Trades ===========
with tab6:
    st.header("💱 Trades (overzicht)")
    if not markets:
        st.warning("Geen markten geladen (check config.yaml).")
        st.stop()
    selected_market_trades = st.selectbox("Selecteer Markt (Trades)", markets, index=0)
    df_trades_tab6 = fetch_data_cached("trades",
                                       market=selected_market_trades,
                                       limit=100,
                                       exchange=selected_exchange)
    if not df_trades_tab6.empty:
        df_trades_tab6["trade_cost"] = pd.to_numeric(df_trades_tab6["trade_cost"], errors="coerce").fillna(0.0)
        df_trades_tab6["pnl_eur"] = pd.to_numeric(df_trades_tab6["pnl_eur"], errors="coerce").fillna(0.0)
        df_trades_tab6["timestamp"] = pd.to_datetime(df_trades_tab6["timestamp"], unit="ms", errors="coerce")
        df_trades_tab6["timestamp"] = df_trades_tab6["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam")
        df_trades_tab6["trade_cost"] = df_trades_tab6["trade_cost"].round(2)
        df_trades_tab6["pnl_eur"] = df_trades_tab6["pnl_eur"].round(2)
        st.dataframe(
            df_trades_tab6.style.format({
                "trade_cost": "{:.2f}",
                "pnl_eur": "{:.2f}"
            })
        )
    else:
        st.warning(f"⚠️ Geen trades gevonden voor {selected_market_trades} / {selected_exchange}.")
