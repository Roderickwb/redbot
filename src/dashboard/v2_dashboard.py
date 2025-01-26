
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

# Imports van andere modules
from src.config.config import DB_FILE, DASHBOARD_LOG_FILE, PAIRS_CONFIG
from src.logger.logger import setup_logger
from src.indicator_analysis.indicators import IndicatorAnalysis, Market
from src.main import db_manager

# =========================================
# 1) Streamlit basisconfig + log
# =========================================
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# PAS AAN: nu zonder 10m
markets = PAIRS_CONFIG
intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]

logger = setup_logger('dashboard', DASHBOARD_LOG_FILE)
logger.info("Dashboard gestart.")


def get_current_local_timestamp():
    """
    CET=UTC+1 in milliseconden.
    (LET OP: eigenlijk werken we liever met UTC in de DB,
     maar we laten deze functie hier nog staan voor debug.)
    """
    cet = timezone(timedelta(hours=1))  # CET is UTC+1
    return int(datetime.now(cet).timestamp() * 1000)


load_dotenv()


# db_manager.create_tables()  # (optioneel)


st.write(f"🔍 **Database pad:** {DB_FILE}")
st.write(f"🔍 **Huidige werkdirectory:** {os.getcwd()}")
st.write(f"🔍 **sys.path:** {sys.path}")

logger.info("DatabaseManager succesvol geïmporteerd! (debug)")
logger.info(f"CWD: {os.getcwd()}")
logger.info(f"sys.path: {sys.path}")


# =========================================
# 2) Hulpfuncties
# =========================================

def add_utc_datetime_column(df, ms_col="timestamp"):
    """
    Voeg in df een kolom 'datetime' toe, gebaseerd op ms_col (in milliseconden).
    Zo zie je direct een leesbare UTC-tijd (YYYY-MM-DD HH:MM:SS).
    """
    if ms_col in df.columns:
        df["datetime"] = df[ms_col].apply(
            lambda x: datetime.fromtimestamp(x / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notnull(x) else None
        )
    return df

@st.cache_resource
def load_config(config_path='config.yaml'):
    """Laad YAML-config"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as error:
        st.error(f"Fout bij laden config: {error}")
        return {}


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
def fetch_data_cached(table_name, market=None, interval=None, limit=100):
    """
    Haal data op via db_manager.fetch_data(...), en voeg
    een extra kolom 'datetime' (UTC) toe als er een 'timestamp' in de data staat.
    """
    logger.info(f"Ophalen data: Tabel={table_name}, Markt={market}, Interval={interval}, Limit={limit}")
    try:
        df = db_manager.fetch_data(table_name, limit=limit, market=market, interval=interval)
        if df is not None and not df.empty:
            logger.info(f"Data succesvol opgehaald (head):\n{df.head()}")
            # === Punt (3): maak extra 'datetime' kolom in UTC ===
            df = add_utc_datetime_column(df, ms_col="timestamp")
            return df
        else:
            st.warning(f"⚠️ Geen data beschikbaar voor {table_name} (market={market}, interval={interval}).")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Fout bij ophalen {table_name}: {e}")
        st.error(f"Databasefout: {e}")
        return pd.DataFrame()


def fetch_and_calculate_indicators(market="XRP-EUR", interval="1m", limit=100) -> pd.DataFrame:
    """
    Haalt candles op via Market(...) en berekent indicatoren met IndicatorAnalysis.
    Zet 'timestamp' als datetimeindex in Europe/Amsterdam.
    """
    try:
        my_market = Market(symbol=market, db_manager=db_manager)
        df_candles = my_market.fetch_candles(interval, limit=limit)
        if df_candles.empty:
            st.warning(f"⚠️ Geen candles voor {market} ({interval}).")
            return pd.DataFrame()

        df_indic = IndicatorAnalysis.calculate_indicators(df_candles)
        if df_indic.empty:
            return pd.DataFrame()

        # timestamp -> datetime (AMS)
        df_indic["timestamp"] = pd.to_datetime(df_indic["timestamp"], unit="ms", errors="coerce")
        df_indic["timestamp"] = df_indic["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam")

        df_indic.set_index("timestamp", inplace=True, drop=True)
        return df_indic
    except Exception as err:
        logger.error(f"Fout in fetch_and_calculate_indicators: {err}")
        st.error(f"Fout in fetch_and_calculate_indicators: {err}")
        return pd.DataFrame()


# =========================================
# 3) Streamlit layout
# =========================================
st.subheader("Indicatoren voor XRP-EUR")
df = fetch_and_calculate_indicators("XRP-EUR", "1m", 100)
if not df.empty:
    st.write("Laatste indicatoren (tail):", df.tail())
else:
    st.warning("Geen data beschikbaar voor indicatoren (XRP,1m).")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Tabellen",
    "📊 Candle Data",
    "📈 Ticker Data",
    "📚 Orderbook Data",
    "📉 Indicatoren",
    "💱 Trades"  # nieuw
])

# =========== TAB 1: SHOW TABLES ===========
with tab1:
    show_tables()

# =========== TAB 2: Candle + Indicators + Weerstandslijnen + Trades ===========
with tab2:
    st.header("📊 Candle Data & Weerstandslijnen + Trades")
    selected_market = st.selectbox("Selecteer Markt (Candles)", markets, index=0)
    selected_interval = st.selectbox("Selecteer Interval (Candles)", intervals, index=0)
    record_limit = st.slider("Aantal Records (Candles)", min_value=10, max_value=500, step=10, value=100)
    chart_type = st.selectbox("Grafiektype", ["Candles", "Line"], index=0)

    df_candles = fetch_and_calculate_indicators(selected_market, selected_interval, record_limit)
    if df_candles.empty:
        st.warning("⚠️ Geen data in tab2.")
        st.stop()
    else:
        st.write("📅 **Candles + indicatoren (head):**", df_candles.head())

        pivot_lines = []

        def fetch_pivot(market, interval):
            df_ = fetch_and_calculate_indicators(market, interval, 50)
            if df_.empty:
                return None
            high_ = df_['high'].max()
            low_ = df_['low'].min()
            close_ = df_['close'].iloc[-1]
            pivot = (high_ + low_ + close_) / 3
            r1 = 2 * pivot - low_
            s1 = 2 * pivot - high_
            return {"pivot": pivot, "R1": r1, "S1": s1, "interval": interval}

        daily_pivot = fetch_pivot(selected_market, "1d")
        if daily_pivot: pivot_lines.append(daily_pivot)
        h4_pivot = fetch_pivot(selected_market, "4h")
        if h4_pivot: pivot_lines.append(h4_pivot)
        h1_pivot = fetch_pivot(selected_market, "1h")
        if h1_pivot: pivot_lines.append(h1_pivot)

        # --- Stap B: Trades (Buy/Sell) ophalen
        df_trades = fetch_data_cached("trades", market=selected_market, interval=None, limit=200)
        if not df_trades.empty:
            df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], unit="ms", errors="coerce")
            df_trades["timestamp"] = df_trades["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam")

            buys = df_trades[df_trades["side"].str.upper() == "BUY"]
            sells = df_trades[df_trades["side"].str.upper() == "SELL"]

            st.write("**[DEBUG] Candles time range:**", df_candles.index.min(), "->", df_candles.index.max())
            st.write("**[DEBUG] Trades time range:**", df_trades["timestamp"].min(), "->", df_trades["timestamp"].max())
        else:
            buys = pd.DataFrame()
            sells = pd.DataFrame()

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.02, row_heights=[0.5, 0.2, 0.3],
            specs=[
                [{"type": "scatter"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}]
            ]
        )

        # === Afhankelijk van de keuze: Candlestick of Lijn ===
        if chart_type == "Candles":
            fig.add_trace(go.Candlestick(
                x=df_candles.index,
                open=df_candles['open'],
                high=df_candles['high'],
                low=df_candles['low'],
                close=df_candles['close'],
                name='Candlesticks',
                increasing=dict(line=dict(color='green')),
                decreasing=dict(line=dict(color='red'))
            ), row=1, col=1)
        else:
            # Lijngrafiek van de closing-prijs
            fig.add_trace(go.Scatter(
                x=df_candles.index,
                y=df_candles['close'],
                mode='lines',
                line=dict(color='green', width=2),
                name='Line Chart'
            ), row=1, col=1)

        # Bollinger
        fig.add_trace(go.Scatter(
            x=df_candles.index,
            y=df_candles.get('bollinger_upper', None),
            line=dict(color='blue', width=1),
            name='Boll.Upper',
            opacity=0.5
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_candles.index,
            y=df_candles.get('bollinger_lower', None),
            line=dict(color='blue', width=1),
            name='Boll.Lower',
            opacity=0.5
        ), row=1, col=1)
        # Moving average
        if "moving_average" in df_candles.columns:
            fig.add_trace(go.Scatter(
                x=df_candles.index,
                y=df_candles["moving_average"],
                line=dict(color='orange', width=1),
                name='MA'
            ), row=1, col=1)

        # RSI
        if "rsi" in df_candles.columns:
            fig.add_trace(go.Scatter(
                x=df_candles.index,
                y=df_candles["rsi"],
                line=dict(color='purple', width=1),
                name='RSI'
            ), row=2, col=1)
            # horizontale 70/30
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        if "macd" in df_candles.columns:
            fig.add_trace(go.Scatter(
                x=df_candles.index,
                y=df_candles["macd"],
                line=dict(color='cyan', width=1),
                name='MACD'
            ), row=3, col=1)
        if "macd_signal" in df_candles.columns:
            fig.add_trace(go.Scatter(
                x=df_candles.index,
                y=df_candles["macd_signal"],
                line=dict(color='magenta', width=1),
                name='MACD Sig'
            ), row=3, col=1)

        # --- (6) Weerstandslijnen ---
        # daily/4h/1h pivot-lijnen
        for pivot_info in pivot_lines:
            # bv: pivot_info={"pivot":1.23, "R1":1.30, "S1":1.20, "interval":"4h"}
            pivot_y = pivot_info["pivot"]
            r1_y = pivot_info["R1"]
            s1_y = pivot_info["S1"]
            ival = pivot_info["interval"]

            fig.add_hline(
                y=pivot_y,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"{ival} Pivot",
                annotation_position="top right",
                row=1, col=1
            )
            fig.add_hline(
                y=r1_y,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{ival} R1",
                annotation_position="top right",
                row=1, col=1
            )
            fig.add_hline(
                y=s1_y,
                line_dash="dash",
                line_color="green",
                annotation_text=f"{ival} S1",
                annotation_position="top right",
                row=1, col=1
            )

        # Trades markers
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

        # Plot layout
        fig.update_layout(
            title=f'Candlestick + Weerstand + Trades voor {selected_market} ({selected_interval})',
            yaxis_title='Prijs',
            xaxis_title='Tijd',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=700,
            width=900,
            margin=dict(l=40, r=40, t=60, b=30)
        )

        # RSI en MACD y-assen
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)

        st.plotly_chart(fig, use_container_width=False)
        st.success("✅ Candlestick + Weerstand + Trades weergegeven.")

# =========== TAB 3: Ticker Data ===========
with tab3:
    st.header("📈 Ticker Data")
    selected_market_ticker = st.selectbox("Selecteer Markt voor Ticker Data", markets, index=0)
    df_ticker = fetch_data_cached("ticker", market=selected_market_ticker, limit=50)
    if not df_ticker.empty:
        st.write("📅 **Ticker Data (head)**", df_ticker.head())
    else:
        st.warning(f"⚠️ Geen ticker data voor {selected_market_ticker}.")

# =========== TAB 4: Orderbook Data ===========
with tab4:
    st.header("📚 Orderbook Data")
    selected_market_ob = st.selectbox("Selecteer Markt voor Orderbook Data", markets, index=0)
    df_bids = fetch_data_cached("orderbook_bids", market=selected_market_ob, limit=50)
    df_asks = fetch_data_cached("orderbook_asks", market=selected_market_ob, limit=50)

    if not df_bids.empty:
        st.write("**Orderbook Bids (head)**", df_bids.head())
    else:
        st.warning("Geen Bids data.")

    if not df_asks.empty:
        st.write("**Orderbook Asks (head)**", df_asks.head())
    else:
        st.warning("Geen Asks data.")

# =========== TAB 5: Indicatoren (losse tab) ===========
with tab5:
    st.header("📉 Indicatoren (losse tab)")
    sel_market_5 = st.selectbox("Markt:", markets, index=0)
    sel_interval_5 = st.selectbox("Interval:", intervals, index=0)
    sel_limit_5 = st.slider("Records:", 10, 500, 100)

    df_5 = fetch_and_calculate_indicators(sel_market_5, sel_interval_5, sel_limit_5)
    if df_5.empty:
        st.warning("⚠️ Indicatoren tabel is leeg in tab5.")
    else:
        st.success("✅ Indicatoren data in tab5.")
        st.write("**Voorbeeld (head):**", df_5.head())

        if "close" in df_5.columns:
            st.line_chart(df_5["close"], use_container_width=True)
        if {"macd", "macd_signal"} <= set(df_5.columns):
            st.line_chart(df_5[["macd", "macd_signal"]], use_container_width=True)

# =========== TAB 6: Trades ===========
with tab6:
    st.header("💱 Trades (overzicht)")
    selected_market_trades = st.selectbox("Selecteer Markt voor Trades", markets, index=0)
    df_trades_tab6 = fetch_data_cached("trades", market=selected_market_trades, limit=100)

    if not df_trades_tab6.empty:
        df_trades_tab6["trade_cost"] = pd.to_numeric(df_trades_tab6["trade_cost"], errors="coerce").fillna(0.0)
        df_trades_tab6["pnl_eur"] = pd.to_numeric(df_trades_tab6["pnl_eur"], errors="coerce").fillna(0.0)

        # Timestamps -> Europe/Amsterdam
        df_trades_tab6["timestamp"] = pd.to_datetime(
            df_trades_tab6["timestamp"], unit="ms", errors="coerce"
        ).dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam")

        df_trades_tab6["trade_cost"] = df_trades_tab6["trade_cost"].round(2)
        df_trades_tab6["pnl_eur"] = df_trades_tab6["pnl_eur"].round(2)

        st.dataframe(
            df_trades_tab6.style.format({
                "trade_cost": "{:.2f}",
                "pnl_eur": "{:.2f}"
            })
        )
    else:
        st.warning("Geen trades gevonden!")
