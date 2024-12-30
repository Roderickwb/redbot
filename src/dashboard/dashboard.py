import os
import sys

# Voeg de project root toe aan sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime, timezone, timedelta
import sqlite3
import pandas as pd
import streamlit as st
import logging
from dotenv import load_dotenv
from logger.logger import setup_logger
import pytz

from database_manager import DatabaseManager
print("DatabaseManager succesvol geïmporteerd!")

def get_current_local_timestamp():
    """Geeft de huidige tijd in CET in milliseconden."""
    cet = timezone(timedelta(hours=1))  # CET is UTC+1
    return int(datetime.now(cet).timestamp() * 1000)  # Tijd in milliseconden

# Laad .env bestand
load_dotenv()

# Bepaal het absolute pad voor de logfile en database
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_FILE = os.path.join(project_root, 'dashboard.log')
DB_PATH = os.path.join(project_root, 'market_data.db')

# Configureer logging via centrale logger
logger = setup_logger('dashboard', LOG_FILE)

# Stel de Streamlit-pagina in
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# Cache de functie om data op te halen, ververs elke 60 seconden
@st.cache_data(ttl=60)  # ttl = tijd in seconden voordat de cache wordt vernieuwd
def fetch_data():
    # Simuleer ophalen van data uit de database
    conn = sqlite3.connect("market_data.db")
    query = "SELECT * FROM candles ORDER BY timestamp DESC LIMIT 100"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Initialiseer logging
logger.info("Dashboard gestart.")

# Gebruik st.write om het pad naar de database te tonen
st.write(f"📁 **Database pad:** {DB_PATH}")

# Log het pad naar de database
logger.info(f"DB_FILE pad: {DB_PATH}")

# Importeer DatabaseManager
try:
    logger.info("DatabaseManager geïmporteerd.")
    st.write("✅ **DatabaseManager geïmporteerd:** `DatabaseManager`")
except ModuleNotFoundError as e:
    st.error(f"❌ Fout bij importeren van `DatabaseManager`: {e}")

# Maak een instantie van DatabaseManager
db_manager = DatabaseManager(db_path=DB_PATH)

# Debugging: log de huidige werkdirectory en sys.path
st.write(f"🔍 **Huidige Werkdirectory (CWD):** {os.getcwd()}")
st.write(f"🛠️ **sys.path:** {sys.path}")
logger.info(f"CWD: {os.getcwd()}")
logger.info(f"sys.path: {sys.path}")

# Roep de functie aan om de candles-tabel aan te maken indien nog niet aanwezig
db_manager.create_tables()

# Functie om tabellen in de database te tonen
@st.cache_data
def fetch_tables():
    """Haal een lijst van alle tabellen in de database op."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(query, conn)
        conn.close()
        return tables
    except Exception as e:
        logger.error(f"Fout bij ophalen van tabellen: {e}")
        return pd.DataFrame()

def show_tables():
    st.header("📋 Tabellen in de Database")
    tables = fetch_tables()
    if not tables.empty:
        st.write("Gevonden tabellen:", tables)
    else:
        st.warning("⚠️ Geen tabellen gevonden in de database.")

# Functie om timestamps te converteren
def convert_timestamp(df):
    """Converteer timestamp naar datetime als het in milliseconden is."""
    if 'timestamp' in df.columns:
        if df['timestamp'].max() > 1e12:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        logger.warning("Kolom 'timestamp' ontbreekt in DataFrame.")
    return df

# Gegevens ophalen met caching
@st.cache_data
def fetch_data_cached(table_name, market=None, interval=None, limit=100):
    """Haal data op uit de opgegeven tabel via DatabaseManager."""
    logger.info(f"Ophalen van data: Tabel={table_name}, Markt={market}, Interval={interval}, Limiet={limit}")
    try:
        # Ophalen van data via DatabaseManager
        df = db_manager.fetch_data(table_name, limit=limit, market=market, interval=interval)
        if df is not None and not df.empty:
            df['timestamp'] = df['timestamp'].apply(
                lambda ts: get_current_local_timestamp() if ts is None or ts <= 0 else ts
            )
            logger.info(f"Data succesvol opgehaald: {df.head()}")
            return df
        else:
            st.warning(f"⚠️ Geen data beschikbaar voor tabel {table_name}.")
            return pd.DataFrame()
    except sqlite3.Error as e:
        logger.error(f"Fout bij ophalen van data uit {table_name}: {e}")
        st.error(f"❌ Databasefout: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Onverwachte fout bij ophalen van data: {e}")
        st.error("❌ Onverwachte fout bij ophalen van data.")
        return pd.DataFrame()

# Tabs voor verschillende datasets
tab1, tab2, tab3 = st.tabs(["📊 Candle Data", "📈 Ticker Data", "📚 Orderbook Data"])

# Candle Data tab
with tab1:
    st.header("📊 Candle Data")
    markets = ["XRP-EUR", "BTC-EUR"]
    intervals = ["1m", "5m", "15m"]

    # Selecties voor gebruiker
    selected_market = st.selectbox("Selecteer Markt", markets)
    selected_interval = st.selectbox("Selecteer Interval", intervals)
    record_limit = st.slider("Aantal Records", min_value=10, max_value=500, step=10, value=100)

    # Ophalen van gegevens
    df_candles = fetch_data_cached("candles", market=selected_market, interval=selected_interval, limit=record_limit)
    if not df_candles.empty:
        st.write("📅 **Candle Data:**", df_candles)
        st.line_chart(df_candles[['timestamp', 'close']].set_index('timestamp'))
    else:
        st.warning(f"⚠️ Geen candle data beschikbaar voor {selected_market} - {selected_interval}.")

# Ticker Data tab
with tab2:
    st.header("📈 Ticker Data")
    selected_market = st.selectbox("Selecteer Markt voor Ticker Data", markets)

    df_ticker = fetch_data_cached("ticker", market=selected_market, limit=50)
    if not df_ticker.empty:
        st.write("📅 **Ticker Data:**", df_ticker)
    else:
        st.warning(f"⚠️ Geen ticker data beschikbaar voor {selected_market}.")

# Orderbook Data tab
with tab3:
    st.header("📚 Orderbook Data")
    selected_market = st.selectbox("Selecteer Markt voor Orderbook Data", markets)

    # Toon Orderbook Bids
    df_orderbook_bids = fetch_data_cached("orderbook_bids", market=selected_market, limit=50)
    if not df_orderbook_bids.empty:
        st.write("📅 **Orderbook Bids Data:**", df_orderbook_bids)
    else:
        st.warning("⚠️ Geen orderbook bids data beschikbaar voor {selected_market}.")

    # Toon Orderbook Asks
    df_orderbook_asks = fetch_data_cached("orderbook_asks", market=selected_market, limit=50)
    if not df_orderbook_asks.empty:
        st.write("📅 **Orderbook Asks Data:**", df_orderbook_asks)
    else:
        st.warning("⚠️ Geen orderbook asks data beschikbaar voor {selected_market}.")