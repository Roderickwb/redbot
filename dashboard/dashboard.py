import sqlite3
import pandas as pd
import streamlit as st
from threading import Thread
from websocket_module import start_websocket  # Zorg ervoor dat je WebSocket-code hierin staat

# Titel en layout
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

DB_FILE = "market_data.db"

@st.cache_data
def fetch_data(table_name, limit=50):
    """Haal data op uit een opgegeven tabel."""
    conn = sqlite3.connect(DB_FILE)
    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Extra logging om te controleren of de juiste gegevens zijn opgehaald
    logging.info(f"Gegevens opgehaald voor {table_name}: {df.head()}")  # Log de eerste paar rijen
    return df


# WebSocket starten als deze nog niet draait
if "websocket_thread" not in st.session_state:
    st.session_state["websocket_thread"] = Thread(target=start_websocket, daemon=True)
    st.session_state["websocket_thread"].start()
    st.sidebar.success("WebSocket gestart voor live gegevens.")

# Tabs voor verschillende datasets
tab1, tab2, tab3 = st.tabs(["📊 Candle Data", "📈 Ticker Data", "📚 Orderbook Data"])

# Candle Data tab
with tab1:
    st.header("Candle Data")
    try:
        df_candles = fetch_data("candles")
        if not df_candles.empty:
            df_candles['timestamp'] = pd.to_datetime(df_candles['timestamp'], unit='ms')
            st.write(df_candles)
        else:
            st.warning("Geen candle data beschikbaar.")
    except Exception as e:
        st.error(f"Fout bij het ophalen van candle data: {e}")

# Ticker Data tab
with tab2:
    st.header("Ticker Data")
    try:
        df_ticker = fetch_data("ticker")
        if not df_ticker.empty:
            df_ticker['timestamp'] = pd.to_datetime(df_ticker['timestamp'], unit='ms')
            st.write(df_ticker)
        else:
            st.warning("Geen ticker data beschikbaar.")
    except Exception as e:
        st.error(f"Fout bij het ophalen van ticker data: {e}")

# Orderbook Data tab
with tab3:
    st.header("Orderbook Data")
    try:
        df_orderbook = fetch_data("orderbook")
        if not df_orderbook.empty:
            df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['timestamp'], unit='ms')
            st.write(df_orderbook)
        else:
            st.warning("Geen orderbook data beschikbaar.")
    except Exception as e:
        st.error(f"Fout bij het ophalen van orderbook data: {e}")



