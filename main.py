# main.py

import logging
from dotenv import load_dotenv
import os
from logger.logger import setup_logger  # Aangepaste import pad
from database_manager import DatabaseManager  # Aangepaste import pad
from my_websocket.client import WebSocketClient  # Aangepaste import pad
from indicator_analysis.indicator_analysis import IndicatorAnalysis  # Aangepaste import pad
from strategy.strategy import Strategy  # Aangepaste import pad
from ML.ML import MLEngine  # Aangepaste import pad (afhankelijk van je ML.py implementatie)
from trading_engine.executor import Executor  # Aangepaste import pad (afhankelijk van je executor.py implementatie)
import time
from threading import Thread


def main():
    # Laad .env bestand voor configuratie
    load_dotenv()

    # Voeg de naam en het logbestand toe
    setup_logger(name="main", log_file="main.log")
    logger = logging.getLogger(__name__)

    ws_client = None  # Initialiseer ws_client als None

    try:
        # Initialiseer database manager
        db_manager = DatabaseManager(db_path=os.getenv("DB_PATH", "market_data.db"))

        # Maak de benodigde tabellen aan
        db_manager.create_tables()

        # Controleer het aantal records in elke tabel
        try:
            candles_count = db_manager.get_table_count("candles")
            ticker_count = db_manager.get_table_count("ticker")
            bids_count = db_manager.get_table_count("orderbook_bids")
            asks_count = db_manager.get_table_count("orderbook_asks")
            logging.info(f"Data beschikbaar - Candles: {candles_count}, Ticker: {ticker_count}, Orderbook Bids: {bids_count}, Orderbook Asks: {asks_count}")
        except Exception as e:
            logging.error(f"Fout bij controleren van data beschikbaarheid: {e}")

        # WebSocket client voor real-time data
        ws_client = WebSocketClient(
            ws_url=os.getenv("WS_URL"),  # Haal de WebSocket URL uit .env
            db_manager=db_manager,        # Geef de DatabaseManager door
            api_key=os.getenv("API_KEY"),  # Haal de API-sleutel uit .env
            api_secret=os.getenv("API_SECRET")  # Haal de API-secret uit .env
        )

        # Start WebSocket client in een aparte thread
        ws_thread = Thread(target=ws_client.start_websocket, daemon=True)
        ws_thread.start()

        # Indicator analyses
        indicators = IndicatorAnalysis(db_manager)

        # Handelsstrategie
        strategy = Strategy(indicators, db_manager)


        # Hoofdloop van de bot
        while True:
            # Hier kan de logica voor trading en strategie implementatie komen
            # Bijvoorbeeld:
            # - Strategie uitvoeren
            # - Risico management toepassen
            # - Loggen van beslissingen, trades, fouten
            time.sleep(5)  # Voeg een slaap toe zodat de loop niet continu draait

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Exception details")
    except KeyboardInterrupt:
        # Dit zorgt ervoor dat de bot stopt bij een Ctrl+C (handmatige stop)
        logger.info("Bot is handmatig gestopt.")
    finally:
        if ws_client:
            ws_client.stop_websocket()  # Stop de WebSocket client
        logger.info("Bot has been shut down.")

if __name__ == "__main__":
    main()

# Test commit




