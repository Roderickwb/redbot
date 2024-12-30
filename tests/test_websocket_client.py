from src.my_websocket.client import WebSocketClient
import logging

logging.basicConfig(level=logging.INFO)

def test_websocket_connection():
    # Maak een instance van de WebSocket client
    ws_client = WebSocketClient(ws_url="wss://ws.bitvavo.com/v2/", db_path="../data/market_data.db", api_key="your_api_key", api_secret="your_api_secret")

    # Start WebSocket
    try:
        ws_client.start_websocket()
    except Exception as e:
        logging.error(f"Fout bij het verbinden met WebSocket: {e}")

if __name__ == "__main__":
    test_websocket_connection()
