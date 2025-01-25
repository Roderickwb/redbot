# Test of de WebSocket gegevens goed ontvangt en opslaat in de database

import json
from src.my_websocket.client import WebSocketClient
from src.database_manager.database_manager import DatabaseManager


# Test of de WebSocket-gegevens goed ontvangen en opgeslagen worden in de database
def test_websocket_database_integration():
    try:
        # Maak een instantie van de DatabaseManager
        db_manager = DatabaseManager()

        # Maak een instantie van de WebSocketClient
        ws_client = WebSocketClient(
            ws_url="wss://ws.bitvavo.com/v2/",  # Zorg ervoor dat dit correct is
            db_manager=db_manager,
            api_key="YOUR_API_KEY",
            api_secret="YOUR_API_SECRET"
        )

        # Simuleer een ontvangen WebSocket-bericht (bijvoorbeeld een 'candle' bericht)
        ws_data = {
            "event": "candle",
            "market": "XRP-EUR",
            "interval": "1m",
            "candle": [
                [1609459200000, 1.05, 1.07, 1.04, 1.06, 1000]
            ]
        }

        # Gebruik de on_message-methode van WebSocketClient om het bericht te verwerken
        ws_client.on_message(None, json.dumps(ws_data))  # 'None' als websocket instance omdat we geen echte verbinding gebruiken

        # Haal de laatst opgeslagen candles op uit de database
        stored_data = db_manager.fetch_data('candles', limit=10)

        # Controleer of de gegevens correct zijn opgeslagen
        if len(stored_data) > 0:
            print(f"Gegevens succesvol opgeslagen in de database: {stored_data}")
        else:
            print("Er zijn geen gegevens opgeslagen in de database!")

        # Test geslaagd als we hier komen
        return True
    except Exception as e:
        print(f"Fout tijdens WebSocket + Database test: {e}")
        return False

# Voer de test uit
test_result = test_websocket_database_integration()

if test_result:
    print("Stap 1: WebSocket + Database integratie is geslaagd.")
else:
    print("Stap 1: WebSocket + Database integratie is mislukt. Controleer foutmeldingen.")
