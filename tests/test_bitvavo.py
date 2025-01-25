from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv
import os
load_dotenv()


# Jouw API-gegevens
API_KEY = os.getenv("API_KEY")  # Haal de API-sleutel uit de omgevingsvariabelen
API_SECRET = os.getenv("API_SECRET")  # Haal het geheime sleutel uit de omgevingsvariabelen
WS_URL = os.getenv("WS_URL", "wss://ws.bitvavo.com/v2/")  # Gebruik standaardwaarde als WS_URL niet is ingesteld

def main():

    bitvavo = Bitvavo({
        'APIKEY': '...',
        'APISECRET': '...',
        'DEBUGGING': True
    })

    # Check of je time() kunt ophalen (public call). Krijg je NoneType error?
    print("Time:", bitvavo.time())

    # Probeer candles
    c = bitvavo.candles("BTC-EUR", "1m", { "limit": 5 })
    print("Candles:", c)

if __name__ == "__main__":
    main()
