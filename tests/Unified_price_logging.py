from datetime import datetime
import time
from python_bitvavo_api.bitvavo import Bitvavo


def main():
    # Setup bitvavo-client (test- of live-API keys, of public data-call)
    client = Bitvavo({
        "APIKEY": "90b22d9d311f10d40b009ff55b60b920a0829ce3f678c168d4ef8ff6af99027b",
        "APISECRET": "7eed494f4b450e4b9bfa70b3b954937158b6c7e23984980cfc194215364d12287d92c4cb2d518cfcc53b822718b8d495ece9a41a32d93773b6fada32d5d2d58a",
        "RESTURL": "https://api.bitvavo.com/v2",
        "WSURL": "wss://ws.bitvavo.com/v2/"
    })

    while True:
        ticker = client.tickerPrice({"market": "XRP-EUR"})
        now = datetime.utcnow().isoformat()
        # Log naar console en file
        with open("debug_prices.csv", "a") as f:
            f.write(f"{now},{ticker['price']}\n")
        print(f"{now} - {ticker['price']}")
        time.sleep(1)


if __name__ == "__main__":
    main()
