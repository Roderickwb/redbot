# API-sleutels configureren

from python_bitvavo_api.bitvavo import Bitvavo



bitvavo = Bitvavo({
    'APIKEY': '90b22d9d311f10d40b009ff55b60b920a0829ce3f678c168d4ef8ff6af99027b',  # Vervang door je eigen API-sleutel
    'APISECRET': '7eed494f4b450e4b9bfa70b3b954937158b6c7e23984980cfc194215364d12287d92c4cb2d518cfcc53b822718b8d495ece9a41a32d93773b6fada32d5d2d58a',
    # Vervang door je eigen geheime sleutel
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/',
    'ACCESSWINDOW': 10000
})