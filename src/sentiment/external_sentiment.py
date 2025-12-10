import logging
from typing import Optional, Dict, Any, List

import requests

logger = logging.getLogger(__name__)

# ============================
# Config
# ============================

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"
COINGECKO_MARKET_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

# Mapping van jouw symbols → Coingecko IDs
COIN_ID_MAP: Dict[str, str] = {
    "BTC": "bitcoin",
    "XBT": "bitcoin",      # Kraken
    "ETH": "ethereum",
    "DOT": "polkadot",
    "ADA": "cardano",
    "XRP": "ripple",
    "LTC": "litecoin",
    "LINK": "chainlink",
    "ALGO": "algorand",
    "ATOM": "cosmos",
    "AVAX": "avalanche-2",
    "ETC": "ethereum-classic",
    "XLM": "stellar",
    "XDG": "dogecoin",     # Kraken ticker voor DOGE
    "DOGE": "dogecoin",
    "UNI": "uniswap",
    "AAVE": "aave",
    "BCH": "bitcoin-cash",
    "SOL": "solana",
    "SAND": "the-sandbox",
    "POL": "polygon-ecosystem-token",  # check later indien nodig
}

# ============================
# Helpers
# ============================


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Kleine wrapper rond requests.get → JSON.
    Retourneert None bij fouten, logt een waarschuwing.
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("[external_sentiment] HTTP GET failed for %s (%s)", url, e)
        return None


def _label_from_score(score: Optional[float], bull_threshold: float = 60.0, bear_threshold: float = 40.0) -> str:
    """
    Converteer een 0–100 score naar bullish / neutral / bearish.
    """
    if score is None:
        return "neutral"
    if score >= bull_threshold:
        return "bullish"
    if score <= bear_threshold:
        return "bearish"
    return "neutral"


def _base_symbol_from_trading_symbol(symbol: str) -> str:
    """
    'XBT-EUR' → 'XBT'
    'ETH-USDT' → 'ETH'
    """
    if not symbol:
        return ""
    return symbol.split("-")[0].upper()


def _coingecko_id_for_symbol(symbol: str) -> Optional[str]:
    """
    Zoek coingecko-id voor een trading symbol.
    """
    base = _base_symbol_from_trading_symbol(symbol)
    return COIN_ID_MAP.get(base)


# ============================
# Macro sentiment (Fear & Greed)
# ============================


def get_macro_sentiment() -> Dict[str, Any]:
    """
    Haal macro cryptosentiment op via Fear & Greed index.
    Retourneert een dict met score + label.
    """
    data = _http_get_json(FEAR_GREED_URL)
    if not data or "data" not in data or not data["data"]:
        # Fallback: neutraal
        return {
            "source": "fear_greed_index",
            "value": None,
            "score": None,
            "label": "neutral",
            "text": "No data; treated as neutral.",
        }

    try:
        entry = data["data"][0]
        value = int(entry["value"])  # 0–100
        # We gebruiken value direct als macro-score
        score = float(value)

        if value >= 60:
            label = "bullish"
        elif value <= 40:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "source": "fear_greed_index",
            "value": value,
            "score": score,
            "label": label,
            "classification": entry.get("value_classification"),
        }
    except Exception as e:
        logger.warning("[external_sentiment] parse fear & greed failed: %s", e)
        return {
            "source": "fear_greed_index",
            "value": None,
            "score": None,
            "label": "neutral",
            "text": "Parse error; treated as neutral.",
        }


# ============================
# Coin sentiment via Coingecko
# ============================


def _compute_pct_change(old: float, new: float) -> Optional[float]:
    if old is None or new is None:
        return None
    if old == 0:
        return None
    return (new - old) / old * 100.0


def _extract_prices(prices: List[List[float]]) -> Optional[Dict[str, float]]:
    """
    prices: lijst van [timestamp, price]
    Retourneert:
      - first
      - last
      - approx_24h_ago (als genoeg punten)
    """
    if not prices or len(prices) < 2:
        return None

    first_price = float(prices[0][1])
    last_price = float(prices[-1][1])

    # Vereenvoudigde benadering: neem ±24 punten terug als 24h, als er genoeg data is.
    # Coingecko geeft voor 7d normaal ~169 punten (ongeveer per uur).
    if len(prices) > 24:
        price_24h_ago = float(prices[-25][1])
    else:
        price_24h_ago = None

    return {
        "first": first_price,
        "last": last_price,
        "approx_24h_ago": price_24h_ago,
    }


def get_coin_sentiment(symbol: str, vs_currency: str = "eur") -> Dict[str, Any]:
    """
    Haal eenvoudige coin-sentiment op:
    - 7d performance
    - approx 24h performance
    - label bullish / neutral / bearish
    """
    cg_id = _coingecko_id_for_symbol(symbol)
    if not cg_id:
        base = _base_symbol_from_trading_symbol(symbol)
        logger.warning("[external_sentiment] No Coingecko ID for symbol %s (base=%s)", symbol, base)
        return {
            "symbol": symbol,
            "coin_id": None,
            "score": None,
            "label": "neutral",
            "perf_7d_pct": None,
            "perf_24h_pct": None,
            "source": "coingecko_market_chart",
            "note": "No coingecko id mapping; treated as neutral.",
        }

    params = {"vs_currency": vs_currency, "days": 7}
    url = COINGECKO_MARKET_URL.format(id=cg_id)
    data = _http_get_json(url, params=params)
    if not data or "prices" not in data:
        logger.warning("[external_sentiment] No prices data for %s (%s)", symbol, cg_id)
        return {
            "symbol": symbol,
            "coin_id": cg_id,
            "score": None,
            "label": "neutral",
            "perf_7d_pct": None,
            "perf_24h_pct": None,
            "source": "coingecko_market_chart",
            "note": "No price data; treated as neutral.",
        }

    price_info = _extract_prices(data["prices"])
    if not price_info:
        return {
            "symbol": symbol,
            "coin_id": cg_id,
            "score": None,
            "label": "neutral",
            "perf_7d_pct": None,
            "perf_24h_pct": None,
            "source": "coingecko_market_chart",
            "note": "Not enough price points; treated as neutral.",
        }

    first = price_info["first"]
    last = price_info["last"]
    p24 = price_info["approx_24h_ago"]

    perf_7d = _compute_pct_change(first, last)
    perf_24h = _compute_pct_change(p24, last) if p24 is not None else None

    # Simpele score op basis van 7d performance, begrensd
    score = None
    if perf_7d is not None:
        clamped = max(-30.0, min(30.0, perf_7d))
        # -30 → 0, 0 → 50, +30 → 100
        score = (clamped + 30.0) * (100.0 / 60.0)

    label = _label_from_score(score, bull_threshold=60.0, bear_threshold=40.0)

    return {
        "symbol": symbol,
        "coin_id": cg_id,
        "score": round(score, 2) if score is not None else None,
        "label": label,
        "perf_7d_pct": round(perf_7d, 2) if perf_7d is not None else None,
        "perf_24h_pct": round(perf_24h, 2) if perf_24h is not None else None,
        "source": "coingecko_market_chart",
    }


# ============================
# Chain sentiment (light versie)
# ============================


def get_chain_sentiment(chain_symbol: str, vs_currency: str = "eur") -> Dict[str, Any]:
    """
    Light versie: gebruik dezelfde Coingecko-data maar dan op chain-niveau
    (bv. 'BTC', 'ETH').
    Later kunnen we dit uitbreiden met echte on-chain data (PRO-laag).
    """
    base = chain_symbol.upper()
    # Voor nu: map chain_symbol via COIN_ID_MAP
    cg_id = COIN_ID_MAP.get(base)
    if not cg_id:
        logger.warning("[external_sentiment] No Coingecko ID for chain %s", chain_symbol)
        return {
            "chain": chain_symbol,
            "coin_id": None,
            "score": None,
            "label": "neutral",
            "source": "coingecko_chain_proxy",
            "note": "No mapping; neutral.",
        }

    params = {"vs_currency": vs_currency, "days": 7}
    url = COINGECKO_MARKET_URL.format(id=cg_id)
    data = _http_get_json(url, params=params)
    if not data or "prices" not in data:
        logger.warning("[external_sentiment] No prices data for chain %s (%s)", chain_symbol, cg_id)
        return {
            "chain": chain_symbol,
            "coin_id": cg_id,
            "score": None,
            "label": "neutral",
            "source": "coingecko_chain_proxy",
            "note": "No price data; neutral.",
        }

    price_info = _extract_prices(data["prices"])
    if not price_info:
        return {
            "chain": chain_symbol,
            "coin_id": cg_id,
            "score": None,
            "label": "neutral",
            "source": "coingecko_chain_proxy",
            "note": "Not enough price points; neutral.",
        }

    first = price_info["first"]
    last = price_info["last"]
    p24 = price_info["approx_24h_ago"]

    perf_7d = _compute_pct_change(first, last)
    perf_24h = _compute_pct_change(p24, last) if p24 is not None else None

    score = None
    if perf_7d is not None:
        clamped = max(-30.0, min(30.0, perf_7d))
        score = (clamped + 30.0) * (100.0 / 60.0)

    label = _label_from_score(score, bull_threshold=60.0, bear_threshold=40.0)

    return {
        "chain": chain_symbol,
        "coin_id": cg_id,
        "score": round(score, 2) if score is not None else None,
        "label": label,
        "perf_7d_pct": round(perf_7d, 2) if perf_7d is not None else None,
        "perf_24h_pct": round(perf_24h, 2) if perf_24h is not None else None,
        "source": "coingecko_chain_proxy",
    }


# ============================
# Overkoepelende helper
# ============================


def get_external_sentiment(symbol: str, chain_symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Haal in één call alle externe sentimentlagen op voor een trading symbol.

    - macro: Fear & Greed index
    - coin: Coingecko-marketdata voor deze coin
    - chain: Coingecko-marketdata voor de onderliggende chain (bv. BTC, ETH)
    """
    if chain_symbol is None:
        # standaard: base van symbol (XBT-EUR → XBT)
        chain_symbol = _base_symbol_from_trading_symbol(symbol)

    macro = get_macro_sentiment()
    coin = get_coin_sentiment(symbol)
    chain = get_chain_sentiment(chain_symbol)

    return {
        "macro": macro,
        "coin": coin,
        "chain": chain,
    }


# ============================
# Kleine CLI-test
# ============================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Macro sentiment ===")
    print(get_macro_sentiment())

    print("\n=== Coin sentiment: XBT-EUR ===")
    print(get_coin_sentiment("XBT-EUR"))

    print("\n=== Chain sentiment: BTC ===")
    print(get_chain_sentiment("BTC"))

    print("\n=== Combined external sentiment: XBT-EUR ===")
    print(get_external_sentiment("XBT-EUR"))
