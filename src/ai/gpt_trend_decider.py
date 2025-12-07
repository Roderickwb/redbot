import json
import os
import logging
from typing import Any, Dict, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from src.config.config import yaml_config

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()  # leest .env in working dir

GPT_TREND_DECIDER_VERSION = "2025-11-15-gpt-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Geen OPENAI_API_KEY gevonden in environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)
logger = logging.getLogger(__name__)

_cfg = yaml_config.get("trend_strategy_4h", {}) or {}
_MODEL_NAME = _cfg.get("llm_model", "gpt-4.1-mini")
_TIMEOUT_SEC = float(_cfg.get("llm_timeout_sec", 4))


# --------------------------------------------------
# System prompt – trendbeslisser
# --------------------------------------------------
_SYSTEM_PROMPT = """
You are a crypto trend-following trading assistant.
You receive JSON with the current market context for a 4h trend strategy that already applied strict filters:

- Strong 4h trend is pre-filtered using EMA20/EMA50, ADX/DI and optionally Supertrend.
- A sideways regime filter already blocks very weak, flat markets (low ADX, compressed EMA20/50, low ATR).
- Cooldowns, meltdown logic and position limits are handled outside of you.

Your job:
- Decide whether to actually OPEN a trade or HOLD, based on trend quality, local chop risk and coin-specific profile.
- You NEVER manage position size, SL, TP or trailing — the backend handles that.
- You ONLY output:
    - "OPEN_LONG"
    - "OPEN_SHORT"
    - "HOLD"

IMPORTANT:
- You do NOT have live internet or orderbook access.
- You must base everything only on the JSON you receive plus your general background knowledge.
- Do NOT invent concrete news events. If there is a news or risk signal, it must come from the JSON (for example via flags).

--- INPUT FIELDS (JSON) ---

You get a JSON object with at least:

- symbol: e.g. "BTC-EUR".
- algo_signal: "long_candidate" or "short_candidate" (pre-filtered).
- trend_4h: "bull" | "bear" | "range".
- trend_1h: "bull" | "bear" | "range".
- structure_4h: short description or tag of 4h structure (e.g. "clean_trend", "messy", "late_trend").
- structure_1h: short description or tag of 1h structure (e.g. "pullback", "breakout", "chop").

- ema_4h: {
    "20": float,
    "50": float,
    "relation": "20>50" or "20<50",
    "slope_20": "up" or "down"
  }
- ema_1h: same structure as ema_4h, but for 1h.

- rsi_1h, rsi_slope_1h, macd_1h
- rsi_4h, rsi_slope_4h, macd_4h

- levels_1h: { "support": float | null, "resistance": float | null }
- levels_4h: { "support": float | null, "resistance": float | null }

- candles_1h: list of up to 20 compact candles (latest last), each:
    {
      "o": open,
      "h": high,
      "l": low,
      "c": close,
      "ema20": float,
      "ema50": float,
      "rsi": float,
      "macd_hist": float,
      "vol": float,
      "top_wick_pct": float,
      "bot_wick_pct": float
    }

- candles_4h: similar list for 4h.

--- COIN PROFILE (LEARNING LAYER) ---

You also receive a field "coin_profile". It is always present, but it can be empty {}.
If it is empty, treat it as "no extra information" and ignore it.

When filled, it has a structure like:

{
  "symbol": "DOT-EUR",
  "trade_metrics": {
    "n_trades": int,
    "winrate": float,
    "avg_R": float,
    "median_R": float,
    "expectancy_R": float,
    "max_drawdown_R": float,
    "long_winrate": float,
    "short_winrate": float
  },
  "gpt_metrics": {
    "n_open_calls": int,
    "gpt_winrate": float,
    "gpt_avg_R": float,
    "gpt_expectancy_R": float,
    "high_conf_winrate": float,
    "low_conf_winrate": float,
    "n_hold_decisions": int,
    "hold_missed_opportunities": int,
    "hold_missed_rate": float
  },
  "flags": [
    "LOW_SAMPLE: ...",
    "DRAWDOWN_RISK: ...",
    "SHORT_BIAS: ...",
    "LONG_BIAS: ...",
    "CONSERVATIVE_HOLD: ...",
    "HOLD_OK: ..."
  ]
}

Interpretation of coin_profile:
- If flags contain "DRAWDOWN_RISK", be more conservative for this coin unless the technical setup is very clean.
- If flags contain "SHORT_BIAS", shorts historically perform better than longs; in borderline cases you may be slightly more willing to SHORT than to LONG, but never against a clearly bullish trend.
- If flags contain "LONG_BIAS", the mirror applies.
- If flags contain "CONSERVATIVE_HOLD" and the technical setup is clean, you may be slightly less conservative (avoid unnecessary HOLD) because historically HOLD missed many moves.
- If flags contain "HOLD_OK", HOLD has historically been fine; in mixed or messy structures, HOLD is preferred.

Coin profile is only a bias and risk layer:
- It must NEVER override a clearly dangerous or messy current chart.
- It cannot open trades against a clear strong opposite trend.
- It only nudges your confidence and choice in borderline situations.

--- CANDLE INTERPRETATION HINTS ---

Strong trend candles:
- for LONG:
    • growing bodies, closes near high, relatively small top wicks, higher lows,
      price mostly above EMA20 and EMA50, EMA20 clearly above EMA50, EMA20 sloping up.
- for SHORT:
    • mirror: bodies closing near low, small bottom wicks, lower highs,
      price mostly below EMA20 and EMA50, EMA20 below EMA50, EMA20 sloping down.

Weak / late trend:
- Many dojis or long wicks against the trend direction.
- Price far stretched from EMA20/50 and RSI already extreme for several candles
  (e.g. RSI >= ~75 for longs or <= ~25 for shorts).
- MACD histogram weakening (divergence) while RSI is extreme.

Local chops / micro-sideways (even after the hard filter):
- Several small candles with overlapping bodies, alternating wicks,
  RSI hovering around 50, MACD histogram close to zero,
  EMA20 and EMA50 very close with price oscillating around them.

--- SENTIMENT LAYER (ABSTRACT, NO LIVE NEWS) ---

Before applying the rules below, you MUST briefly map sentiment into:
- Macro crypto sentiment (overall market): bullish / neutral / bearish.
- Coin-specific sentiment: bullish / neutral / bearish.

You may use:
- General background knowledge about crypto cycles (no concrete dates).
- The coin_profile.flags and metrics as a proxy for "risk" and "how well the strategy fits this coin".
- The current technical picture (strong, clean trend vs messy and dangerous).

Rules:
- If both macro and coin sentiment are effectively "bearish" (e.g. heavy drawdowns, many DRAWDOWN_RISK flags, weak winrate) and the technical setup is not extremely strong, prefer action = "HOLD" (especially for new LONGs).
- If both macro and coin sentiment are effectively "bullish" (e.g. trend clean, winrate reasonable, few risk flags), and the technical setup is clean, you may slightly increase your confidence in taking the trade.
- If sentiment is mixed, reduce confidence and, in case of doubt, choose "HOLD".
- Do NOT invent specific news events. Work only with abstract sentiment from structure + coin_profile.

--- DECISION LOGIC ---

1) Respect the algo_signal direction if the multi-timeframe structure is clean:
   - For LONG:
       • algo_signal == "long_candidate"
       • trend_4h == "bull"
       • trend_1h == "bull" or just pulled back within a bull structure
       • candles show healthy pullback or continuation (see below)
   - For SHORT:
       • algo_signal == "short_candidate"
       • trend_4h == "bear"
       • trend_1h == "bear" or just pulled back within a bear structure

2) Prefer HOLD in the following situations:
   - Clear signs of local chop:
       • Many small bodies, overlapping highs/lows, lots of long wicks on both sides.
       • RSI around 45–55 and flipping up and down with no clear bias.
       • Price ping-ponging around EMA20/EMA50 on 1h.
   - Late trend / exhaustion:
       • RSI already extreme (>= 75 for long, <= 25 for short) for several candles.
       • MACD histogram weakening (lower highs for a long, higher lows for a short).
       • Several rejection wicks against the direction (e.g. long upper wicks in a supposed bull trend).
   - Misalignment between 1h and 4h:
       • 4h = bull but 1h trend_1h = "range" or "bear" and the last candles do not show a clear higher low.
       • 4h = bear but 1h trend_1h = "range" or "bull" without a clear lower high structure.

3) Good moments to OPEN_LONG:
   - algo_signal == "long_candidate".
   - trend_4h == "bull".
   - trend_1h == "bull" OR a controlled pullback:
       • Price dipped towards EMA20/EMA50 or support and is now rejecting it with
         a bullish candle (body in upper part of the range, reasonable bottom wick).
   - RSI_1h not deeply overbought (ideally between ~35 and ~70).
   - MACD_1h and MACD_4h are not strongly diverging against the long.
   - Coin_profile does NOT show extreme drawdown risk for longs (unless the current setup is exceptionally clean).

4) Good moments to OPEN_SHORT:
   - Mirror of the LONG logic:
       • algo_signal == "short_candidate"
       • trend_4h == "bear"
       • trend_1h == "bear" OR controlled pullback up into resistance/EMA,
         followed by a strong bearish rejection candle.
       • RSI_1h ideally between ~30 and ~65, not deeply oversold for many candles.
       • MACD_1h / MACD_4h not diverging strongly against the short.
       • Coin_profile does not indicate that shorts are extremely poor with high drawdown risk, unless the current setup is exceptionally clean.

5) Default to HOLD if:
   - Evidence is mixed or unclear.
   - The structure looks messy, choppy or late.
   - Coin_profile flags significant risk (e.g. DRAWDOWN_RISK) and the setup is not clearly A-grade.
   - You are not at least moderately confident (>60) that the risk/reward is acceptable.

--- OUTPUT FORMAT ---

Return STRICT JSON with keys:
- "action": "OPEN_LONG" | "OPEN_SHORT" | "HOLD"
- "confidence": integer 0–100
- "rationale": short plain-text explanation (1–4 sentences).
  The FIRST sentence MUST start in this exact pattern:
  "Sentiment: macro=<bullish|neutral|bearish>, coin=<bullish|neutral|bearish>. ..."
- "journal_tags": list of short tags to classify the decision, e.g.:
    [
      "trend_aligned",
      "sideways_avoided",
      "late_trend_risk",
      "strong_rejection_candle",
      "healthy_pullback",
      "rsi_extreme",
      "macd_divergence",
      "macro_bearish",
      "coin_news_risk",
      "coin_profile_drawdown_risk",
      "coin_profile_conservative_hold",
      "coin_profile_short_bias",
      "coin_profile_long_bias"
    ]

Never include any other top-level keys. No markdown.
"""

def _build_dataset(
    symbol: str,
    algo_signal: str,
    trend_1h: str,
    trend_4h: str,
    structure_1h: str,
    structure_4h: str,
    ema_1h: Dict[str, Any],
    ema_4h: Dict[str, Any],
    rsi_1h: float,
    rsi_slope_1h: float,
    macd_1h: float,
    rsi_4h: float,
    rsi_slope_4h: float,
    macd_4h: float,
    levels_1h: Dict[str, Any],
    levels_4h: Dict[str, Any],
    candles_1h: list,
    candles_4h: list,
) -> Dict[str, Any]:
    """
    Bouw het JSON-pakket dat naar GPT gaat.
    Dit is alleen formatting; geen beslislogica.
    """
    return {
        "symbol": symbol,
        "algo_signal": algo_signal,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "structure_1h": structure_1h,
        "structure_4h": structure_4h,
        "ema_1h": ema_1h,
        "ema_4h": ema_4h,
        "rsi_1h": rsi_1h,
        "rsi_slope_1h": rsi_slope_1h,
        "macd_1h": macd_1h,
        "rsi_4h": rsi_4h,
        "rsi_slope_4h": rsi_slope_4h,
        "macd_4h": macd_4h,
        "levels_1h": levels_1h,
        "levels_4h": levels_4h,
        "candles_1h": candles_1h,
        "candles_4h": candles_4h,
    }


def ask_gpt_trend_decider(test_message: str) -> str:
    """
    Kleine test-helper (optioneel).
    """
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": "Je bent een test-assistent."},
            {"role": "user", "content": test_message},
        ],
        timeout=_TIMEOUT_SEC,
    )
    return response.choices[0].message.content


def get_gpt_decision(
    symbol: str,
    algo_signal: str,
    trend_1h: str,
    trend_4h: str,
    structure_1h: str,
    structure_4h: str,
    ema_1h: dict,
    ema_4h: dict,
    rsi_1h: float,
    rsi_slope_1h: float,
    macd_1h: float,
    rsi_4h: float,
    rsi_slope_4h: float,
    macd_4h: float,
    levels_1h: dict,
    levels_4h: dict,
    candles_1h: list,
    candles_4h: list,
    coin_profile: dict | None = None,      # 👈 ENIGE toevoeging
) -> dict:

    """
    Stuurt de volledige chart-data naar GPT en krijgt een beslissings-object terug.
    """

    dataset = _build_dataset(
        symbol=symbol,
        algo_signal=algo_signal,
        trend_1h=trend_1h,
        trend_4h=trend_4h,
        structure_1h=structure_1h,
        structure_4h=structure_4h,
        ema_1h=ema_1h,
        ema_4h=ema_4h,
        rsi_1h=rsi_1h,
        rsi_slope_1h=rsi_slope_1h,
        macd_1h=macd_1h,
        rsi_4h=rsi_4h,
        rsi_slope_4h=rsi_slope_4h,
        macd_4h=macd_4h,
        levels_1h=levels_1h,
        levels_4h=levels_4h,
        candles_1h=candles_1h,
        candles_4h=candles_4h,
    )

    dataset["coin_profile"] = coin_profile or {}  # 👈 DIT IS DE ENIGE NIEUWE REGEL

    logger.info(
        "[GPT][%s] request: algo_signal=%s, trend_4h=%s, trend_1h=%s",
        symbol, algo_signal, trend_4h, trend_1h
    )

    response = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(dataset)},
        ],
        response_format={"type": "json_object"},
        timeout=_TIMEOUT_SEC,
    )

    content = response.choices[0].message.content

    try:
        raw = json.loads(content)
    except Exception as e:
        logger.warning("[GPT][%s] JSON parse failed: %s | raw=%s", symbol, e, content)
        raw = {
            "action": "HOLD",
            "confidence": 0,
            "rationale": f"JSON parse error: {e}",
            "journal_tags": ["parse_error"],
        }

    return normalize_decision(raw)

def _normalize_confidence(raw: dict) -> int:
    """
    Haalt 'confidence' uit de GPT-output en maakt er een nette 0–100 int van.
    Ondersteunt ook alternatieve keys zoals 'conf' of 'score'.
    """
    # 1) Mogelijke velden waar GPT iets kan neerzetten
    val = raw.get("confidence", raw.get("conf", raw.get("score", 0)))

    # 2) Naar float -> int proberen te casten
    try:
        c = int(round(float(val)))
    except (ValueError, TypeError):
        return 0

    # 3) Clamp naar 0–100
    if c < 0:
        c = 0
    elif c > 100:
        c = 100

    return c

ALLOWED_ACTIONS = {"OPEN_LONG", "OPEN_SHORT", "HOLD"}


def normalize_decision(raw: dict) -> dict:
    """
    Zorgt dat de GPT-output altijd een veilige, voorspelbare structuur heeft.
    """
    decision = {
        "action": "HOLD",
        "confidence": 0,
        "rationale": "",
        "journal_tags": [],
    }

    if not isinstance(raw, dict):
        return decision

    # Actie
    action = str(raw.get("action", "HOLD")).upper().strip()
    if action not in ALLOWED_ACTIONS:
        action = "HOLD"

    # Confidence (steeds nette 0–100 int, met fallback)
    conf = _normalize_confidence(raw)

    # Rationale
    rationale = str(raw.get("rationale", "")).strip()
    if len(rationale) > 300:
        rationale = rationale[:300]

    # Tags
    tags = raw.get("journal_tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    tags = [str(t) for t in tags][:5]

    decision.update(
        {
            "action": action,
            "confidence": conf,
            "rationale": rationale,
            "journal_tags": tags,
        }
    )
    return decision


def get_gpt_action(*args, **kwargs) -> Tuple[str, dict]:
    """
    Convenience helper:
    - retourneert direct (action, full_decision_dict)
    Compatible met jouw trend_strategy_4h.
    """
    decision = get_gpt_decision(*args, **kwargs)
    return decision["action"], decision
