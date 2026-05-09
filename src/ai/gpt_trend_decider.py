import json
import os
import logging
from typing import Any, Dict, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from src.config.config import yaml_config
from src.analysis.coin_profile_loader import load_coin_profile_json


from src.database_manager.database_manager import (
    DatabaseManager,
    get_current_utc_timestamp_ms,
)


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

- chart_features: compact computed market-structure summary, for example:
  {
    "1h": {
      "structure_label": "clean_trend" | "pullback" | "late_trend" | "chop" | "mixed",
      "entry_timing": "early" | "clean" | "late" | "noisy",
      "ema20_distance_pct": float,
      "ema50_distance_pct": float,
      "ema_spread_pct": float,
      "atr_pct": float,
      "trend_age_bars": int,
      "pullback_depth_pct": float,
      "support_distance_pct": float | null,
      "resistance_distance_pct": float | null,
      "last_candle_quality": "bull_rejection" | "bear_rejection" | "strong_bull" | "strong_bear" | "doji" | "neutral",
      "last_close_location_pct": float,
      "recent_doji_count": int,
      "recent_opposing_wick_count": int
    },
    "4h": { "... same idea ..." }
  }

--- COIN PROFILE (LEARNING LAYER) ---

You also receive a field "coin_profile". It is always present, but it can be empty {}.
If it is empty, treat it as "no extra information" and ignore it.

When filled, it has a structure like:

{
  "symbol": "XBT-EUR",
  "market_regime": "bull" | "bear" | "range",
  "regime_strength": float between 0.0 and 1.0,
  "long_edge": float,        // historical edge for longs (avg R)
  "short_edge": float,       // historical edge for shorts (avg R)
  "bias": "long_edge" | "short_edge" | "neutral",

  "winrate": float,          // overall winrate of the strategy on this coin
  "expectancy_R": float,     // average R per trade on this coin
  "max_drawdown_R": float,   // worst historical drawdown in R

  "hold_missed_rate": float, // how often HOLD missed a good move
  "hold_behavior": "too_conservative" | "hold_ok" | "balanced" | "unknown",

  "risk_multiplier": float between 0.25 and 1.0, // lower = more risk / worse history

  "learning_confidence": "low" | "medium" | "high",
  "learning_metrics": {
    "events": int,
    "trade_open": int,
    "trade_winrate_pct": float,
    "trade_pnl_eur": float,
    "missed_opportunity": int,
    "missed_rate_pct": float,
    "range_events": int,
    "range_breakout_rate_pct": float,
    "cf_simulated": int,              // counterfactual trades simulated with bot-like SL/TP/trailing rules
    "cf_positive_rate_pct": float,    // share of simulated outcomes with positive R
    "cf_avg_r": float,                // average simulated R-multiple
    "cf_loss": int,
    "cf_win": int,
    "cf_tp1_then_positive": int,
    "cf_ambiguous_intrabar": int
  },

  "flags": [
    "LOW_SAMPLE: ...",
    "DRAWDOWN_RISK: ...",
    "SHORT_BIAS: ...",
    "LONG_BIAS: ...",
    "CONSERVATIVE_HOLD: ...",
    "HOLD_OK: ...",
    "FILTER_REVIEW: ...",
    "RANGE_BREAKOUT_CANDIDATE: ...",
    "SAMPLE_LOW: ...",
    "COUNTERFACTUAL_EDGE_POSITIVE: ...",
    "COUNTERFACTUAL_EDGE_NEGATIVE: ..."
  ]
}

Interpretation of coin_profile:
- Use market_regime + regime_strength as a soft sentiment for how well the strategy fits this coin:
  • bull  → history is generally positive for this strategy on this coin.
  • bear  → history is generally negative.
  • range → unclear / mixed.
- Use risk_multiplier as a simple risk dial:
  • 0.25 → be very selective, only A-grade setups.
  • 0.5  → be conservative.
  • 0.75 → slightly cautious.
  • 1.0  → normal risk.

- If flags contain "DRAWDOWN_RISK", be more conservative for this coin unless the technical setup is very clean.
- If flags contain "SHORT_BIAS", shorts historically perform better than longs; in borderline cases you may be slightly more willing to SHORT than to LONG, but never against a clearly bullish trend.
- If flags contain "LONG_BIAS", the mirror applies.
- If flags contain "CONSERVATIVE_HOLD" and the technical setup is clean, you may be slightly less conservative (avoid unnecessary HOLD) because historically HOLD missed many moves.
- If flags contain "HOLD_OK", HOLD has historically been fine; in mixed or messy structures, HOLD is preferred.
- If flags contain "FILTER_REVIEW", the learning layer found missed opportunities after skips/holds. Do not blindly open, but in a clean current setup be slightly less conservative.
- If flags contain "RANGE_BREAKOUT_CANDIDATE", this coin has recently broken out after range-like contexts. Treat range/chop risk with nuance: still HOLD messy chop, but do not reject a clean breakout/pullback only because the profile says range.
- If flags contain "COUNTERFACTUAL_EDGE_NEGATIVE", recent simulated bot-style outcomes were negative for this coin. Require unusually clean chart quality; in borderline cases choose HOLD.
- If flags contain "COUNTERFACTUAL_EDGE_POSITIVE", recent simulated bot-style outcomes were positive for this coin. This can support confidence in a clean setup, but never creates a trade by itself.
- If flags contain "SAMPLE_LOW" or learning_confidence is "low", treat all profile hints as weak evidence. Current chart quality must dominate.

Coin profile is only a bias and risk layer:
- It must NEVER override a clearly dangerous or messy current chart.
- It cannot open trades against a clear strong opposite trend.
- It only nudges your confidence and choice in borderline situations.
- Profile flags are soft nudges, not commands. Technical trend quality and risk filters remain primary.

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

You also receive a field "sentiment" in the JSON with this structure:

- sentiment.macro: overall crypto market sentiment from external data
  (e.g. Fear & Greed index), with a "label" field: "bullish" / "neutral" / "bearish".
- sentiment.coin: coin-specific performance sentiment, also with a "label".
- sentiment.chain: chain-level sentiment (e.g. BTC, ETH), also with a "label".

When you write the first sentence of the rationale, you MUST derive:
- macro=<...> from sentiment.macro.label (fallback: "neutral" if missing).
- coin=<...> from sentiment.coin.label (fallback: "neutral" if missing).

You may combine these external labels with coin_profile (risk and bias)
and the current technical picture, but never invent news or events.

Rules:
- If both macro and coin sentiment are effectively "bearish" (e.g. heavy drawdowns, many DRAWDOWN_RISK flags, weak winrate) and the technical setup is not extremely strong, prefer action = "HOLD" (especially for new LONGs).
- If both macro and coin sentiment are effectively "bullish" (e.g. trend clean, winrate reasonable, few risk flags), and the technical setup is clean, you may slightly increase your confidence in taking the trade.
- If sentiment is mixed, reduce confidence and, in case of doubt, choose "HOLD".
- Do NOT invent specific news events. Work only with abstract sentiment from structure + coin_profile.

--- DECISION HIERARCHY ---

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
      "coin_profile_filter_review",
      "coin_profile_range_breakout_candidate",
      "coin_profile_counterfactual_positive",
      "coin_profile_counterfactual_negative",
      "coin_profile_low_sample",
      "coin_profile_short_bias",
      "coin_profile_long_bias"
    ]
- "scores": object with integer 0-100 values:
    {
      "trend": 0-100,
      "entry": 0-100,
      "risk": 0-100,
      "learning": 0-100,
      "sentiment": 0-100
    }
- "primary_veto": one of:
    "none",
    "trend_misaligned",
    "weak_entry",
    "local_chop",
    "late_trend",
    "counterfactual_negative",
    "drawdown_risk",
    "sentiment_risk",
    "mixed_evidence"
- "learning_effect": one of:
    "none",
    "reduced_confidence",
    "increased_confidence",
    "blocked_trade",
    "allowed_clean_setup"
- "risk_notes": short string, max 160 chars. Name the biggest invalidation risk.

Never include any other top-level keys. No markdown.
"""

_SYSTEM_PROMPT += """

--- DECISION HIERARCHY OVERRIDE V2 ---

Use this hierarchy as the primary decision process. It takes precedence over
any softer wording above. Think like a risk manager first and a signal-taker
second.

Step 1 - Direction and hard reject checks.
Return HOLD immediately if any hard reject is present:
- The requested action would conflict with algo_signal.
- OPEN_LONG without trend_4h="bull".
- OPEN_SHORT without trend_4h="bear".
- The 1h chart is materially against the 4h trend and the latest candles do not show a controlled pullback/rejection back into the 4h direction.
- Clear local chop: overlapping candles, alternating wicks, RSI near 50, flat MACD histogram, price whipsawing around EMA20/EMA50.
- Clear exhaustion: RSI extreme for several candles, weakening MACD histogram, repeated rejection wicks against the intended direction.

Step 2 - Chart quality gate.
Only continue toward OPEN if the current chart itself is good enough:
- A clean 4h trend exists in the intended direction.
- The 1h entry is either trend-aligned or a controlled pullback into EMA/support/resistance with a fresh rejection in the intended direction.
- Latest candles are not mostly dojis, not mostly opposing wicks, and not stretched far away from EMA20/EMA50.
- MACD and RSI do not show obvious momentum decay against the trade.
- Use chart_features as the compact trader summary:
  - structure_label="chop" or entry_timing="noisy" is a strong HOLD signal.
  - structure_label="late_trend" or entry_timing="late" lowers confidence sharply.
  - structure_label="pullback" with a supportive rejection candle is often better than chasing extension.
  - high trend_age_bars plus large EMA distance means late-entry risk.
  - last_candle_quality should support the intended direction; opposing rejection is a HOLD signal.

If chart quality is weak, HOLD. Coin profile and sentiment are not allowed to rescue a weak chart.

Step 3 - Learning profile gate.
Use coin_profile after the chart passes Step 2:
- risk_multiplier <= 0.5: require an exceptional A-grade chart and confidence >= 80.
- risk_multiplier == 0.75: require a cleaner-than-average chart and confidence >= 70.
- DRAWDOWN_RISK or COUNTERFACTUAL_EDGE_NEGATIVE: treat the coin as hostile until proven otherwise. OPEN only when the technical setup is unusually clean; otherwise HOLD.
- COUNTERFACTUAL_EDGE_POSITIVE: may add confidence to a clean setup, but never opens a trade by itself.
- FILTER_REVIEW or CONSERVATIVE_HOLD: the bot may have been too conservative before. In a genuinely clean setup, avoid unnecessary HOLD.
- HOLD_OK: in mixed or unclear conditions, HOLD is preferred.
- RANGE_BREAKOUT_CANDIDATE: do not reject a clean breakout or clean pullback only because the broader context had range behavior. Still HOLD messy chop.
- SAMPLE_LOW or learning_confidence="low": profile evidence is weak. Let current chart quality dominate.

Use learning_metrics when present:
- cf_avg_r < -0.25 or cf_positive_rate_pct < 45 means the simulated bot-style exits have been poor recently; lower confidence and prefer HOLD unless the setup is excellent.
- cf_avg_r > 0.25 and cf_positive_rate_pct >= 55 supports the trade only after chart quality is already clean.
- trade_open below 5 means real-trade evidence is still thin; do not overtrust winrate or pnl.

Step 4 - Sentiment layer.
Sentiment is a small adjustment only:
- Bearish macro + bearish coin sentiment makes LONG harder to justify unless the technical setup is very strong.
- Bullish macro + bullish coin sentiment can support a clean setup, but cannot rescue a messy one.
- Mixed sentiment lowers confidence and favors HOLD in borderline cases.

Step 5 - Final action rule.
OPEN only when all are true:
- Direction agrees with algo_signal and 4h trend.
- Entry quality is clean enough now, not just historically.
- Learning profile does not strongly warn against this coin, or the chart is exceptional enough to override the warning.
- The rationale can explain why entering now is better than waiting.

HOLD when:
- evidence is mixed,
- chart is late/choppy,
- entry is weak,
- risk flags dominate,
- or the case depends mostly on profile/sentiment instead of current price structure.

Confidence scale:
- 80-90: clean trend, clean entry, no major risk flags, learning not negative.
- 70-79: valid setup with minor concerns, or risk_multiplier=0.75 but chart is strong.
- 60-69: valid but borderline; usually HOLD unless learning and chart both support entry.
- below 60: HOLD.

Before finalizing, ask internally:
- Why not HOLD here?
- What is the biggest invalidation risk?
- Is the trade justified by the chart first, or mostly by learning/sentiment?
If it is not chart-first, choose HOLD.

Structured scoring discipline:
- trend score: quality/alignment of 4h and 1h trend.
- entry score: quality of the latest entry candles and rejection/pullback.
- risk score: higher means cleaner risk, lower means drawdown/chop/exhaustion risk.
- learning score: higher means coin_profile supports the setup, lower means profile warns against it.
- sentiment score: higher means sentiment supports the setup, lower means sentiment warns against it.
- For HOLD, primary_veto must name the main reason.
- For OPEN, primary_veto should be "none" unless there is a minor risk you explicitly accepted.
- learning_effect must explain whether profile data reduced, increased, blocked, or merely allowed the decision.

Journal tag discipline:
- Include at least one chart tag: trend_aligned, healthy_pullback, strong_rejection_candle, local_chop, late_trend_risk, or weak_entry.
- Include profile tags when profile materially influenced the decision:
  coin_profile_drawdown_risk, coin_profile_counterfactual_negative,
  coin_profile_counterfactual_positive, coin_profile_filter_review,
  coin_profile_conservative_hold, coin_profile_range_breakout_candidate,
  coin_profile_low_sample.
- Include hold_default_mixed_evidence when choosing HOLD mainly because the case is unclear.
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
    chart_features: Dict[str, Any] | None = None,
    coin_profile: Dict[str, Any] | None = None,
    sentiment: Dict[str, Any] | None = None,
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
        "chart_features": chart_features or {},
        "coin_profile": coin_profile or {},
        "sentiment": sentiment or {},
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
    chart_features: dict | None = None,
    coin_profile: dict | None = None,
    sentiment: dict | None = None,
    *,
    strategy_name: str = "trend_4h",
    db: DatabaseManager | None = None,
) -> dict:
    """
    Stuurt de volledige chart-data naar GPT en krijgt een beslissings-object terug.
    """

    # --- coin_profile: altijd de FULL JSON uit DB (als db beschikbaar is) ---
    if coin_profile is None and db is not None:
        coin_profile = load_coin_profile_json(db, symbol, strategy_name=strategy_name) or {}

    # provenance (zodat je in Telegram/logs ziet waar het vandaan komt)
    if isinstance(coin_profile, dict):
        coin_profile["_source"] = "db" if coin_profile else "none"


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
        chart_features=chart_features,
        coin_profile=coin_profile,
        sentiment=sentiment,
    )

    logger.info(
        "[GPT][%s] request: algo_signal=%s, trend_4h=%s, trend_1h=%s",
        symbol, algo_signal, trend_4h, trend_1h
    )

    # 1) GPT-call
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

    # 2) Normaliseren
    decision = normalize_decision(raw)

    # 3) Optioneel: in DB loggen
    if db is not None:
        ts_now = get_current_utc_timestamp_ms()

        # --- gpt_decisions ---
        try:
            db.save_gpt_decision({
                "timestamp": ts_now,
                "symbol": symbol,
                "strategy_name": strategy_name,

                "algo_signal": algo_signal,
                "gpt_action": decision.get("action"),
                "confidence": decision.get("confidence"),
                "rationale": decision.get("rationale"),
                "journal_tags": decision.get("journal_tags"),

                "request_json": dataset,         # volledige input
                "response_json": decision,       # genormaliseerde output
                "gpt_version": GPT_TREND_DECIDER_VERSION,
                "trade_id": None,                # trade bestaat hier nog niet
            })
        except Exception as e:
            logger.error("[GPT][%s] save_gpt_decision failed: %s", symbol, e)

        # --- gpt_review_cases ---
        try:
            sent_obj = sentiment or {}
            macro_sent = (sent_obj.get("macro") or {}).get("label")
            coin_sent = (sent_obj.get("coin") or {}).get("label")
            chain_sent = (sent_obj.get("chain") or {}).get("label")

            cp = coin_profile or {}
            bias = cp.get("bias")
            risk_mult = cp.get("risk_multiplier")

            db.save_gpt_review_case({
                "timestamp": ts_now,
                "symbol": symbol,
                "strategy_name": strategy_name,

                "trade_id": None,
                "algo_signal": algo_signal,
                "gpt_action": decision.get("action"),
                "confidence": decision.get("confidence"),

                "rationale_short": decision.get("rationale", "")[:160],
                "sentiment_macro": macro_sent,
                "sentiment_coin": coin_sent,
                "sentiment_chain": chain_sent,

                "coin_profile_bias": bias,
                "coin_profile_risk": risk_mult,

                # Deze drie worden later (in je analyse-laag) ingevuld
                "result_label": None,
                "review_label": None,
                "notes": None,

                "raw_json": {
                    "request": dataset,
                    "decision": decision,
                },
            })
        except Exception as e:
            logger.error("[GPT][%s] save_gpt_review_case failed: %s", symbol, e)

    return decision


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
        "scores": {
            "trend": 0,
            "entry": 0,
            "risk": 0,
            "learning": 0,
            "sentiment": 0,
        },
        "primary_veto": "mixed_evidence",
        "learning_effect": "none",
        "risk_notes": "",
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

    # Structured scores for later learning/debugging.
    scores = raw.get("scores") or {}
    if not isinstance(scores, dict):
        scores = {}
    normalized_scores = {
        "trend": _score_0_100(scores.get("trend")),
        "entry": _score_0_100(scores.get("entry")),
        "risk": _score_0_100(scores.get("risk")),
        "learning": _score_0_100(scores.get("learning")),
        "sentiment": _score_0_100(scores.get("sentiment")),
    }

    primary_veto = _normalize_choice(
        raw.get("primary_veto"),
        allowed={
            "none",
            "trend_misaligned",
            "weak_entry",
            "local_chop",
            "late_trend",
            "counterfactual_negative",
            "drawdown_risk",
            "sentiment_risk",
            "mixed_evidence",
        },
        default="mixed_evidence" if action == "HOLD" else "none",
    )
    if action != "HOLD" and primary_veto == "mixed_evidence":
        primary_veto = "none"

    learning_effect = _normalize_choice(
        raw.get("learning_effect"),
        allowed={
            "none",
            "reduced_confidence",
            "increased_confidence",
            "blocked_trade",
            "allowed_clean_setup",
        },
        default="none",
    )

    risk_notes = str(raw.get("risk_notes", "")).strip()
    if len(risk_notes) > 160:
        risk_notes = risk_notes[:160]

    decision.update(
        {
            "action": action,
            "confidence": conf,
            "rationale": rationale,
            "journal_tags": tags,
            "scores": normalized_scores,
            "primary_veto": primary_veto,
            "learning_effect": learning_effect,
            "risk_notes": risk_notes,
        }
    )
    return decision


def _score_0_100(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    if score < 0:
        return 0
    if score > 100:
        return 100
    return score


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    choice = str(value or "").strip()
    if choice in allowed:
        return choice
    return default

def get_gpt_action(
    *args,
    strategy_name: str = "trend_4h",
    db: DatabaseManager | None = None,
    **kwargs,
) -> Tuple[str, dict]:
    """
    Convenience helper:
    - retourneert direct (action, full_decision_dict)
    - kan optioneel een DatabaseManager gebruiken om te loggen
    """
    decision = get_gpt_decision(
        *args,
        strategy_name=strategy_name,
        db=db,
        **kwargs,
    )
    return decision["action"], decision
