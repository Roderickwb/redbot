import json
import os
import openai

GPT_TREND_DECIDER_VERSION = "2025-11-15"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    raise RuntimeError("Geen OPENAI_API_KEY gevonden in environment variables.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def ask_gpt_trend_decider(test_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Je bent een test-assistent."},
            {"role": "user", "content": test_message}
        ]
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
):
    """
    Stuurt de volledige chart-data naar GPT en krijgt LONG/SHORT/HOLD terug.
    """

    # ------- 1) Bouw de volledige dataset -------
    dataset = {
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
        "macd_hist_1h": macd_1h,
        "rsi_4h": rsi_4h,
        "rsi_slope_4h": rsi_slope_4h,
        "macd_hist_4h": macd_4h,
        "levels_1h": levels_1h,
        "levels_4h": levels_4h,
        "candles_1h": candles_1h,
        "candles_4h": candles_4h,
    }

    dataset_str = json.dumps(dataset)

    # ------- 2) Bouw de prompt -------
    prompt = f"""
Je bent een crypto-trendanalist en strategische beslisser.
Analyseer de volledige 1h & 4h dataset hieronder.

DATASET:
{dataset_str}

Geef ALLEEN geldige JSON terug:

{{
  "action": "OPEN_LONG" | "OPEN_SHORT" | "HOLD",
  "confidence": 0-100,
  "rationale": "<max 20 woorden>",
  "journal_tags": ["korte labels"]
}}
    """

    # ------- 3) GPT call -------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Je bent een strikt JSON-only crypto beslisser."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content

    # ------- 4) JSON parse -------
    try:
        result = json.loads(content)
    except Exception:
        # fallback: probeer JSON tussen ``` te halen
        text = content.strip()
        if "```" in text:
            parts = text.split("```")
            # pak het middelste stuk als er code fences zijn
            if len(parts) >= 3:
                text = parts[1]
        result = json.loads(text)

    # ------- 5) Normaliseren -------
    return normalize_decision(result)

ALLOWED_ACTIONS = {"OPEN_LONG", "OPEN_SHORT", "HOLD"}


def normalize_decision(raw: dict) -> dict:
    """
    Zorgt dat de GPT-output altijd een veilige, voorspelbare structuur heeft.
    """
    # Basis fallback
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

    # Confidence
    conf = raw.get("confidence", 0)
    try:
        conf = int(conf)
    except (ValueError, TypeError):
        conf = 0
    conf = max(0, min(100, conf))

    # Rationale
    rationale = str(raw.get("rationale", "")).strip()
    # Beetje inkorten just in case
    if len(rationale) > 200:
        rationale = rationale[:200]

    # Tags
    tags = raw.get("journal_tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    tags = [str(t) for t in tags][:5]  # max 5 tags

    decision.update({
        "action": action,
        "confidence": conf,
        "rationale": rationale,
        "journal_tags": tags,
    })
    return decision

def get_gpt_action(*args, **kwargs) -> tuple[str, dict]:
    """
    Convenience helper:
    - retourneert direct (action, full_decision_dict)
    """
    decision = get_gpt_decision(*args, **kwargs)
    return decision["action"], decision
