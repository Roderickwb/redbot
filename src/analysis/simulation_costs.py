"""Round-trip fee accounting in R, using the simulator's exit quantities."""

COST_VERSION = "entry_exit_fees_v1"


def apply_simulation_costs(trade: dict, fee_rate: float) -> dict:
    result = dict(trade)
    if result.get("r_multiple") is None:
        return result
    entry = float(result.get("entry_price") or 0)
    risk = float(result.get("risk_per_unit") or 0)
    exit_price = result.get("exit_price")
    if entry <= 0 or risk <= 0 or exit_price is None:
        result["cost_status"] = "missing_prices_or_risk"
        return result
    if not 0 <= fee_rate < 1:
        raise ValueError("fee_rate must be a fraction between 0 and 1")
    gross = round(float(result.get("gross_r_multiple", result["r_multiple"])), 4)
    portion = min(max(float(result.get("tp1_portion_pct") or 0), 0), 1) if result.get("tp1_hit") else 0
    exit_notional = portion * float(result.get("tp1_price") or exit_price) + (1 - portion) * float(exit_price)
    fees_r = fee_rate * (entry + exit_notional) / risk
    result.update(gross_r_multiple=round(gross, 4), fees_r=round(fees_r, 4),
                  r_multiple=round(gross - fees_r, 4), fee_rate=fee_rate,
                  cost_version=COST_VERSION, cost_status="included",
                  end_of_horizon_liquidation=result.get("exit_reason") == "OPEN_END")
    net = result["r_multiple"]
    if result.get("status") == "ambiguous_intrabar":
        label = "cf_ambiguous_intrabar"
    elif net < 0:
        label = "cf_loss"
    elif net == 0:
        label = "cf_breakeven"
    elif result.get("tp1_hit"):
        label = "cf_tp1_then_positive"
    else:
        label = "cf_win" if net >= 1 else "cf_small_win"
    result["label"] = label
    return result
