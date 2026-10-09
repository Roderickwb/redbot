"""Accounting for legacy positions whose child PnL includes exit fees."""


def position_accounting(master: dict, children: list[dict]) -> dict:
    exits = [row for row in children if row.get("status") in {"partial", "closed"}]
    exit_fees = sum(float(row.get("fees") or 0) for row in exits)
    total_fees = float(master.get("fees") or 0)
    entry_fees = total_fees - exit_fees
    if entry_fees < -0.000001:
        raise ValueError("Master fees are smaller than accumulated exit fees")
    entry_fees = max(entry_fees, 0.0)
    reported = sum(float(row.get("pnl_eur") or 0) for row in exits) if exits else float(master.get("pnl_eur") or 0)
    return {"pnl_eur": reported - entry_fees, "reported_pnl_eur": reported,
            "entry_fees_eur": entry_fees, "exit_fees_eur": exit_fees,
            "fees_eur": total_fees, "accounting_version": "net_entry_exit_v1"}
