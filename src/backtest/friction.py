"""
Indian Market Friction Calculator Module.
Computes realistic NSE/BSE transaction costs: STT, GST on brokerage, SEBI turnover charge, stamp duty, and slippage.
"""

from pydantic import BaseModel, Field


class TransactionCosts(BaseModel):
    entry_brokerage: float
    exit_brokerage: float
    entry_stt: float
    exit_stt: float
    sebi_turnover_charge: float
    exchange_txn_charge: float
    gst_on_brokerage: float
    stamp_duty: float
    slippage_cost: float
    total_cost_rupees: float
    total_cost_pct: float


class IndianFrictionModel:
    """
    Calculates realistic Indian equity market round-trip trading costs.
    Based on NSE/BSE official charge schedules as of FY2026.
    """

    # Zerodha/discount broker flat fee for equity delivery
    BROKERAGE_PER_LEG = 0.0  # Zero brokerage for equity delivery
    BROKERAGE_MIN = 0.0
    STT_BUY = 0.001       # 0.1% on buy (CNC equity delivery)
    STT_SELL = 0.001      # 0.1% on sell (CNC equity delivery)
    NSE_TXN_CHARGE = 0.0000297   # NSE equity transaction charge
    SEBI_CHARGE = 0.000001        # SEBI regulatory fee per Rupee
    STAMP_DUTY = 0.00015         # 0.015% on buy-side only
    GST_RATE = 0.18               # 18% GST on brokerage
    DEFAULT_SLIPPAGE_PCT = 0.0005  # 0.05% per-side slippage estimate

    @classmethod
    def calculate_round_trip(
        cls,
        entry_price: float,
        exit_price: float,
        shares: int,
        slippage_pct: float | None = None,
    ) -> TransactionCosts:
        """Calculates full round-trip transaction costs for an Indian equity swing trade."""
        slip = slippage_pct if slippage_pct is not None else cls.DEFAULT_SLIPPAGE_PCT

        entry_value = entry_price * shares
        exit_value = exit_price * shares

        # Brokerage (zero for discount broker equity delivery)
        entry_brokerage = 0.0
        exit_brokerage = 0.0

        # STT
        entry_stt = entry_value * cls.STT_BUY
        exit_stt = exit_value * cls.STT_SELL

        # SEBI turnover charge (on both legs)
        sebi_charge = (entry_value + exit_value) * cls.SEBI_CHARGE

        # NSE Exchange Transaction Charge
        exchange_txn = (entry_value + exit_value) * cls.NSE_TXN_CHARGE

        # GST on brokerage (zero when brokerage is zero)
        gst = (entry_brokerage + exit_brokerage) * cls.GST_RATE

        # Stamp duty (only on buy leg)
        stamp = entry_value * cls.STAMP_DUTY

        # Slippage cost (one-side spread on each leg)
        slippage_cost = (entry_value + exit_value) * slip

        total_cost = (
            entry_stt + exit_stt + sebi_charge + exchange_txn + gst + stamp + slippage_cost
        )
        total_cost_pct = (total_cost / entry_value) * 100.0 if entry_value > 0 else 0.0

        return TransactionCosts(
            entry_brokerage=round(entry_brokerage, 2),
            exit_brokerage=round(exit_brokerage, 2),
            entry_stt=round(entry_stt, 2),
            exit_stt=round(exit_stt, 2),
            sebi_turnover_charge=round(sebi_charge, 2),
            exchange_txn_charge=round(exchange_txn, 2),
            gst_on_brokerage=round(gst, 2),
            stamp_duty=round(stamp, 2),
            slippage_cost=round(slippage_cost, 2),
            total_cost_rupees=round(total_cost, 2),
            total_cost_pct=round(total_cost_pct, 4),
        )
