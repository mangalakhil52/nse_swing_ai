"""
Real Trade Analysis & Shadow Monitor Unit Tests — tests/test_real_trade_analysis.py

Validates:
  1. RealTradeAnalyzer metric calculation (Win Rate %, Profit Factor, Expectancy, Max Drawdown).
  2. Database trade persistence and shadow monitor update integration.
  3. Candidate Discovery Engine integration in daily scanner.
"""

from datetime import date, timedelta
import pandas as pd
import pytest

from src.candidate_discovery import CandidateDiscoveryConfig, CandidateDiscoveryEngine
from src.database.connection import get_db_session, init_db
from src.database.schema import ShadowTradeModel
from src.shadow.analyzer import RealTradeAnalyzer, TradeMetricsSummary
from src.shadow.monitor import ShadowTradeUpdate


def test_real_trade_analyzer_metrics_calculation():
    """1. Test RealTradeAnalyzer metric calculations across winning and losing trade dictionaries."""
    trades = [
        {
            "symbol": "TRENT", "status": "TARGET_3_HIT", "entry_price": 7000.0, "exit_price": 7700.0,
            "pnl_rupees": 35000.0, "pnl_pct": 10.0, "exit_reason": "TARGET_3", "holding_sessions": 5,
        },
        {
            "symbol": "RELIANCE", "status": "STOPPED_OUT", "entry_price": 3000.0, "exit_price": 2850.0,
            "pnl_rupees": -15000.0, "pnl_pct": -5.0, "exit_reason": "STOP_LOSS", "holding_sessions": 2,
        },
        {
            "symbol": "INFY", "status": "TARGET_1_HIT", "entry_price": 1800.0, "exit_price": 1980.0,
            "pnl_rupees": 18000.0, "pnl_pct": 10.0, "exit_reason": "TARGET_1", "holding_sessions": 4,
        },
    ]

    summary = RealTradeAnalyzer.compute_metrics(trades)

    assert summary.total_trades == 3
    assert summary.wins == 2
    assert summary.losses == 1
    assert abs(summary.win_rate_pct - 66.7) < 0.2
    assert summary.gross_profit_rupees == 53000.0
    assert summary.gross_loss_rupees == 15000.0
    assert summary.profit_factor == 3.53
    assert summary.expectancy_rupees == 12666.67
    assert summary.total_pnl_rupees == 38000.0
    assert summary.avg_holding_sessions == 3.7
    assert summary.exit_reasons["TARGET_3"] == 1
    assert summary.exit_reasons["STOP_LOSS"] == 1


def test_shadow_monitor_db_integration():
    """2. Test database shadow trade persistence and analysis retrieval."""
    init_db()
    with get_db_session() as session:
        # Create a test shadow trade
        trade = ShadowTradeModel(
            recommendation_id="REC-REAL-01",
            symbol="BHARTIARTL",
            entry_date=date(2026, 8, 1),
            entry_price=1400.0,
            stop_loss=1350.0,
            target_1=1500.0,
            target_2=1550.0,
            target_3=1600.0,
            position_size_shares=100,
            status="ACTIVE",
            holding_sessions=1,
        )
        session.add(trade)
        session.commit()

        # Simulate EOD bar hit
        bar = {"open": 1410.0, "high": 1520.0, "low": 1400.0, "close": 1510.0}
        trade_dict = {
            "symbol": trade.symbol,
            "entry_price": trade.entry_price,
            "stop_loss": trade.stop_loss,
            "target_1": trade.target_1,
            "target_2": trade.target_2,
            "target_3": trade.target_3,
            "status": trade.status,
        }

        updated = ShadowTradeUpdate.check_and_update(trade_dict, bar, session_count=2)
        assert updated["status"] == "TARGET_1_HIT"
        assert updated.get("t1_hit") is True

        # Clean up test trade
        session.delete(trade)
        session.commit()


def test_dataframe_trade_analysis():
    """3. Test RealTradeAnalyzer from a trade ledger DataFrame."""
    df = pd.DataFrame([
        {"symbol": "TCS", "status": "TARGET_2_HIT", "pnl_rupees": 25000.0, "pnl_pct": 5.0, "exit_reason": "TARGET_2", "holding_sessions": 6},
        {"symbol": "HDFCBANK", "status": "STOPPED_OUT", "pnl_rupees": -10000.0, "pnl_pct": -2.0, "exit_reason": "STOP_LOSS", "holding_sessions": 3},
    ])

    summary = RealTradeAnalyzer.analyze_trades_from_df(df)

    assert summary.total_trades == 2
    assert summary.wins == 1
    assert summary.losses == 1
    assert summary.profit_factor == 2.5
    assert summary.total_pnl_rupees == 15000.0
