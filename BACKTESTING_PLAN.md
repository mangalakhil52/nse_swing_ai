# Backtesting Engine Architecture & Validation Plan
**Project**: NSE Swing Intelligence Engine (`nse_swing_ai`)  
**Core Standard**: Zero Data Leakage, True NSE Execution Friction, Walk-Forward Out-of-Sample Validation

---

## 1. Backtesting Philosophy & Anti-Bias Controls

A backtest that produces unrealistic 80% win rates on paper is useless in live trading. The backtesting engine enforces four fundamental anti-bias controls:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        ANTI-BIAS CONTROLS                              │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Zero Look-Ahead Bias: Only data available at T-0 (15:30 IST) is     │
│    used for generating signals for T+1 execution.                      │
│ 2. Point-in-Time Universe (Survivorship Bias Control): Uses historical │
│    active listings; does not backtest only current survivors.          │
│ 3. Realistic Execution & Fill Modeling: Orders are filled at Open/VWAP │
│    of T+1 with slippage, not theoretical previous-day Close.           │
│ 4. Gap-Through-Stop Logic: If stock opens below Stop Loss, fill at     │
│    Open price (worse price), not the theoretical Stop Loss price.      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Walk-Forward Validation Framework

To prevent curve-fitting and parameter overfitting, the engine implements rolling Walk-Forward Analysis across distinct market cycles (Bull, Consolidation, Correction).

```text
┌───────────────────────────────────────────────────────────────────────┐
│                    WALK-FORWARD SLIDING WINDOWS                       │
├───────────────────────────────────────────────────────────────────────┤
│ Window 1:                                                             │
│ [   TRAIN (24M)   ] -> [ VALIDATE (6M) ] -> [ OUT-OF-SAMPLE TEST (6M)]│
│                                                                       │
│ Window 2 (Rolled Forward 6 Months):                                   │
│       [   TRAIN (24M)   ] -> [ VALIDATE (6M) ] -> [ TEST (6M) ]       │
│                                                                       │
│ Window 3 (Rolled Forward 6 Months):                                   │
│             [   TRAIN (24M)   ] -> [ VALIDATE (6M) ] -> [ TEST (6M) ] │
└───────────────────────────────────────────────────────────────────────┘
```

### Rejection Thresholds for Strategy Validation:
- If Out-of-Sample Sharpe Ratio drops by $> 40\%$ compared to In-Sample $\rightarrow$ **Reject Model Configuration (Overfitted)**.
- If Win Rate variance across different regimes exceeds $25\%$ $\rightarrow$ **Flag Regime Vulnerability**.

---

## 3. Order Execution & Trade Lifecycle Simulator

### Lifecycle States:
1. **Signal Generation (T-0, EOD 16:00 IST)**: Strategy produces `EntryTrigger`, `StopLoss`, `Target1`, `Target2`, `Target3`.
2. **Order Submission (T+1 Pre-Open / Open 09:15 IST)**:
   - If stock opens within $\pm 1.0\%$ of Entry Trigger: Order filled at $\max(\text{Open}, \text{Entry Trigger}) + \text{Slippage}(0.1\%)$.
   - If stock gaps up $> 2.5\%$ above Entry Trigger: **Order Cancelled** (Chasing Penalty).
3. **Intraday Monitoring & Exit Mechanics (Holding Period: 3–15 Sessions)**:
   - **Stop-Loss Hit**: If $\text{Low} \le \text{Stop Loss} \rightarrow$ Exit at $\min(\text{Stop Loss}, \text{Open}) - \text{Slippage}$.
   - **Target 1 Reached**: Close $50\%$ position at Target 1. Move Stop Loss on remaining $50\%$ to Breakeven ($\text{Entry Price} + \text{Friction}$).
   - **Target 2 Reached**: Close $30\%$ position at Target 2. Trail remaining $20\%$ using 9 EMA Daily.
   - **Time Stop**: If after 15 trading sessions neither Target 1 nor Stop Loss is hit $\rightarrow$ Exit at Close of 15th session (Capital re-allocation).

---

## 4. Quantitative Metrics Suite

```python
@dataclass
class BacktestPerformanceReport:
    # Capital & Returns
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    benchmark_cagr_pct: float     # Nifty 50 Buy & Hold CAGR
    
    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    win_loss_ratio: float         # Average Win / Average Loss
    profit_factor: float          # Gross Profits / Gross Losses
    expectancy_r: float           # (Win Rate * Avg Win R) - (Loss Rate * 1.0)
    
    # Risk & Drawdown
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    sharpe_ratio: float           # Risk-Free Rate = 6.5% (India 10Y Yield)
    sortino_ratio: float
    calmar_ratio: float
    
    # Execution & Holding Profile
    avg_holding_sessions: float
    max_adverse_excursion_avg_pct: float  # MAE
    max_favorable_excursion_avg_pct: float # MFE
    max_consecutive_losses: int
    total_friction_paid_rupees: float     # STT + Brokerage + Slippage
```

---

## 5. Performance Attribution Matrix

The `BACKTEST_AGENT` runs comparative ablation backtests to quantify the exact value-add of each agent layer:

| Configuration | Test Focus | Target Win Rate | Target Expectancy |
|---|---|---|---|
| **Base Model** | Technical Breakout Only | $42 - 46\%$ | $+0.25\text{ R}$ |
| **+ Relative Strength** | Technical + Sector & Nifty Alpha | $48 - 52\%$ | $+0.40\text{ R}$ |
| **+ Market Regime Filter**| Trading only in `BULL` / `STRONG_BULL` | $54 - 58\%$ | $+0.65\text{ R}$ |
| **+ Fundamental & Forensic**| Eliminating low ROE, high pledge, debt spikes | $57 - 62\%$ | $+0.78\text{ R}$ |
| **Full Multi-Agent Desk**| All agents + Confluence + Risk Veto | $\mathbf{60 - 68\%}$ | $\mathbf{+0.95\text{ R}}$ |

If an individual agent's inclusion degrades the out-of-sample Expectancy ($R$), that agent is automatically flagged for weight reduction or structural refactoring.
