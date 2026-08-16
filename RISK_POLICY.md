# Risk Management Policy, Position Sizing & Veto System
**Project**: NSE Swing Intelligence Engine (`nse_swing_ai`)  
**Core Principle**: Capital Preservation > Data Integrity > Risk Control > Return Potential

---

## 1. Absolute Risk Veto Policy

Under no circumstances can a high numerical score or attractive technical setup override a risk rejection. If a candidate triggers any single Hard Disqualifier, it is marked `REJECT` immediately, logged in the audit ledger, and barred from recommendation.

```text
               ┌─────────────────────────────────────────┐
               │    CANDIDATE WITH HIGH SCORE (e.g. 91)  │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
               ┌─────────────────────────────────────────┐
               │         HARD DISQUALIFIERS CHECK        │
               │   1. Data integrity verified?           │
               │   2. Liquidity >= ₹5 Cr ADTV?           │
               │   3. Pledging <= 20%?                   │
               │   4. No ASM/GSM Stage >= 2?             │
               │   5. No Earnings in <= 3 sessions?      │
               │   6. Risk distance <= 8.0%?             │
               │   7. Risk:Reward (T1) >= 1:1.8?         │
               │   8. Market Regime != NO_TRADE?         │
               │   9. No unresolved agent conflicts?     │
               └────────────────────┬────────────────────┘
                                    │
                       ┌────────────┴────────────┐
                       │ ALL CONDITIONS MET?     │
                       ├────────────┬────────────┤
                       ▼ (YES)      ▼ (NO: ANY SINGLE FAILURE)
             ┌────────────────┐   ┌───────────────────────────┐
             │ ISSUE APPROVAL │   │    IMMEDIATE RISK VETO    │
             │ PROCEED TO CIO │   │     STATUS: REJECT        │
             └────────────────┘   │ (Log Disqualification     │
                                  │  Reason in Ledger)        │
                                  └───────────────────────────┘
```

---

## 2. Hard Disqualification Catalog

| # | Hard Disqualifier | Exact Threshold / Condition | Rationale |
|---|---|---|---|
| **1** | **Data Integrity Failure** | Missing bars $> 3$ in 60 days, unverified prices, data freshness `STALE` | Prevents operating on corrupted or hallucinated inputs. |
| **2** | **Liquidity & Spread Risk** | 20-day ADTV $< ₹5\text{ Cr}$ OR bid-ask spread $> 0.35\%$ | Avoids market impact and illiquid slippage traps. |
| **3** | **Surveillance Measures** | Security under SEBI/NSE ASM Stage $\ge 2$ or GSM Stage $\ge 1$ | 100% margin requirements and high regulatory risk. |
| **4** | **Promoter Pledging** | Total pledged promoter shares $> 20.0\%$ | High risk of margin call liquidations on sudden drops. |
| **5** | **Imminent Binary Event** | Board Meeting for Quarterly Earnings within $\le 3$ trading sessions | Extreme overnight gap risk bypassing stop loss orders. |
| **6** | **Excessive Stop Distance** | Structural Stop Loss $> 8.0\%$ from Entry Price | Unfavorable volatility geometry for 3–15 day swing horizons. |
| **7** | **Inadequate Risk/Reward** | $\text{Reward}_{T1} / \text{Risk} < 1.8$ OR $\text{Reward}_{T2} / \text{Risk} < 2.5$ | Violates positive mathematical expectancy requirements. |
| **8** | **Circuit Limit Lock Risk** | Stock hit 5% or 10% lower circuit within past 10 sessions | High probability of trapped positions during adverse market moves. |
| **9** | **Hostile Market Regime** | `MARKET_REGIME` is `STRONG_BEAR` / `NO_TRADE` | Trend-following equity swing trades experience $> 70\%$ failure in severe bear markets. |
| **10**| **Agent Contradiction** | `FORENSIC_AGENT` = `RED_FLAG` OR `NEWS_AGENT` = `CRITICAL_NEGATIVE` | Unresolved fundamental/governance toxicity. |

---

## 3. Position Sizing Framework (Fixed Fractional Volatility Model)

Position sizing is mathematically computed based on account capital, volatility, and exact stop-loss distance.

### A. Core Sizing Equation
$$\text{Max Capital at Risk per Trade} = \text{Total Account Capital} \times \text{Risk Per Trade \%}$$

$$\text{Position Size (Shares)} = \left\lfloor \frac{\text{Max Capital at Risk}}{\text{Entry Price} - \text{Stop Loss Price}} \right\rfloor$$

$$\text{Allocated Capital} = \text{Position Size (Shares)} \times \text{Entry Price}$$

### B. Standard Parameter Constraints
- **Account Capital Base (Default)**: ₹10,00,000 (Configurable via `.env`)
- **Risk Per Trade ($\text{Risk\%}$)**:
  - `STRONG_BULL`: $1.00\%$ of Account Capital (₹10,000 max risk)
  - `BULL`: $0.75\%$ of Account Capital (₹7,500 max risk)
  - `SELECTIVE / NEUTRAL`: $0.50\%$ of Account Capital (₹5,000 max risk)
  - `DEFENSIVE`: $0.25\%$ of Account Capital (₹2,500 max risk)
  - `NO_TRADE`: $0.00\%$
- **Maximum Single Stock Allocation**: Cap at $\le 20\%$ of total account capital (prevents overconcentration in ultra-tight stop setups).
- **Maximum Portfolio Equity Heat (Total Open Risk)**: Cap at $\le 4.0\%$ of total capital across all concurrent positions.

---

## 4. Portfolio Correlation & Sector Exposure Controls

When assembling the final daily basket of 0–3 trades:
1. **Sector Diversification**: Maximum **1 stock per Sector** in a single daily batch. (e.g. If both TRENT and DMART qualify with $A+$, pick only the highest ranked).
2. **Beta Exposure**: Total portfolio weighted Beta against NIFTY 50 must not exceed $1.35$.
3. **F&O vs Non-F&O Balance**: At least 1 of the top 3 recommendations must be an F&O-listed stock (ensuring liquidity floor).

---

## 5. Execution Realism & Indian Tax/Slippage Friction Model

All trade models, targets, and backtests must factor in true net friction costs:

| Cost Component | Rate (NSE Delivery Equities) |
|---|---|
| **Brokerage** | ₹20 per order or 0.05% (whichever is lower) |
| **Securities Transaction Tax (STT)** | 0.1% on both Buy and Sell turnover |
| **Exchange Turnover Charges (NSE)** | 0.00345% |
| **GST** | 18% on (Brokerage + Exchange Charges) |
| **SEBI Turnover Charges** | ₹10 per Crore (0.0001%) |
| **Stamp Duty** | 0.015% on Buy turnover |
| **Execution Slippage Buffer** | 0.10% on Entry and 0.10% on Exit (0.20% total) |
| **Total Round-Trip Friction** | **~0.45% to 0.55% of Traded Value** |

*Rule*: A trade setup with expected gross target of $< 3.5\%$ is rejected as statistically unprofitable after accounting for friction and adverse gap slippage.
