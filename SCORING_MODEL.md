# 100-Point Quantitative Scoring Model & Decision Framework
**Project**: NSE Swing Intelligence Engine (`nse_swing_ai`)  
**Model Version**: `v1.0.0` (Production Baseline)  
**Philosophy**: Objective, mathematically verifiable factor scoring with non-linear penalties and risk veto priority.

---

## 1. Score Allocation Breakdown (Total: 100 Points)

```text
┌────────────────────────────────────────────────────────┬────────┐
│ DIMENSION / FACTOR                                     │ WEIGHT │
├────────────────────────────────────────────────────────┼────────┤
│ 1. Technical Setup Structure & Pattern Quality         │ 20 pts │
│ 2. Relative Strength (Stock vs Nifty 50 & Sector)      │ 15 pts │
│ 3. Asymmetric Risk / Reward Geometry                   │ 15 pts │
│ 4. Market Regime & Macro Environment                   │ 10 pts │
│ 5. Volume Profile, Accumulation & Delivery Surge       │ 10 pts │
│ 6. Momentum Acceleration & Trend Strength              │ 10 pts │
│ 7. Fundamental Quality & Solvency Baseline             │ 10 pts │
│ 8. Catalysts, Corporate News & Events                  │  5 pts │
│ 9. Sector Strength & Industry Leadership               │  5 pts │
├────────────────────────────────────────────────────────┼────────┤
│ TOTAL COMPOSITE SCORE                                  │100 pts │
└────────────────────────────────────────────────────────┴────────┘
```

---

## 2. Factor Mathematical Formulations

### 1. Technical Setup Structure & Pattern Quality (Max: 20 pts)
Evaluates clarity of the chart setup using objective geometry:
- **Consolidation Duration & Tightness (8 pts)**:
  - Tight range ($< 5\%$ variance between High and Low for $\ge 10$ sessions): **8 pts**
  - Moderate range ($5-8\%$ variance for $\ge 7$ sessions): **5 pts**
  - Wide range ($> 8\%$ variance): **2 pts**
- **Pattern Match Quality (8 pts)**:
  - Volatility Contraction Pattern (VCP) with $\ge 2$ contracting cycles: **8 pts**
  - Flat Base / Cup & Handle / High Tight Flag: **7 pts**
  - 20 EMA Dynamic Pullback with reversal confirmation: **6 pts**
  - Unstructured Breakout: **3 pts**
- **Resistance Overhang Distance (4 pts)**:
  - Clear air / All-Time High / 52W High within 2%: **4 pts**
  - Overhead resistance $> 5\%$ away: **3 pts**
  - Heavy overhead resistance $< 3\%$ away: **0 pts**

---

### 2. Relative Strength (Max: 15 pts)
Calculates alpha against benchmark indices:
- **Mansfield Relative Strength vs NIFTY 50 (8 pts)**:
  $$RS_{50}(t) = \left(\frac{P_{stock}(t) / P_{Nifty}(t)}{SMA(P_{stock}/P_{Nifty}, 50)} - 1\right) \times 100$$
  - $RS > +5.0\%$ with positive slope: **8 pts**
  - $RS \in [0.0\%, +5.0\%]$: **5 pts**
  - $RS < 0.0\%$ (underperforming benchmark): **0 pts**
- **Relative Strength vs Sector Index (4 pts)**:
  - Stock outperforming Sector index by $> 3\%$: **4 pts**
  - Stock in line with Sector index ($\pm 3\%$): **2 pts**
  - Underperforming Sector: **0 pts**
- **3-Month & 6-Month RS Percentile Rank (3 pts)**:
  - Universe Percentile $\ge 90$th: **3 pts**
  - Universe Percentile $75\text{th}-89\text{th}$: **2 pts**
  - Universe Percentile $< 75$th: **0 pts**

---

### 3. Asymmetric Risk / Reward Geometry (Max: 15 pts)
Evaluates trade efficiency and risk symmetry:
$$\text{Risk} = \text{Entry} - \text{Stop Loss}, \quad \text{Reward}_{T1} = \text{Target 1} - \text{Entry}, \quad \text{Reward}_{T2} = \text{Target 2} - \text{Entry}$$
- **R:R to Target 1 (7 pts)**:
  - $\text{Reward}_{T1} / \text{Risk} \ge 2.0$: **7 pts**
  - $\text{Reward}_{T1} / \text{Risk} \in [1.5, 2.0)$: **4 pts**
  - $\text{Reward}_{T1} / \text{Risk} < 1.5$: **0 pts** (May trigger risk review)
- **R:R to Target 2 (5 pts)**:
  - $\text{Reward}_{T2} / \text{Risk} \ge 3.0$: **5 pts**
  - $\text{Reward}_{T2} / \text{Risk} \in [2.2, 3.0)$: **3 pts**
  - $\text{Reward}_{T2} / \text{Risk} < 2.2$: **1 pt**
- **Stop Loss Tightness & Structural Invalidation (3 pts)**:
  - Risk distance $\le 3.5\%$ with clean structural floor (swing low/EMA): **3 pts**
  - Risk distance $3.6\% - 6.0\%$: **2 pts**
  - Risk distance $> 6.0\%$: **0 pts**

---

### 4. Market Regime & Macro Environment (Max: 10 pts)
Determined by `MARKET_REGIME_AGENT`:
- `STRONG_BULL` (Nifty $>$ 20/50 EMA, Breadth $> 65\%$, VIX $< 15$): **10 pts**
- `BULL` (Nifty $>$ 50 EMA, Breadth $> 55\%$, VIX $< 18$): **8 pts**
- `NEUTRAL` (Nifty oscillating around 50 EMA, mixed breadth): **5 pts**
- `BEAR` (Nifty $<$ 50 EMA, Breadth $< 45\%$): **2 pts**
- `STRONG_BEAR` (Nifty $<$ 200 EMA, Breadth $< 30\%$, VIX $> 22$): **0 pts**

---

### 5. Volume Profile & Delivery Surge (Max: 10 pts)
Evaluates institutional accumulation signatures:
- **Relative Volume on Breakout/Contraction (5 pts)**:
  $$RVol_{20} = \frac{\text{Volume}_{current}}{SMA(\text{Volume}, 20)}$$
  - Breakout $RVol \ge 2.5\times$: **5 pts**
  - Breakout $RVol \in [1.5, 2.5)\times$: **3 pts**
  - Low volume contraction prior to breakout ($< 0.6\times$ 20 SMA): **5 pts**
- **Delivery Percentage & Quantity Trend (5 pts)**:
  - Delivery $\%$ $> 55\%$ AND Delivery Volume $> 1.8\times$ 20-day average: **5 pts**
  - Delivery $\%$ $> 40\%$: **3 pts**
  - Low delivery / purely speculative intraday churning: **1 pt**

---

### 6. Momentum Acceleration & Trend Strength (Max: 10 pts)
- **ADX(14) Trend Strength (4 pts)**:
  - $ADX \ge 25$ with $+DI > -DI$: **4 pts**
  - $ADX \in [20, 25)$: **2 pts**
  - $ADX < 20$ (choppy/trendless): **0 pts**
- **RSI(14) Swing Alignment (4 pts)**:
  - RSI between $58$ and $72$ (strong momentum, not overbought): **4 pts**
  - RSI between $50$ and $58$: **2 pts**
  - RSI $> 80$ (extended) or $< 45$ (weak): **0 pts**
- **Moving Average Stacking (2 pts)**:
  - Price $> 20 \text{ EMA} > 50 \text{ EMA} > 200 \text{ EMA}$ with positive slopes: **2 pts**

---

### 7. Fundamental Quality & Solvency Baseline (Max: 10 pts)
Evaluates financial stability and earnings growth:
- **TTM Sales & PAT Growth (4 pts)**:
  - PAT YoY $> 20\%$ AND Sales YoY $> 15\%$: **4 pts**
  - PAT YoY $> 10\%$ AND Sales YoY $> 8\%$: **2 pts**
  - Negative earnings growth: **0 pts**
- **Profitability / Return Ratios (3 pts)**:
  - $ROE \ge 15\%$ AND $ROCE \ge 18\%$: **3 pts**
  - $ROE \ge 10\%$: **1 pt**
- **Balance Sheet Health & Cash Flow (3 pts)**:
  - Debt/Equity $< 0.8$ AND $CFO / PAT > 0.8$: **3 pts**
  - Debt/Equity $< 1.5$: **1 pt**
  - High debt ($> 2.5$) or negative operating cash flow: **0 pts**

---

### 8. Catalysts, Corporate News & Events (Max: 5 pts)
- High-impact positive catalyst (Capacity expansion, major order win, positive regulatory ruling): **5 pts**
- Moderate positive corporate announcement: **3 pts**
- Clean news feed, no near-term binary risk: **2 pts**
- Unverified rumours or negative press: **0 pts**

---

### 9. Sector Strength & Industry Leadership (Max: 5 pts)
- Stock belongs to top 3 leading sectors by 20-day momentum: **5 pts**
- Stock belongs to middle-tier neutral sector: **3 pts**
- Stock belongs to bottom 3 lagging sectors: **0 pts**

---

## 3. Conviction Grading & Action Matrix

| Grade | Min Score | Confluence State | Min R:R (T1) | Actionable Policy |
|---|---|---|---|---|
| **A+** | $\ge 88.0$ | `VERY_HIGH` | $\ge 1:2.0$ | **Primary Swing Trade Pick** (Immediate Alert) |
| **A** | $\ge 80.0$ | `HIGH` | $\ge 1:1.8$ | **Actionable Swing Trade Pick** (Immediate Alert) |
| **B+** | $\ge 72.0$ | `MODERATE` | $\ge 1:1.5$ | Watchlist only; Actionable ONLY in `STRONG_BULL` regime |
| **B** | $60.0 - 71.9$ | `MODERATE` / `LOW` | Any | Reject from daily actionable basket |
| **C** | $< 60.0$ | `LOW` | Any | Immediate Rejection |
| **REJECT** | Any | `CONFLICTED` / VETO | Any | **Immediate Disqualification** (Risk Veto / Red Flag) |

---

## 4. Final Recommendation Allocation Rule

At the conclusion of each daily research cycle:
- **Maximum Recommendations**: **3 Stocks**.
- If 5 stocks score $\ge 80$ (`A` / `A+`), the CIO selects only the **top 2–3** with highest composite scores and lowest mutual sector/factor correlation.
- If 0 stocks meet the $\ge 80$ threshold (or if market regime is `STRONG_BEAR` / `NO_TRADE`), the system emits:

```text
==================================================
                 NO TRADE TODAY
==================================================
Capital Preservation Priority Enforced.
Reason: 0 out of 2,146 scanned securities satisfied 
        the minimum A-grade criteria (>= 80.0 pts)
        and risk clearance.
==================================================
```
