# Multi-Agent Specification & Communication Contracts
**Project**: NSE Swing Intelligence Engine (`nse_swing_ai`)  
**Architecture Model**: Specialized Decoupled Domain Agents with Structured Contracts & Evidence Graph

---

## 1. Agent Hierarchy & Interaction Topology

The agent system is strictly hierarchical and role-segregated. No agent is allowed to perform generalist guessing outside its designated domain.

```text
                                  ┌────────────────────────┐
                                  │   CIO_ORCHESTRATOR     │
                                  │ (Workflow Coordinator) │
                                  └───────────┬────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │ UNIVERSE_DISCOVERY_AGENT│                       │   MARKET_REGIME_AGENT   │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │ (~2,200 Equities)                               │ (Nifty 50/500/Sector Breadth)
                     ▼                                                 │
        ┌─────────────────────────┐                                    │
        │  QUANT_SCREENING_AGENT  │                                    │
        └────────────┬────────────┘                                    │
                     │ (50–100 Candidates)                             │
                     ▼                                                 │
 ┌───────────────────────────────────────────────────────────┐         │
 │              PARALLEL SPECIALIST RESEARCH LAYER           │         │
 │ ┌───────────────────────┬───────────────────────────────┐ │         │
 │ │ TECHNICAL_AGENT       │ FUNDAMENTAL_AGENT             │ │         │
 │ ├───────────────────────┼───────────────────────────────┤ │         │
 │ │ RELATIVE_STRENGTH_AGENT│ NEWS_INTELLIGENCE_AGENT      │ │         │
 │ ├───────────────────────┼───────────────────────────────┤ │         │
 │ │ SECTOR_ROTATION_AGENT │ CATALYST_AGENT                │ │         │
 │ ├───────────────────────┼───────────────────────────────┤ │         │
 │ │ INSTITUTIONAL_AGENT   │ FORENSIC_ANALYSIS_AGENT       │ │         │
 │ └───────────────────────┴───────────────────────────────┘ │         │
 └─────────────────────────────┬─────────────────────────────┘         │
                               │ Structured Evidence Artifacts         │
                               ▼                                       │
                 ┌───────────────────────────┐                         │
                 │   CONFLUENCE_AGENT        │ ◄───────────────────────┘
                 │ (Disagreement & Synthesis)│
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │     QUANT_SCORE_AGENT     │
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │   RISK_MANAGEMENT_AGENT   │ ◄── VETO AUTHORITY
                 │  (Hard Veto & Risk Guard) │
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ TRADE_CONSTRUCTION_AGENT  │
                 │ (Entry, SL, T1-T3, Sizing)│
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │      FINAL_CIO_AGENT      │
                 │ (Classify A+/A/B/C/Reject)│
                 └─────────────┬─────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       0–3 Top Trades                   NO TRADE TODAY
               │                               │
               ▼                               ▼
        ALERT_AGENT                     ALERT_AGENT
      (Telegram/Email)                (Daily Summary)
```

---

## 2. Agent Catalog & Domain Responsibilities

### Level 0: Master Orchestrator
#### `CIO_ORCHESTRATOR`
- **Role**: Coordinates the entire workflow lifecycle, manages concurrency, tracks agent execution latencies, and enforces timeout policies.
- **Rules**: Cannot modify agent evidence, cannot inject subjective bias, must abort or degrade gracefully if critical agents fail.

---

### Level 1: Market & Universe Discovery
#### 1. `UNIVERSE_DISCOVERY_AGENT`
- **Role**: Dynamically builds the canonical NSE equity master table.
- **Filters**: Excludes delisted, suspended, SME-illiquid, penny stocks (< ₹20), and securities under ASM (Additional Surveillance Measure) / GSM (Graded Surveillance Measure) Stage $\ge 2$.
- **Output**: Array of validated `SymbolMetadata` with exchange token, ISIN, sector, industry, and F&O eligibility.

#### 2. `MARKET_REGIME_AGENT`
- **Role**: Analyzes NIFTY 50, NIFTY 500, NIFTY MIDCAP 150, BANK NIFTY, India VIX, and advance/decline market breadth.
- **Metrics**: 20/50/100/200 EMA structure, ADX(14), % of Nifty 500 stocks above 50 SMA, 10-day McClellan Oscillator.
- **Classification**:
  - `Regime`: `STRONG_BULL`, `BULL`, `NEUTRAL`, `BEAR`, `STRONG_BEAR`
  - `TradingStance`: `AGGRESSIVE`, `NORMAL`, `SELECTIVE`, `DEFENSIVE`, `NO_TRADE`

---

### Level 2: Quantitative Screening & Technical Analysis
#### 3. `QUANT_SCREENING_AGENT`
- **Role**: Deterministic stage-1 screening across universe.
- **Criteria**:
  1. ADTV (20-day Average Daily Traded Value) $\ge$ ₹5 Crore.
  2. Price $>$ 20 EMA $>$ 50 EMA $>$ 200 EMA (for bullish trend).
  3. Distance from 52-week High $\le 20\%$.
  4. 20-day ATR% between $1.5\%$ and $7.0\%$ (filters dead and extreme-volatility lottery stocks).
  5. 20-day Relative Volume Ratio $\ge 1.0$.
- **Output**: Shortlist of 40–100 candidate symbols with pre-calculated indicators.

#### 4. `TECHNICAL_ANALYSIS_AGENT`
- **Role**: Validates high-probability structural setups using deterministic pattern recognition.
- **Patterns Detected**:
  - *Breakout from Consolidation* (Range contraction for $\ge 7$ bars, breakout candle body $> 60\%$ of bar, volume $\ge 1.5\times$ 20 SMA).
  - *Volatility Contraction Pattern (VCP)* (Consecutive contracting contractions: e.g. 12% $\rightarrow$ 6% $\rightarrow$ 2.5% with decreasing volume).
  - *Pullback to 20/50 EMA with Reversal Hammer/Engulfing*.
  - *High Tight Flag & Flat Base*.
- **Multi-Timeframe**: Daily primary swing structure + Weekly trend confirmation + 1H intraday breakout confirmation.

#### 5. `RELATIVE_STRENGTH_AGENT`
- **Role**: Calculates Mansfield Relative Strength and comparative alpha against NIFTY 50 and benchmark Sector Index.
- **Metrics**:
  $$RS_{Stock, Benchmark}(t) = \left(\frac{Price_{Stock}(t) / Price_{Benchmark}(t)}{SMA\left(Price_{Stock}/Price_{Benchmark}, 50\right)} - 1\right) \times 100$$
- **Percentile Score**: Top decile ($> 85$th percentile) relative strength ranking.

---

### Level 3: Fundamental, Sector & Intelligence Analysis
#### 6. `FUNDAMENTAL_ANALYSIS_AGENT`
- **Role**: Evaluates financial health, earnings trajectory, and solvency to prevent entering fundamentally deteriorating traps.
- **Metrics**:
  - TTM Sales Growth YoY $> 10\%$
  - TTM PAT Growth YoY $> 15\%$
  - ROE $> 12\%$, ROCE $> 15\%$
  - Debt-to-Equity $< 1.5$ (or Net Debt/EBITDA $< 3.0$, relaxed for BFSI/NBFCs)
  - CFO / PAT ratio $> 0.7$ (Cash flow earnings quality check)
- **Classification**: `EXCEPTIONAL`, `STRONG`, `GOOD`, `AVERAGE`, `WEAK`, `DANGEROUS`.

#### 7. `SECTOR_ROTATION_AGENT`
- **Role**: Analyzes the 14 major NSE Sector Indices (Nifty IT, Nifty Auto, Nifty Pharma, Nifty FMCG, Nifty Metal, etc.).
- **Evaluation**: Identifies top 3 leading sectors and lagging sectors over 5D, 20D, and 60D rolling windows.

#### 8. `INSTITUTIONAL_FLOW_AGENT`
- **Role**: Evaluates smart money footprints.
- **Data Tracked**:
  - Official NSE Delivery Percentage vs 20-day Average Delivery (e.g. Delivery % $> 60\%$ with volume spike).
  - Net FII / DII monthly holding changes from quarterly BSE/NSE shareholding patterns.
  - Institutional Block/Bulk deals logged on NSE in the past 14 days.

#### 9. `NEWS_INTELLIGENCE_AGENT`
- **Role**: Ingests and processes financial news from verified Tier 1/2 publishers and NSE corporate announcements.
- **Anti-Hallucination Constraints**: Every news item must carry an exact URL, publication timestamp, source tier, and sentiment score with explicit extraction reasoning.

#### 10. `CATALYST_AGENT`
- **Role**: Identifies forward-looking positive catalysts that can trigger momentum during the 3–15 day holding period.
- **Categories**: Confirmed earnings date within window, substantial order wins ($> 10\%$ annual revenue), capacity commissioning, regulatory clearances, FDA approvals.

---

### Level 4: Defense, Red Flag & Risk Management
#### 11. `FORENSIC_ANALYSIS_AGENT`
- **Role**: Actively attempts to kill the trade thesis by finding corporate governance and accounting red flags.
- **Checks**:
  - Promoter Pledging $> 15\%$ (Immediate Disqualifier if $> 20\%$).
  - Promoter Holding reduction $> 2\%$ over last 2 quarters (excluding OFS/PE exits).
  - Auditor qualifications / resignations within 12 months.
  - SEBI investigation / ASM Stage $\ge 2$ / GSM inclusion.
  - Contingent liabilities $> 50\%$ of Net Worth.
- **Verdict**: `CLEAN`, `MINOR_CONCERN`, `RED_FLAG_REJECT`.

#### 12. `RISK_MANAGEMENT_AGENT` (Absolute Veto Power)
- **Role**: Evaluates event risk, gap risk, volatility risk, and stop-loss viability.
- **Hard Veto Rules**:
  - If Earnings announcement is scheduled in $\le 3$ trading sessions (Gap Risk).
  - If required structural stop loss is $> 8\%$ from entry.
  - If Risk-to-Reward ratio to Target 1 is $< 1:1.8$ or Target 2 $< 1:2.5$.
  - If 5-day daily average spread $> 0.25\%$.
- **Action**: Issues `APPROVED`, `CAUTION`, or `REJECT`. A `REJECT` terminates candidate processing immediately.

---

### Level 5: Synthesis, Trade Construction & Final Decision
#### 13. `CONFLUENCE_AGENT`
- **Role**: Evaluates cross-agent alignment and detects logical contradictions.
- **Disagreement Matrix**:
  - If Technical is `STRONG_BUY` but News is `CRITICAL_NEGATIVE` $\rightarrow$ State: `CONFLICTED` $\rightarrow$ Reject.
  - If Sector is `LAGGING` and Market is `DEFENSIVE` $\rightarrow$ Penalize score by 15 points.

#### 14. `QUANT_SCORE_AGENT`
- **Role**: Computes the calibrated 100-point composite score with explicit factor attribution.

#### 15. `TRADE_CONSTRUCTION_AGENT`
- **Role**: Determines precise execution plan:
  - `Entry Price`: Limit order at breakout trigger or limit pullback to 9/20 EMA.
  - `Stop Loss`: Anchored below swing low / structural support / ATR-buffer.
  - `Target 1` (Partial booking 50%): $1.5 \times \text{Risk}$ or immediate resistance.
  - `Target 2` (Trailing balance 30%): $2.5 \times \text{Risk}$ or measured move.
  - `Target 3` (Runner 20%): $4.0 \times \text{Risk}$ / 20 EMA trail.
  - `Position Size`: Calculated using Fixed Fractional Volatility formula:
    $$\text{Shares} = \left\lfloor \frac{\text{Account Capital} \times \text{Risk Per Trade \%}}{\text{Entry Price} - \text{Stop Loss Price}} \right\rfloor$$

#### 16. `FINAL_CIO_AGENT`
- **Role**: Reviews the candidate dossier, evaluates cross-candidate correlations (ensuring no multi-stock concentration in the same sector), applies final conviction grade (`A+`, `A`, `B+`, `B`, `C`, `REJECT`), and outputs the final 0–3 recommendations.

#### 17. `ALERT_AGENT`
- **Role**: Dispatches formatted markdown reports to Telegram and internal logs.

#### 18. `BACKTEST_AGENT`
- **Role**: Simulates historical performance with zero look-ahead bias, accounting for slippage, liquidity limits, and Indian tax/statutory costs.

---

## 3. Standardized Agent JSON Output Contract

Every agent communicates via a strict Pydantic-validated JSON contract:

```json
{
  "agent_name": "technical_analysis_agent",
  "symbol": "TRENT",
  "run_id": "RUN-2026-08-16-160000",
  "timestamp": "2026-08-16T16:05:32+05:30",
  "status": "SUCCESS",
  "signal": "BULLISH",
  "score": 88.5,
  "confidence": 0.92,
  "data_freshness": "RECENT",
  "metrics": {
    "pattern_detected": "VOLATILITY_CONTRACTION_PATTERN",
    "pattern_quality_score": 90.0,
    "consolidation_days": 18,
    "relative_volume_breakout": 2.45,
    "rsi_14": 62.4,
    "adx_14": 28.6,
    "distance_from_52w_high_pct": 2.1
  },
  "evidence": [
    {
      "metric_name": "20_day_vcp_compression",
      "observed_value": "12.4% -> 5.8% -> 2.1%",
      "unit": "range_contraction",
      "source": "NSE_BHAVCOPY_EOD",
      "timestamp": "2026-08-16T15:30:00+05:30",
      "verification_status": "VERIFIED"
    },
    {
      "metric_name": "breakout_volume_surge",
      "observed_value": 2.45,
      "unit": "ratio_to_20_sma",
      "source": "NSE_LIVE_TRADE_TICK",
      "timestamp": "2026-08-16T15:30:00+05:30",
      "verification_status": "VERIFIED"
    }
  ],
  "risks_identified": [
    "Immediate overhead resistance at ₹7,250 (historical swing high 45 days ago)"
  ],
  "disqualification_triggered": false,
  "disqualification_reason": null
}
```
