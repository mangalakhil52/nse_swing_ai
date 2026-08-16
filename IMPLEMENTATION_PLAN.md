# Step-by-Step Implementation Roadmap
**Project**: NSE Swing Intelligence Engine (`nse_swing_ai`)  
**Methodology**: Phased, Test-Driven Development (TDD), Strict Module Decoupling

---

## 1. Phased Development Roadmap

```text
 PHASE 0          PHASE 1          PHASE 2          PHASE 3
Architecture  ►  Data Layer    ►  Quant & RS    ►  Specialist
& Base Models    & Validation     Screener         Agents
 (Day 1)          (Day 2)          (Day 3)          (Day 4)
    │                │                │                │
    ▼                ▼                ▼                ▼
 PHASE 4          PHASE 5          PHASE 6          PHASE 7
Confluence &  ►  Backtest &    ►  Shadow Mode   ►  Production
Trade Sizing     Validation       & Alerting       CLI & Hardening
 (Day 5)          (Day 6)          (Day 7)          (Day 8)
```

---

## 2. Detailed Work Breakdown Structure (WBS)

### Phase 0: Project Setup, Core Types & Pydantic Data Models
- [x] Create project structure and environment configuration (`pyproject.toml`, `.env.example`, `config/settings.py`).
- [ ] Implement core domain enums (`MarketRegime`, `ConvictionGrade`, `AgentStatus`, `PatternType`, `TradeStatus`) in `src/core/types.py`.
- [ ] Implement standard Pydantic models and JSON contracts for all agent outputs and evidence artifacts in `src/core/models.py`.
- [ ] Implement evidence model and graph tracer in `src/core/evidence.py`.

### Phase 1: Data Ingestion Layer & Quality Engine
- [ ] Define provider abstract base classes (`MarketDataProvider`, `FundamentalProvider`, `NewsProvider`, `CorporateActionsProvider`) in `src/data/base.py`.
- [ ] Implement NSE Bhavcopy & live quote provider in `src/data/nse_provider.py`.
- [ ] Implement Yahoo Finance fallback provider in `src/data/yfinance_provider.py`.
- [ ] Implement Screener/Filings fundamental data provider in `src/data/fundamental_provider.py`.
- [ ] Implement News and RSS feed scraper in `src/data/news_provider.py`.
- [ ] Implement strict OHLCV validation and corporate actions checker in `src/data/validation.py`.
- [ ] Implement dynamic canonical universe builder in `src/data/universe.py`.
- [ ] Implement asynchronous database engine, migrations, and repository in `src/database/`.

### Phase 2: Quantitative Engine, Pattern Recognition & Stage-1 Screener
- [ ] Vectorized indicator computation library (EMA, SMA, RSI, ATR, MACD, ADX, Bollinger Bands, Volume SMA) in `src/quant/indicators.py`.
- [ ] Mansfield Relative Strength and comparative alpha vs Nifty 50 and Sector indices in `src/quant/relative_strength.py`.
- [ ] Deterministic pattern detection algorithms (VCP, Breakout from Flat Base, Pullback to 20 EMA, High Tight Flag) in `src/quant/patterns.py`.
- [ ] Nifty 50 / 500 Market Regime & Breadth Classifier in `src/quant/regime.py`.
- [ ] High-speed Stage-1 Universe Screener (2,200 stocks $\rightarrow$ 50–100 candidates) in `src/quant/screener.py`.

### Phase 3: Parallel Domain Specialist Research Agents
- [ ] Base Agent framework with telemetry, timeout handling, and JSON validation in `src/agents/base_agent.py`.
- [ ] `TechnicalAnalysisAgent` in `src/agents/technical_agent.py`.
- [ ] `FundamentalAnalysisAgent` in `src/agents/fundamental_agent.py`.
- [ ] `NewsIntelligenceAgent` with anti-hallucination verification in `src/agents/news_agent.py`.
- [ ] `CatalystAgent` in `src/agents/catalyst_agent.py`.
- [ ] `SectorRotationAgent` in `src/agents/sector_agent.py`.
- [ ] `InstitutionalFlowAgent` (Delivery %, FII/DII flow) in `src/agents/institutional_agent.py`.
- [ ] `ForensicAnalysisAgent` (Promoter pledge, auditor flags, debt spikes) in `src/agents/forensic_agent.py`.
- [ ] `RiskManagementAgent` (Hard Veto & Risk Guard) in `src/agents/risk_agent.py`.

### Phase 4: Confluence, Scoring, Risk Veto & Trade Construction
- [ ] Cross-agent disagreement detector and confluence engine in `src/agents/confluence_agent.py`.
- [ ] Calibrated 100-point factor scoring engine in `src/agents/quant_score_agent.py`.
- [ ] Hard Disqualifiers Veto processor in `src/risk/veto.py`.
- [ ] Trade Construction engine (Entry, Stop Loss, T1, T2, T3, R:R calculation) in `src/agents/trade_construction_agent.py`.
- [ ] Fixed Fractional Volatility Position Sizer & Sector Correlation Guard in `src/risk/sizing.py` and `src/risk/correlation.py`.
- [ ] Master `CioOrchestrator` & `FinalCioAgent` (A+/A/B/C/REJECT classifier and 0–3 stock selection) in `src/agents/cio_orchestrator.py`.

### Phase 5: Backtesting Engine & Walk-Forward Validation
- [ ] Realistic execution simulator (slippage, STT, brokerage, gap-through-stop) in `src/backtest/slippage.py` and `src/backtest/engine.py`.
- [ ] Quantitative performance metrics engine (Sharpe, Sortino, Expectancy R, MAE, MFE, Drawdowns) in `src/backtest/metrics.py`.
- [ ] Walk-forward sliding window optimizer in `src/backtest/walk_forward.py`.
- [ ] Agent performance attribution suite in `src/paper/attribution.py`.

### Phase 6: Shadow Mode (Paper Trading) & Alerting System
- [ ] Hypothetical trade execution and tracking ledger in `src/paper/tracker.py`.
- [ ] Real-time price tracking and exit state machine (T1 Hit, Stop Hit, Time Stop) in `src/paper/tracker.py`.
- [ ] High-conviction trade and "NO TRADE TODAY" formatted report dispatchers in `src/alerts/formatters.py` and `src/alerts/telegram.py`.

### Phase 7: CLI Operational Tools, Scheduling & Hardening
- [ ] End-to-End Daily Scan script in `scripts/run_daily_scan.py`.
- [ ] Backtest execution runner CLI in `scripts/run_backtest.py`.
- [ ] Paper trading monitor CLI in `scripts/run_shadow_monitor.py`.
- [ ] Unit and integration test suite covering all modules in `tests/`.
