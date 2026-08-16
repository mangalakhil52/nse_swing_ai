# High-Level Architecture & System Design
**Project**: NSE Swing Intelligence Engine (`nse_swing_ai`)  
**Target Market**: National Stock Exchange of India (NSE Equities)  
**Time Horizon**: 3–15 Trading Sessions (Swing)  
**Core Objective**: Identify 0–3 highest-probability, risk-adjusted swing trading opportunities or explicitly output **NO TRADE TODAY**.

---

## 1. Executive Summary & Design Philosophy

The `nse_swing_ai` platform is engineered as a zero-trust, multi-agent quantitative and qualitative equity research desk for the Indian stock market. Rather than relying on monolithic LLM prompt engineering or consensus-averaging chatbots, the system operates as a **funnel pipeline** combining:
1. Deterministic high-speed quantitative data ingestion and validation.
2. Mathematical pre-screening and universe filtration (reducing ~2,200 NSE listings to ~40–100 candidates).
3. Domain-specific, decoupled, specialized research agents operating asynchronously.
4. An evidence graph capturing all empirical facts with exact timestamps and primary sources.
5. Deterministic confluence scoring, strict risk veto engines, and portfolio correlation controls.
6. A final Chief Investment Officer (CIO) agent enforcing capital preservation.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ENTIRE NSE UNIVERSE                             │
│                     (~2,200+ listed equities)                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION & QUALITY ENGINE                     │
│  - Corporate Actions (Splits, Bonuses, Mergers, Delistings)            │
│  - Anomaly detection, zero-volume filter, split adjustment checks      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 CANONICAL UNIVERSE & QUANT PRE-SCREEN                  │
│  - Liquidity filter (> ₹5-10 Cr ADTV, min volume, price > ₹20)         │
│  - Trend & momentum ranking, distance from 52W High, ATR filters      │
│  - Filter down to 50–200 Focus Candidates                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   PARALLEL SPECIALIST RESEARCH LAYER                   │
│  ┌───────────────────────┬──────────────────────┬────────────────────┐ │
│  │ TECHNICAL AGENT       │ FUNDAMENTAL AGENT    │ NEWS AGENT         │ │
│  │ (Deterministic Rules) │ (Growth, Margins, CF)│ (Materiality, Cat) │ │
│  ├───────────────────────┼──────────────────────┼────────────────────┤ │
│  │ RELATIVE STRENGTH     │ INSTITUTIONAL FLOW   │ SECTOR ROTATION    │ │
│  │ (vs Nifty/Sector)     │ (FII/DII, Delivery)  │ (Leadership)       │ │
│  ├───────────────────────┼──────────────────────┼────────────────────┤ │
│  │ FORENSIC AGENT        │ REGIME AGENT         │ CATALYST AGENT     │ │
│  │ (Pledging, Dilution)  │ (Market Breadth/EMA) │ (Binary Events)    │ │
│  └───────────────────────┴──────────────────────┴────────────────────┘ │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Structured Evidence Outputs (JSON)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   CONFLUENCE & QUANT SCORING ENGINE                    │
│  - Transparent 100-pt scoring model across 9 dimensions                │
│  - Disagreement detection & conflict resolution                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  RISK VETO & TRADE CONSTRUCTION ENGINE                 │
│  - Hard Disqualifiers Check (Pledging > 20%, Binary Event < 3d, etc.)  │
│  - Structural Stop-Loss, Targets (T1, T2, T3), R:R >= 1:2.0            │
│  - Volatility-adjusted Position Sizing & Sector Correlation Check      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FINAL CIO ORCHESTRATOR                          │
│  - Classifies: A+, A, B+, B, C, REJECT                                 │
│  - Selects Top 0–3 actionable setups                                   │
│  - Emits Full Evidence Dossier OR "NO TRADE TODAY"                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       ALERTING & SHADOW LOGGING                        │
│  - Telegram / Email / JSON API Dispatcher                              │
│  - Paper-Trading / Shadow Execution Ledger & Performance Attribution   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack & Framework Choices

| Component | Technology | Rationale |
|---|---|---|
| **Programming Language** | Python 3.11+ | High performance, rich quant ecosystem (pandas, numpy, scipy), async I/O. |
| **Data Validation & Schemas** | Pydantic V2 | Strict type validation, deterministic serialization, fast JSON contracts. |
| **Database & Persistence** | PostgreSQL 16 + TimescaleDB (or SQLite/DuckDB for local zero-overhead fast analytical caching) + SQLAlchemy 2.0 Async | Efficient time-series storage for tick/candle data, ACID compliance for trade ledgers and agent logs. |
| **Technical Computing** | NumPy, Pandas, Numba, TA-Lib / Vectorized Custom Indicators | Zero-latency indicator calculation without blackbox recalculation bugs. |
| **Multi-Agent Orchestration** | Custom Asynchronous Agent Harness + Antigravity Subagents | Lightweight, deterministic state machine; no bloated agent frameworks; native async parallel execution. |
| **LLM Inference** | Gemini Models (Flash / Pro via Antigravity SDK) | High-speed structured extraction for unstructured news/filings; reasoning for synthesis. |
| **Task Scheduling** | APScheduler / Cron integration | Exact alignment with NSE market hours (IST): Pre-market (08:30 IST), Live Monitor, Post-Market EOD Scan (16:00 IST). |
| **Alerting & UI** | Telegram Bot API, Structured Markdown Reports, CLI Dashboard | Real-time actionable notifications with complete setup rationale. |

---

## 3. Directory Structure

```text
nse_swing_ai/
├── ARCHITECTURE.md
├── AGENTS.md
├── DATA_SOURCES.md
├── DATABASE_SCHEMA.md
├── SCORING_MODEL.md
├── RISK_POLICY.md
├── BACKTESTING_PLAN.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py           # System settings, thresholds, API keys, paths
│   ├── market_hours.py       # NSE trading sessions, holidays calendar
│   └── scoring_weights.json  # Calibrated model weights with model_version
├── src/
│   ├── __init__.py
│   ├── core/                 # Core engine primitives
│   │   ├── types.py          # Enums (Regime, Conviction, Signal, etc.)
│   │   ├── models.py         # Pydantic data models & agent contracts
│   │   ├── evidence.py       # Evidence graph & citation tracking
│   │   └── exceptions.py     # Custom domain exceptions
│   ├── data/                 # Ingestion & Data Providers
│   │   ├── base.py           # Provider abstract base classes (ABCs)
│   │   ├── nse_provider.py   # Primary NSE live & EOD data provider
│   │   ├── yfinance_provider.py # Secondary fallback market provider
│   │   ├── fundamental_provider.py # Screener / Exchange filings provider
│   │   ├── news_provider.py  # RSS, Google News, Moneycontrol, Exchange announcements
│   │   ├── validation.py     # OHLCV & Corporate Action data integrity validator
│   │   └── universe.py       # Dynamic NSE universe discovery engine
│   ├── database/             # Storage Layer
│   │   ├── connection.py     # Async database session manager
│   │   ├── schema.py         # SQLAlchemy models
│   │   └── repository.py     # Data access queries & bulk persistence
│   ├── quant/                # Quantitative & Technical Engines
│   │   ├── indicators.py     # Vectorized technical indicator computations
│   │   ├── relative_strength.py # Mansfied RS, Stock vs Nifty/Sector RS
│   │   ├── patterns.py       # Deterministic chart pattern detection
│   │   ├── screener.py       # Stage 1: Universe to Candidate Pre-Screener
│   │   └── regime.py         # Nifty 50/500 market regime classifier
│   ├── agents/               # Multi-Agent Intelligence Layer
│   │   ├── base_agent.py     # Base agent class with metrics, timing, structured output
│   │   ├── technical_agent.py
│   │   ├── fundamental_agent.py
│   │   ├── news_agent.py
│   │   ├── catalyst_agent.py
│   │   ├── sector_agent.py
│   │   ├── institutional_agent.py
│   │   ├── forensic_agent.py
│   │   ├── risk_agent.py
│   │   ├── trade_construction_agent.py
│   │   ├── confluence_agent.py
│   │   └── cio_orchestrator.py
│   ├── risk/                 # Risk Management & Portfolio Sizing
│   │   ├── veto.py           # Hard disqualifiers evaluation
│   │   ├── sizing.py         # Volatility/ATR-based capital allocation
│   │   └── correlation.py    # Cross-asset & sector concentration matrix
│   ├── backtest/             # Backtesting & Validation Suite
│   │   ├── engine.py         # Vectorized & Event-driven backtester
│   │   ├── slippage.py       # NSE realistic execution, spread, and STT/tax model
│   │   ├── walk_forward.py   # Out-of-sample walk-forward optimizer
│   │   └── metrics.py        # Sharpe, Sortino, MAE, MFE, Expectancy calculations
│   ├── paper/                # Shadow / Paper Trading Ledger
│   │   ├── tracker.py        # Hypothetical trade tracking & price monitor
│   │   └── attribution.py    # Per-agent performance attribution analyzer
│   └── alerts/               # Notification & Reporting
│       ├── formatters.py     # Clean markdown & terminal alert formatters
│       └── telegram.py       # Telegram notification dispatcher
├── tests/                    # Comprehensive Automated Test Suite
│   ├── test_data_validation.py
│   ├── test_indicators.py
│   ├── test_patterns.py
│   ├── test_screener.py
│   ├── test_risk_veto.py
│   ├── test_trade_construction.py
│   ├── test_agents.py
│   └── test_backtest_engine.py
└── scripts/                  # Operational CLI entry points
    ├── run_daily_scan.py     # End-to-End daily swing pipeline
    ├── run_backtest.py       # Backtest execution runner
    └── run_shadow_monitor.py # Intraday shadow execution tracker
```

---

## 4. Execution Pipeline & Data Flow

```text
Time (IST)   Stage                    Description
─────────────────────────────────────────────────────────────────────────────
08:30        Pre-Market Ingestion     Fetch overnight global cues, SGX/GIFT Nifty,
                                      corporate announcements, results calendar.
15:45        Post-Market Sync         Fetch official NSE Bhavcopy, delivery stats,
                                      index EOD values, sector performance.
15:50        Data Validation          Run integrity checks: zero volume, split adjustment,
                                      missing candles.
15:55        Universe & Pre-Screen    Update active symbols. Run 8-factor deterministic
                                      quant pre-screen (2,200 -> 50-100 candidates).
16:00        Parallel Specialist      Spawn async research agents for top candidates.
             Analysis                 Compute technicals, fundamentals, news, forensics.
16:15        Confluence & Risk Veto   Aggregate evidence, detect disagreements, enforce
                                      hard disqualifiers, calculate trade levels (SL/T1/T2/T3).
16:20        CIO Final Decision       Review candidate dossiers. Classify A+/A/B+/B/C/REJECT.
                                      Select 0–3 trades or trigger NO TRADE TODAY.
16:25        Dispatch & Logging       Send formatted Alert via Telegram/Logs.
                                      Record recommendation in Paper Trading Ledger.
```

---

## 5. Key System Invariants (Non-Negotiable)

1. **No Trade is a Successful Outcome**: Zero recommendations is preferred over low-conviction or risk-compromised trades.
2. **Deterministic Pattern Rules**: Patterns are identified via strict mathematical equations (e.g. consolidation within 4% range for $\ge 7$ bars with volume contraction $< 0.6\times$ 20 SMA), not subjective LLM vision.
3. **No Hallucinated Evidence**: Every single claim (P/E ratio, revenue growth, order win, RSI) must carry a verified data source timestamp and primary URL/metric key.
4. **Absolute Risk Veto**: The `RISK_AGENT` and `FORENSIC_AGENT` have complete veto authority. High quantitative scores cannot overturn a risk disqualifier.
5. **Execution Realism**: All backtests and recommendations account for Indian market realities: circuit limits (5%, 10%, 20%), STT (0.1% on delivery), exchange fees, GST, SEBI turnover charges, and bid-ask slippage.
