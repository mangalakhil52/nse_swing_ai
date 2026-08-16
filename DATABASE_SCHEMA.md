# Database Schema & Relational Architecture
**Project**: NSE Swing Intelligence Engine (`nse_swing_ai`)  
**RDBMS**: PostgreSQL 16+ with TimescaleDB Extension (or SQLite/DuckDB for local analytical development)  
**ORM / Schema Layer**: SQLAlchemy 2.0 Async + Alembic Migrations

---

## 1. Entity Relationship Overview

```text
┌────────────────────────┐         ┌────────────────────────┐
│      securities        │◄───────┤      ohlcv_daily       │ (Hypertable)
│ (Canonical Master)     │         └────────────────────────┘
└───────────┬────────────┘
            │                      ┌────────────────────────┐
            ├─────────────────────┤  technical_indicators  │
            │                      └────────────────────────┘
            │                      ┌────────────────────────┐
            ├─────────────────────┤      fundamentals      │
            │                      └────────────────────────┘
            │                      ┌────────────────────────┐
            ├─────────────────────┤     news_articles      │
            │                      └────────────────────────┘
            │                      ┌────────────────────────┐
            ├─────────────────────┤   corporate_actions    │
            │                      └────────────────────────┘
            │                      ┌────────────────────────┐
            └─────────────────────┤ trade_recommendations  │
                                   └───────────┬────────────┘
                                               │
┌────────────────────────┐                     │
│       agent_runs       │                     │
└───────────┬────────────┘                     │
            │                                  │
            ├─────────────────────┐            │
            │                     │            ▼
            ▼                     ▼    ┌────────────────────────┐
┌────────────────────────┐ ┌───────────┤     shadow_trades      │
│     agent_outputs      │ │ candidate_│ (Trade Execution Ledger│
│ (Raw Agent JSON & Logs)│ │  scores   │  & Performance Tracking)│
└────────────────────────┘ └───────────┴────────────────────────┘
```

---

## 2. Table Definitions (DDL & SQLAlchemy Schemas)

### Table 1: `securities` (Canonical Master)
```sql
CREATE TABLE securities (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(30) UNIQUE NOT NULL,      -- e.g. 'TRENT', 'RELIANCE'
    company_name VARCHAR(255) NOT NULL,
    isin VARCHAR(12) UNIQUE,
    exchange VARCHAR(10) DEFAULT 'NSE',
    sector VARCHAR(100),
    industry VARCHAR(100),
    is_fno_eligible BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    asm_gsm_stage INTEGER DEFAULT 0,         -- 0 = Normal, 1-4 = ASM/GSM stages
    lot_size INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_securities_active ON securities(is_active, is_fno_eligible);
```

### Table 2: `ohlcv_daily` (TimescaleDB Hypertable)
```sql
CREATE TABLE ohlcv_daily (
    time TIMESTAMPTZ NOT NULL,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    open NUMERIC(12, 2) NOT NULL,
    high NUMERIC(12, 2) NOT NULL,
    low NUMERIC(12, 2) NOT NULL,
    close NUMERIC(12, 2) NOT NULL,
    volume BIGINT NOT NULL,
    delivery_volume BIGINT,
    delivery_pct NUMERIC(5, 2),
    vwap NUMERIC(12, 2),
    turnover_crores NUMERIC(12, 2),
    is_split_adjusted BOOLEAN DEFAULT TRUE,
    data_source VARCHAR(50) DEFAULT 'NSE_BHAVCOPY',
    PRIMARY KEY (time, security_id)
);
-- TimescaleDB Hypertable setup:
-- SELECT create_hypertable('ohlcv_daily', 'time');
```

### Table 3: `technical_indicators`
```sql
CREATE TABLE technical_indicators (
    time TIMESTAMPTZ NOT NULL,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    ema_20 NUMERIC(12, 2),
    ema_50 NUMERIC(12, 2),
    ema_200 NUMERIC(12, 2),
    sma_200 NUMERIC(12, 2),
    rsi_14 NUMERIC(6, 2),
    adx_14 NUMERIC(6, 2),
    atr_14 NUMERIC(12, 2),
    atr_pct NUMERIC(6, 2),
    mansfield_rs NUMERIC(8, 2),              -- vs Nifty 50
    sector_rs NUMERIC(8, 2),                 -- vs Sector Index
    distance_52w_high_pct NUMERIC(6, 2),
    volatility_contraction_score NUMERIC(5, 2),
    relative_volume_20d NUMERIC(6, 2),
    PRIMARY KEY (time, security_id)
);
```

### Table 4: `market_regimes`
```sql
CREATE TABLE market_regimes (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    index_symbol VARCHAR(30) DEFAULT 'NIFTY 50',
    close NUMERIC(12, 2) NOT NULL,
    trend_ema_structure VARCHAR(20),         -- BULLISH / BEARISH
    breadth_advance_decline_ratio NUMERIC(6, 2),
    pct_stocks_above_50_sma NUMERIC(5, 2),
    india_vix NUMERIC(6, 2),
    regime VARCHAR(30) NOT NULL,             -- STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR
    trading_stance VARCHAR(30) NOT NULL,     -- AGGRESSIVE, NORMAL, SELECTIVE, DEFENSIVE, NO_TRADE
    summary TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Table 5: `fundamentals`
```sql
CREATE TABLE fundamentals (
    id SERIAL PRIMARY KEY,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    period_end_date DATE NOT NULL,
    sales_growth_yoy NUMERIC(8, 2),
    pat_growth_yoy NUMERIC(8, 2),
    ebitda_margin NUMERIC(6, 2),
    roe NUMERIC(6, 2),
    roce NUMERIC(6, 2),
    debt_to_equity NUMERIC(6, 2),
    cfo_to_pat NUMERIC(6, 2),
    promoter_holding_pct NUMERIC(6, 2),
    promoter_pledging_pct NUMERIC(6, 2),
    fii_holding_pct NUMERIC(6, 2),
    dii_holding_pct NUMERIC(6, 2),
    fundamental_grade VARCHAR(20),           -- EXCEPTIONAL, STRONG, GOOD, AVERAGE, WEAK, DANGEROUS
    data_source VARCHAR(50),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (security_id, period_end_date)
);
```

### Table 6: `news_articles`
```sql
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    headline TEXT NOT NULL,
    summary TEXT,
    publisher VARCHAR(100) NOT NULL,
    source_tier INTEGER DEFAULT 2,           -- 1, 2, 3
    source_url TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    sentiment VARCHAR(20) NOT NULL,          -- POSITIVE, NEGATIVE, NEUTRAL, MIXED
    materiality_score NUMERIC(4, 2),         -- 0.0 to 1.0
    is_catalyst BOOLEAN DEFAULT FALSE,
    catalyst_type VARCHAR(50),               -- EARNINGS, ORDER_WIN, FDA, EXPANSION
    raw_extraction_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Table 7: `agent_runs` & `agent_outputs`
```sql
CREATE TABLE agent_runs (
    id VARCHAR(50) PRIMARY KEY,              -- 'RUN-2026-08-16-160000'
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status VARCHAR(30) NOT NULL,             -- RUNNING, COMPLETED, FAILED
    market_regime VARCHAR(30),
    universe_size INTEGER,
    quant_candidates_count INTEGER,
    researched_count INTEGER,
    recommended_count INTEGER,
    model_version VARCHAR(30) DEFAULT 'v1.0.0',
    log_text TEXT
);

CREATE TABLE agent_outputs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    agent_name VARCHAR(60) NOT NULL,
    status VARCHAR(30) NOT NULL,             -- SUCCESS, FAILED, DATA_UNAVAILABLE
    signal VARCHAR(30),                      -- BULLISH, BEARISH, NEUTRAL, REJECT
    score NUMERIC(5, 2),
    confidence NUMERIC(4, 2),
    disqualification_triggered BOOLEAN DEFAULT FALSE,
    disqualification_reason TEXT,
    metrics_json JSONB,
    evidence_json JSONB,
    risks_json JSONB,
    execution_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Table 8: `candidate_scores` & `trade_recommendations`
```sql
CREATE TABLE candidate_scores (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    composite_score NUMERIC(5, 2) NOT NULL,
    conviction_grade VARCHAR(10) NOT NULL,   -- A+, A, B+, B, C, REJECT
    confluence_state VARCHAR(30) NOT NULL,   -- VERY_HIGH, HIGH, MODERATE, CONFLICTED
    factor_scores_json JSONB NOT NULL,
    passed_risk_veto BOOLEAN NOT NULL,
    rejection_reasons JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE trade_recommendations (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL REFERENCES agent_runs(id),
    security_id INTEGER NOT NULL REFERENCES securities(id),
    recommendation_date DATE NOT NULL,
    conviction VARCHAR(10) NOT NULL,         -- 'A+', 'A'
    current_market_price NUMERIC(12, 2) NOT NULL,
    entry_trigger_price NUMERIC(12, 2) NOT NULL,
    stop_loss_price NUMERIC(12, 2) NOT NULL,
    risk_percentage NUMERIC(5, 2) NOT NULL,
    target_1 NUMERIC(12, 2) NOT NULL,
    target_2 NUMERIC(12, 2) NOT NULL,
    target_3 NUMERIC(12, 2) NOT NULL,
    risk_reward_t1 NUMERIC(5, 2) NOT NULL,
    risk_reward_t2 NUMERIC(5, 2) NOT NULL,
    position_size_shares INTEGER NOT NULL,
    holding_period_days VARCHAR(20) DEFAULT '3-15 sessions',
    technical_setup VARCHAR(100),
    invalidation_criteria TEXT,
    trade_dossier_markdown TEXT NOT NULL,
    status VARCHAR(30) DEFAULT 'ACTIVE',     -- ACTIVE, FILLED, EXPIRED, CANCELLED
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Table 9: `shadow_trades` (Paper Execution Tracking)
```sql
CREATE TABLE shadow_trades (
    id SERIAL PRIMARY KEY,
    recommendation_id INTEGER NOT NULL REFERENCES trade_recommendations(id),
    security_id INTEGER NOT NULL REFERENCES securities(id),
    entry_date DATE NOT NULL,
    entry_price NUMERIC(12, 2) NOT NULL,
    exit_date DATE,
    exit_price NUMERIC(12, 2),
    exit_reason VARCHAR(50),                 -- TARGET_1_HIT, TARGET_2_HIT, STOP_LOSS_HIT, TIME_STOP, TRAILING_STOP
    pnl_percentage NUMERIC(6, 2),
    pnl_rupees NUMERIC(12, 2),
    max_adverse_excursion_pct NUMERIC(6, 2), -- Worst drawdown during trade
    max_favorable_excursion_pct NUMERIC(6, 2),-- Peak profit during trade
    holding_sessions INTEGER,
    status VARCHAR(30) DEFAULT 'OPEN'        -- OPEN, CLOSED
);
```

### Table 10: `model_versions` & `backtest_runs`
```sql
CREATE TABLE model_versions (
    version VARCHAR(30) PRIMARY KEY,         -- 'v1.0.0'
    scoring_weights_json JSONB NOT NULL,
    risk_parameters_json JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE backtest_runs (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(30) NOT NULL REFERENCES model_versions(version),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital NUMERIC(15, 2) DEFAULT 1000000.00,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate_pct NUMERIC(5, 2),
    profit_factor NUMERIC(6, 2),
    expectancy_r NUMERIC(6, 2),
    max_drawdown_pct NUMERIC(5, 2),
    sharpe_ratio NUMERIC(6, 2),
    sortino_ratio NUMERIC(6, 2),
    cagr_pct NUMERIC(6, 2),
    attribution_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```
