# Point-In-Time (PIT) Data Source Inventory & Leakage Audit Report (Corrected)

**Repository**: `nse_swing_ai`  
**Document**: `docs/PIT_DATA_SOURCE_INVENTORY.md`  
**Scope**: Complete repository-wide data flow audit tracing provider $\rightarrow$ filter $\rightarrow$ consumer paths for point-in-time safety, lookahead bias, availability timestamps, and system architecture.

---

## 1. System Architecture & Component Status

The system architecture contains a mix of **ACTIVE** production components and **SCAFFOLDED** multi-agent modules:

| Component / Layer | Module / File | Operational Status | Description |
| :--- | :--- | :--- | :--- |
| **Portfolio Backtest Engine** | `src/backtest/portfolio.py` | **ACTIVE** | Executes chronological backtests date-by-date ($t \in [\text{eval\_start}, \text{eval\_end}]$). |
| **Pattern Recognizer** | `src/quant/patterns.py` | **ACTIVE** | Evaluates price patterns (VCP, Cup & Handle, Double Bottom) on trailing bars. |
| **Technical Indicators** | `src/quant/indicators.py` | **ACTIVE** | Calculates SMA, EMA, RSI, ATR on trailing price series. |
| **Trade Construction Engine** | `src/agents/trade_construction_agent.py` | **ACTIVE** | Determines entry triggers, stop-loss levels, and R:R targets. |
| **Position Sizing Engine** | `src/quant/position_sizing.py` | **ACTIVE** | Enforces 0.5% max risk per trade and portfolio risk budgets. |
| **Performance Analyzer** | `src/backtest/performance.py` | **ACTIVE** | Downstream analytics calculating CAGR, Sharpe, Sortino, drawdowns. |
| **Walk-Forward Validator** | `src/backtest/walk_forward.py` | **ACTIVE** | Physical train/val/test dataset slicing and window immutability verifier. |
| **Technical Agent** | `src/agents/technical_agent.py` | **SCAFFOLDED** | Agent wrapper around technical indicators and pattern recognition. |
| **Fundamental Agent** | `src/agents/fundamental_agent.py` | **SCAFFOLDED** | Earnings acceleration and cash conversion analysis; called when context supplied. |
| **News Agent** | `src/agents/news_agent.py` | **SCAFFOLDED** | Sentiment and catalyst extraction; not integrated in active backtest loop. |
| **Market Regime Classifier** | `src/quant/regime.py` | **ACTIVE / SCAFFOLDED** | Trend and breadth risk posture classifier; called by CIO Orchestrator. |

---

## 2. Complete Data Source Inventory

| Data Source | Actual File | Actual Class/Function | Consumed By | Timestamp Field | Timestamp Meaning | Availability Field | Availability Meaning | Current PIT Protection | Filter Used in Consumer Path? | Leakage Risk | Status | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OHLCV Bar Series** | `src/data/historical_provider.py` | `NseDataProvider.fetch_ohlcv()` | `PortfolioBacktestEngine`, `PatternRecognizer`, `TradeConstructionEngine` | `timestamp` | `event_time` (Market session close $T$) | `timestamp` | `availability_time` (Market close IST) | `PointInTimeFilter.filter_market_data` & `eval_end_date` boundary | **YES**: `PortfolioBacktestEngine` slices $t \le \text{current\_date}$ before calling engines | LOW / PREVENTED | `VERIFIED` | `src/backtest/portfolio.py:168`, `src/data/point_in_time.py:23` |
| **Technical Indicators** | `src/quant/indicators.py` | `calculate_sma()`, `calculate_ema()`, `calculate_rsi()`, `calculate_atr()` | `PatternRecognizer`, `TradeConstructionEngine`, `TechnicalAgent` | Indexed by `timestamp` | `event_time` ($T$) | Indexed by `timestamp` | `availability_time` ($T$) | Trailing rolling windows (`center=False`), positive shifts `shift(1)` | **YES**: Consumes $t \le T$ sliced DataFrames from `PortfolioBacktestEngine` | NONE | `VERIFIED` | `src/quant/indicators.py:22` (`rolling`), `src/quant/indicators.py:46` (`shift(1)`) |
| **Fundamental Data** | `src/data/fundamental_provider.py` | `ScreenerFundamentalProvider.get_quarterly_financials()` | `FundamentalAnalysisAgent` | `period_end_date` | `period_end` (Quarter end date e.g. 03-31) | `filing_date`, `available_at` | `publication_time` / `availability_time` | `PointInTimeFilter.filter_quarterly_financials` checks $\text{available\_at} \le T$ | **NO**: Model supports availability fields, but provider defaults them to `None` and uses fallback mock data | UNVERIFIED (Provider lacks reliable filing timestamps) | `UNVERIFIED` | `src/data/fundamental_provider.py:48`, `src/core/models.py:115` |
| **Historical Outcomes** | `src/quant/historical_setup_outcome_store.py` | `WalkForwardValidator.build_training_context()` | `WalkForwardValidator`, `ProbabilityEngine` | `setup_date`, `exit_date` | `event_time` ($T_{\text{setup}}$) | $T_{\text{setup}} + H_{\text{holding}}$ | `availability_time` (Outcome completion date) | `WalkForwardValidator.is_outcome_label_eligible` ($T_{\text{completion}} \le \text{train\_end}$) | **NO**: Label eligibility active in Walk-Forward, but full end-to-end outcome store consumer pipeline is scaffolded | UNVERIFIED (End-to-end consumer integration unverified) | `UNVERIFIED` | `src/backtest/walk_forward.py:509`, `src/quant/historical_outcome_generator.py:40` |
| **Market Regime** | `src/quant/regime.py` | `MarketRegimeClassifier.classify_regime()` | `CIOOrchestrator`, `RiskAgent` | Indexed by `timestamp` | `event_time` ($T$) | Indexed by `timestamp` | `availability_time` ($T$) | Trailing 50-day / 200-day rolling SMA on NIFTY 50 | **PARTIAL**: Classifier math is trailing, but caller pipeline must explicitly pass sliced NIFTY DataFrame | UNVERIFIED (Caller date slicing not guaranteed by provider) | `UNVERIFIED` | `src/quant/regime.py:34`, `src/agents/cio_orchestrator.py:120` |
| **News / Events** | `src/data/news_provider.py` | `FinancialNewsProvider.fetch_latest_news()` | `NewsAgent` (Scaffolded) | `published_at` | `publication_time` | `published_at` | `availability_time` | `PointInTimeFilter.filter_news` checks $\text{published\_at} \le T$ | **NO**: Filter exists, but news provider and agent are scaffolded and not connected to backtest engine | UNVERIFIED (Not integrated in active backtest loop) | `UNVERIFIED` | `src/data/news_provider.py:110`, `src/data/point_in_time.py:33` |
| **Benchmark / NIFTY** | `src/data/global_markets.py` | `PerformanceAnalyzer._compute_benchmark_metrics()` | `PerformanceAnalyzer` | `timestamp` | `event_time` ($T$) | `timestamp` | `availability_time` ($T$) | Date matching in `PerformanceAnalyzer` | **PARTIAL**: `PerformanceAnalyzer` aligns dates downstream, but upstream benchmark provider lacks explicit PIT filter | UNVERIFIED (Upstream provider unverified) | `UNVERIFIED` | `src/backtest/performance.py:615`, `src/data/global_markets.py:80` |
| **Corporate Actions** | `src/data/historical_provider.py` | `CorporateActionsProvider.fetch_actions()` | `UniverseDiscoveryEngine` | `ex_date`, `record_date` | `event_time` | UNKNOWN | UNKNOWN (Raw unadjusted log not stored) | None | **NO**: Data providers return standard backward-adjusted OHLCV series. | UNVERIFIED (Historical price series pre-adjusted) | `UNVERIFIED` | `src/data/historical_provider.py:120`, `src/core/models.py:181` |
| **Normalization / Scaling** | `src/quant/indicators.py` | `calculate_atr()`, `calculate_universe_percentile_ranks()` | `TechnicalAgent`, `RelativeStrengthAgent` | Technical metrics | Sliced per session $T$ | Sliced per session $T$ | `availability_time` ($T$) | Point-in-time calculation per bar or session | **PARTIAL**: No global fit calls exist, but custom normalization paths are not uniformly wrapped in PIT guards | UNVERIFIED (Custom normalization paths partially verified) | `UNVERIFIED` | `src/quant/relative_strength.py:59` |
| **Cross-Sectional Ranking** | `src/quant/relative_strength.py` | `RelativeStrengthEngine.calculate_universe_percentile_ranks()` | `RelativeStrengthAgent` | Relative strength scores | Sliced per session $T$ | Sliced per session $T$ | `availability_time` ($T$) | Per-session cross-sectional percentile ranking | **NO**: Rank formula is single-session, but universe membership is static across historical time (no historical index rebalancing/surveillance log) | UNVERIFIED (Static universe limitation) | `UNVERIFIED` | `src/quant/relative_strength.py:59`, `src/data/universe.py:45` |

---

## 3. Common Leakage Pattern Code Search Audit

| Pattern Searched | Search Result in `src/` | Classification | Repository Evidence & Function Reference |
| :--- | :--- | :--- | :--- |
| `shift(-N)` (Negative Shift) | **0 occurrences** | `SAFE` | Verified zero negative shifts exist across `src/`. All shifts use positive offsets (e.g. `shift(1)` in `src/quant/indicators.py:46`). |
| `rolling(center=True)` | **0 occurrences** | `SAFE` | Verified all rolling windows use default `center=False` (e.g. `src/quant/indicators.py:22`). |
| `bfill()` (Backward Fill) | **0 occurrences** | `SAFE` | Zero backward fills exist in feature engineering. |
| `ffill()` (Forward Fill) | 1 occurrence in `src/data/validation.py:112` | `SAFE` | Forward fills past known values $\le T$ forward into missing non-trading dates (`df.ffill()`). |
| `StandardScaler` / `MinMaxScaler` | **0 occurrences** | `SAFE` | Zero sklearn scaler `fit()` or `fit_transform()` calls exist in `src/`. |
| `merge_asof` / **Future Joins** | **0 occurrences** | `SAFE` | Zero `merge_asof` or future joins exist in `src/`. |
| **Future Resampling** | **0 occurrences** | `SAFE` | Zero future-aware resampling calls exist in `src/`. |
| **Custom Normalization (Rank/Percentile)** | Found in `src/quant/relative_strength.py:59` | `CONTEXT_DEPENDENT` | Ranks symbols for a single date $T$, but relies on static historical universe metadata. |

---

## 4. Scope & Limitation of `PITRegressionHelper`

- **Module**: `src/data/point_in_time.py`
- **Exact Scope**: `PITRegressionHelper.verify_future_mutation_safety()` mutates future `close` and `high` prices for rows $> \text{as\_of\_date}$ to test OHLCV price lookahead safety.
- **Limitation**: It does **NOT** mutate low, volume, fundamental records, news publication timestamps, benchmark index series, or universe membership. It is a targeted helper for price-based signal functions, not a global proof for all data sources.

---

## 5. Revised Summary Statistics

- **Total Data Sources Audited**: **10**
- **VERIFIED**: **2** (`OHLCV Bar Series`, `Technical Indicators`)
- **UNVERIFIED**: **8** (`Fundamental Data`, `Historical Outcomes`, `Market Regime`, `News / Events`, `Benchmark / NIFTY`, `Corporate Actions`, `Normalization / Scaling`, `Cross-Sectional Ranking`)
- **NOT_IMPLEMENTED**: **0**
- **LEAKAGE_DETECTED**: **0**

---

## 6. Audit Conclusion

- **Audit Corrected**: **YES**
- **Summary**: Only OHLCV bar series and trailing Technical Indicators are fully verified end-to-end from provider to active backtest execution path. Fundamentals, News, Regime, Benchmark, Outcomes, Scaling, and Cross-Sectional Ranking contain unverified availability timestamps, static historical universe limitations, or scaffolded consumer paths, requiring formal PIT layer enforcement in #12A-STEP-2.
