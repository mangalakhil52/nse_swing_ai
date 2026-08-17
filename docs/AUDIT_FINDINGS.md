# AUDIT FINDINGS — NSE SWING AI

**Total Findings Logged**: 160

| File | Line | Keyword | Severity | Finding & Impact | Required Fix |
|---|---|---|---|---|---|
| `.\config\market_hours.py` | 75 | `timedelta(days=` | MEDIUM/HIGH | `curr -= timedelta(days=1)` | Remove fallback / Use real data |
| `.\config\market_hours.py` | 106 | `timedelta(days=` | MEDIUM/HIGH | `curr = start_date + timedelta(days=1)` | Remove fallback / Use real data |
| `.\config\market_hours.py` | 110 | `timedelta(days=` | MEDIUM/HIGH | `curr += timedelta(days=1)` | Remove fallback / Use real data |
| `.\config\market_hours.py` | 117 | `timedelta(days=` | MEDIUM/HIGH | `curr = start_date - timedelta(days=1)` | Remove fallback / Use real data |
| `.\config\market_hours.py` | 121 | `timedelta(days=` | MEDIUM/HIGH | `curr -= timedelta(days=1)` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 9 | `asyncio.run` | MEDIUM/HIGH | `'asyncio.run', 'np.random', 'np.full', 'np.linspace', 'Rando` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 9 | `np.random` | MEDIUM/HIGH | `'asyncio.run', 'np.random', 'np.full', 'np.linspace', 'Rando` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 9 | `np.full` | MEDIUM/HIGH | `'asyncio.run', 'np.random', 'np.full', 'np.linspace', 'Rando` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 9 | `np.linspace` | MEDIUM/HIGH | `'asyncio.run', 'np.random', 'np.full', 'np.linspace', 'Rando` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 9 | `RandomState` | MEDIUM/HIGH | `'asyncio.run', 'np.random', 'np.full', 'np.linspace', 'Rando` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 9 | `normal(` | MEDIUM/HIGH | `'asyncio.run', 'np.random', 'np.full', 'np.linspace', 'Rando` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 10 | `500.0` | MEDIUM/HIGH | `'500.0', '800000', '55.0', 'score = 90', 'score = 88', 'adva` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 10 | `800000` | MEDIUM/HIGH | `'500.0', '800000', '55.0', 'score = 90', 'score = 88', 'adva` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 10 | `55.0` | MEDIUM/HIGH | `'500.0', '800000', '55.0', 'score = 90', 'score = 88', 'adva` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 10 | `score = 90` | MEDIUM/HIGH | `'500.0', '800000', '55.0', 'score = 90', 'score = 88', 'adva` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 10 | `score = 88` | MEDIUM/HIGH | `'500.0', '800000', '55.0', 'score = 90', 'score = 88', 'adva` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 10 | `advance_decline_ratio` | MEDIUM/HIGH | `'500.0', '800000', '55.0', 'score = 90', 'score = 88', 'adva` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 11 | `pct_above_50_sma` | MEDIUM/HIGH | `'pct_above_50_sma', 'timedelta(days=', 'FLAT_BASE_BREAKOUT',` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 11 | `timedelta(days=` | MEDIUM/HIGH | `'pct_above_50_sma', 'timedelta(days=', 'FLAT_BASE_BREAKOUT',` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 11 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `'pct_above_50_sma', 'timedelta(days=', 'FLAT_BASE_BREAKOUT',` | Remove fallback / Use real data |
| `.\scripts\audit_script.py` | 11 | `cfo_to_pat_ratio if ratios else` | MEDIUM/HIGH | `'pct_above_50_sma', 'timedelta(days=', 'FLAT_BASE_BREAKOUT',` | Remove fallback / Use real data |
| `.\scripts\run_automated_scheduler.py` | 114 | `asyncio.run` | MEDIUM/HIGH | `df_hist = asyncio.run(hist_provider.get_daily_ohlcv(sym, sta` | Remove fallback / Use real data |
| `.\scripts\run_automated_scheduler.py` | 121 | `asyncio.run` | MEDIUM/HIGH | `nifty_df = asyncio.run(hist_provider.get_daily_ohlcv("NIFTY ` | Remove fallback / Use real data |
| `.\scripts\run_automated_scheduler.py` | 124 | `500.0` | MEDIUM/HIGH | `nifty_c = float(bhav_nifty_rows.iloc[0]["close"]) if not bha` | Remove fallback / Use real data |
| `.\scripts\run_automated_scheduler.py` | 133 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio=1.65,` | Remove fallback / Use real data |
| `.\scripts\run_automated_scheduler.py` | 134 | `pct_above_50_sma` | MEDIUM/HIGH | `pct_above_50_sma=68.0,` | Remove fallback / Use real data |
| `.\scripts\run_automated_scheduler.py` | 217 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(execute_daily_5pm_cycle(now.date()))` | Remove fallback / Use real data |
| `.\scripts\run_automated_scheduler.py` | 232 | `asyncio.run` | MEDIUM/HIGH | `exit_code = asyncio.run(execute_daily_5pm_cycle(target, forc` | Remove fallback / Use real data |
| `.\scripts\run_backtest.py` | 82 | `np.random` | MEDIUM/HIGH | `np.random.seed(42)` | Remove fallback / Use real data |
| `.\scripts\run_backtest.py` | 84 | `np.random` | MEDIUM/HIGH | `close = np.cumprod(1.0 + np.random.normal(0.0008, 0.018, n))` | Remove fallback / Use real data |
| `.\scripts\run_backtest.py` | 84 | `normal(` | MEDIUM/HIGH | `close = np.cumprod(1.0 + np.random.normal(0.0008, 0.018, n))` | Remove fallback / Use real data |
| `.\scripts\run_backtest.py` | 85 | `np.random` | MEDIUM/HIGH | `high = close * (1.0 + np.abs(np.random.normal(0.005, 0.01, n` | Remove fallback / Use real data |
| `.\scripts\run_backtest.py` | 85 | `normal(` | MEDIUM/HIGH | `high = close * (1.0 + np.abs(np.random.normal(0.005, 0.01, n` | Remove fallback / Use real data |
| `.\scripts\run_backtest.py` | 86 | `np.random` | MEDIUM/HIGH | `low = close * (1.0 - np.abs(np.random.normal(0.005, 0.01, n)` | Remove fallback / Use real data |
| `.\scripts\run_backtest.py` | 86 | `normal(` | MEDIUM/HIGH | `low = close * (1.0 - np.abs(np.random.normal(0.005, 0.01, n)` | Remove fallback / Use real data |
| `.\scripts\run_backtest.py` | 88 | `np.random` | MEDIUM/HIGH | `volume = np.random.randint(200000, 1000000, n)` | Remove fallback / Use real data |
| `.\scripts\run_daily_scan.py` | 106 | `asyncio.run` | MEDIUM/HIGH | `df_hist = asyncio.run(hist_provider.get_daily_ohlcv(sym, sta` | Remove fallback / Use real data |
| `.\scripts\run_daily_scan.py` | 114 | `asyncio.run` | MEDIUM/HIGH | `nifty_df = asyncio.run(hist_provider.get_daily_ohlcv("NIFTY ` | Remove fallback / Use real data |
| `.\scripts\run_daily_scan.py` | 118 | `500.0` | MEDIUM/HIGH | `nifty_c = float(bhav_nifty_rows.iloc[0]["close"]) if not bha` | Remove fallback / Use real data |
| `.\scripts\run_daily_scan.py` | 127 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio=1.65,` | Remove fallback / Use real data |
| `.\scripts\run_daily_scan.py` | 128 | `pct_above_50_sma` | MEDIUM/HIGH | `pct_above_50_sma=68.0,` | Remove fallback / Use real data |
| `.\scripts\run_daily_scan.py` | 199 | `asyncio.run` | MEDIUM/HIGH | `exit_code = asyncio.run(run_scan(scan_date, dry_run=args.dry` | Remove fallback / Use real data |
| `.\scripts\run_premarket_news_scan.py` | 174 | `asyncio.run` | MEDIUM/HIGH | `exit_code = asyncio.run(execute_premarket_news_scan(target, ` | Remove fallback / Use real data |
| `.\src\agents\cio_orchestrator.py` | 218 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `pattern_str = tech_out.metrics.get("pattern_detected", Patte` | Remove fallback / Use real data |
| `.\src\agents\cio_orchestrator.py` | 222 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `pattern_enum = PatternType.FLAT_BASE_BREAKOUT` | Remove fallback / Use real data |
| `.\src\agents\forensic_agent.py` | 41 | `cfo_to_pat_ratio if ratios else` | MEDIUM/HIGH | `cfo_pat = ratios.cfo_to_pat_ratio if ratios else 0.9` | Remove fallback / Use real data |
| `.\src\agents\fundamental_agent.py` | 51 | `cfo_to_pat_ratio if ratios else` | MEDIUM/HIGH | `cfo_pat = ratios.cfo_to_pat_ratio if ratios else 0.88` | Remove fallback / Use real data |
| `.\src\agents\institutional_agent.py` | 51 | `55.0` | MEDIUM/HIGH | `if delivery_pct >= 55.0 and delivery_surge >= 1.2:` | Remove fallback / Use real data |
| `.\src\agents\sector_agent.py` | 42 | `score = 90` | MEDIUM/HIGH | `score = 90.0 - (rank - 1) * 5.0` | Remove fallback / Use real data |
| `.\src\agents\thesis_killer_agent.py` | 78 | `55.0` | MEDIUM/HIGH | `disqualified = fragility_score >= 55.0` | Remove fallback / Use real data |
| `.\src\core\models.py` | 104 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio: float = Field(..., ge=0.0)` | Remove fallback / Use real data |
| `.\src\core\types.py` | 44 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `FLAT_BASE_BREAKOUT = "FLAT_BASE_BREAKOUT"` | Remove fallback / Use real data |
| `.\src\data\historical_provider.py` | 96 | `timedelta(days=` | MEDIUM/HIGH | `curr += timedelta(days=1)` | Remove fallback / Use real data |
| `.\src\data\news_provider.py` | 85 | `timedelta(days=` | MEDIUM/HIGH | `cutoff = datetime.utcnow() - timedelta(days=lookback_days)` | Remove fallback / Use real data |
| `.\src\data\news_provider.py` | 116 | `timedelta(days=` | MEDIUM/HIGH | `cutoff = datetime.utcnow() - timedelta(days=lookback_days)` | Remove fallback / Use real data |
| `.\src\data\nse_provider.py` | 163 | `timedelta(days=` | MEDIUM/HIGH | `current_date += timedelta(days=1)` | Remove fallback / Use real data |
| `.\src\data\nse_provider.py` | 261 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio=ad_ratio,` | Remove fallback / Use real data |
| `.\src\data\nse_provider.py` | 273 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio=1.0,` | Remove fallback / Use real data |
| `.\src\database\schema.py` | 108 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio: Mapped[float | None] = mapped_column(` | Remove fallback / Use real data |
| `.\src\database\schema.py` | 109 | `pct_above_50_sma` | MEDIUM/HIGH | `pct_above_50_sma: Mapped[float | None] = mapped_column(Float` | Remove fallback / Use real data |
| `.\src\quant\patterns.py` | 101 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `return PatternMatchResult(pattern_type=PatternType.FLAT_BASE` | Remove fallback / Use real data |
| `.\src\quant\patterns.py` | 121 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `pattern_type=PatternType.FLAT_BASE_BREAKOUT,` | Remove fallback / Use real data |
| `.\src\quant\patterns.py` | 133 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `return PatternMatchResult(pattern_type=PatternType.FLAT_BASE` | Remove fallback / Use real data |
| `.\src\quant\probability_engine.py` | 40 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `PatternType.FLAT_BASE_BREAKOUT: {"p_win": 0.64, "sample_size` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 22 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio: float` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 23 | `pct_above_50_sma` | MEDIUM/HIGH | `pct_above_50_sma: float` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 37 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio: float = 1.2,` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 38 | `pct_above_50_sma` | MEDIUM/HIGH | `pct_above_50_sma: float = 62.0,` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 50 | `500.0` | MEDIUM/HIGH | `nifty_close=24500.0,` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 52 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio=advance_decline_ratio,` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 53 | `pct_above_50_sma` | MEDIUM/HIGH | `pct_above_50_sma=pct_above_50_sma,` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 91 | `pct_above_50_sma` | MEDIUM/HIGH | `if pct_above_50_sma >= 65.0:` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 93 | `pct_above_50_sma` | MEDIUM/HIGH | `elif pct_above_50_sma >= 50.0:` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 95 | `pct_above_50_sma` | MEDIUM/HIGH | `elif pct_above_50_sma < 35.0:` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 98 | `advance_decline_ratio` | MEDIUM/HIGH | `if advance_decline_ratio >= 1.5:` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 100 | `advance_decline_ratio` | MEDIUM/HIGH | `elif advance_decline_ratio < 0.7:` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 145 | `pct_above_50_sma` | MEDIUM/HIGH | `f"Breadth (>50 SMA): {pct_above_50_sma:.1f}% | VIX: {india_v` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 154 | `advance_decline_ratio` | MEDIUM/HIGH | `advance_decline_ratio=advance_decline_ratio,` | Remove fallback / Use real data |
| `.\src\quant\regime.py` | 155 | `pct_above_50_sma` | MEDIUM/HIGH | `pct_above_50_sma=pct_above_50_sma,` | Remove fallback / Use real data |
| `.\src\quant\screener.py` | 127 | `55.0` | MEDIUM/HIGH | `if rsi >= 55.0:` | Remove fallback / Use real data |
| `.\src\shadow\degradation_monitor.py` | 31 | `55.0` | MEDIUM/HIGH | `MIN_WIN_RATE_PCT = 55.0` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 24 | `np.linspace` | MEDIUM/HIGH | `close = np.linspace(100, 180, n)` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 28 | `np.full` | MEDIUM/HIGH | `volume = np.full(n, 800000)` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 28 | `800000` | MEDIUM/HIGH | `volume = np.full(n, 800000)` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 32 | `np.full` | MEDIUM/HIGH | `"volume": volume, "turnover_crores": turnover, "delivery_pct` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 48 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 55 | `np.linspace` | MEDIUM/HIGH | `nifty_close = pd.Series(np.linspace(22000, 23500, 80))` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 63 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 74 | `55.0` | MEDIUM/HIGH | `promoter_pct=55.0,` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 80 | `500.0` | MEDIUM/HIGH | `debt_to_equity=0.5, cfo_crores=500.0, cfo_to_pat_ratio=0.85,` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 87 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 104 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_agents.py` | 128 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 28 | `np.linspace` | MEDIUM/HIGH | `close = np.linspace(1000, 1300, n)` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 32 | `np.full` | MEDIUM/HIGH | `volume = np.full(n, 500000)` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 56 | `np.linspace` | MEDIUM/HIGH | `close = np.linspace(1000, 700, n)` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 60 | `np.full` | MEDIUM/HIGH | `volume = np.full(n, 500000)` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 81 | `np.random` | MEDIUM/HIGH | `np.random.seed(99)` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 83 | `np.random` | MEDIUM/HIGH | `close = np.cumprod(1.0 + np.random.normal(0.0005, 0.015, n))` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 83 | `normal(` | MEDIUM/HIGH | `close = np.cumprod(1.0 + np.random.normal(0.0005, 0.015, n))` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 83 | `500.0` | MEDIUM/HIGH | `close = np.cumprod(1.0 + np.random.normal(0.0005, 0.015, n))` | Remove fallback / Use real data |
| `.\tests\test_backtest.py` | 88 | `np.full` | MEDIUM/HIGH | `df = pd.DataFrame({"open": open_p, "high": high, "low": low,` | Remove fallback / Use real data |
| `.\tests\test_data_validation.py` | 17 | `timedelta(days=` | MEDIUM/HIGH | `dates = [datetime.utcnow() - timedelta(days=i) for i in rang` | Remove fallback / Use real data |
| `.\tests\test_data_validation.py` | 20 | `np.linspace` | MEDIUM/HIGH | `"open": np.linspace(100, 150, 100),` | Remove fallback / Use real data |
| `.\tests\test_data_validation.py` | 21 | `np.linspace` | MEDIUM/HIGH | `"high": np.linspace(102, 153, 100),` | Remove fallback / Use real data |
| `.\tests\test_data_validation.py` | 22 | `np.linspace` | MEDIUM/HIGH | `"low": np.linspace(99, 148, 100),` | Remove fallback / Use real data |
| `.\tests\test_data_validation.py` | 23 | `np.linspace` | MEDIUM/HIGH | `"close": np.linspace(101, 151, 100),` | Remove fallback / Use real data |
| `.\tests\test_data_validation.py` | 24 | `np.random` | MEDIUM/HIGH | `"volume": np.random.randint(100000, 500000, size=100),` | Remove fallback / Use real data |
| `.\tests\test_data_validation.py` | 25 | `np.random` | MEDIUM/HIGH | `"delivery_volume": np.random.randint(50000, 250000, size=100` | Remove fallback / Use real data |
| `.\tests\test_data_validation.py` | 26 | `np.random` | MEDIUM/HIGH | `"delivery_pct": np.random.uniform(40.0, 70.0, size=100),` | Remove fallback / Use real data |
| `.\tests\test_indicators.py` | 14 | `np.random` | MEDIUM/HIGH | `np.random.seed(42)` | Remove fallback / Use real data |
| `.\tests\test_indicators.py` | 16 | `np.random` | MEDIUM/HIGH | `close = np.cumprod(1.0 + np.random.normal(0.001, 0.015, n)) ` | Remove fallback / Use real data |
| `.\tests\test_indicators.py` | 16 | `normal(` | MEDIUM/HIGH | `close = np.cumprod(1.0 + np.random.normal(0.001, 0.015, n)) ` | Remove fallback / Use real data |
| `.\tests\test_indicators.py` | 17 | `np.random` | MEDIUM/HIGH | `high = close * (1.0 + np.random.uniform(0.005, 0.02, n))` | Remove fallback / Use real data |
| `.\tests\test_indicators.py` | 18 | `np.random` | MEDIUM/HIGH | `low = close * (1.0 - np.random.uniform(0.005, 0.02, n))` | Remove fallback / Use real data |
| `.\tests\test_indicators.py` | 20 | `np.random` | MEDIUM/HIGH | `volume = np.random.randint(100000, 500000, n)` | Remove fallback / Use real data |
| `.\tests\test_models.py` | 53 | `800000` | MEDIUM/HIGH | `delivery_volume=2800000,` | Remove fallback / Use real data |
| `.\tests\test_p0_p1_integrity.py` | 73 | `500.0` | MEDIUM/HIGH | `"open": [2500.0] * 60, "high": [2550.0] * 60, "low": [2480.0` | Remove fallback / Use real data |
| `.\tests\test_p0_p1_integrity.py` | 83 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_p0_p1_integrity.py` | 110 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_p1_intelligence.py` | 52 | `timedelta(days=` | MEDIUM/HIGH | `published_at=datetime.utcnow() - timedelta(days=1),` | Remove fallback / Use real data |
| `.\tests\test_p1_intelligence.py` | 68 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_p1_intelligence.py` | 88 | `timedelta(days=` | MEDIUM/HIGH | `period_end_date=date.today() - timedelta(days=90),` | Remove fallback / Use real data |
| `.\tests\test_p1_intelligence.py` | 118 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_p1_intelligence.py` | 151 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_p1_intelligence.py` | 170 | `500.0` | MEDIUM/HIGH | `current_price=500.0,` | Remove fallback / Use real data |
| `.\tests\test_p2_self_improving.py` | 23 | `55.0` | MEDIUM/HIGH | `{"pnl_pct": -4.0, "agent_scores": {"technical_agent": 60.0, ` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 21 | `np.linspace` | MEDIUM/HIGH | `prices.extend(np.linspace(1000, 1120, 10))` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 22 | `np.linspace` | MEDIUM/HIGH | `prices.extend(np.linspace(1120, 1010, 10))` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 24 | `np.linspace` | MEDIUM/HIGH | `prices.extend(np.linspace(1010, 1070, 10))` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 25 | `np.linspace` | MEDIUM/HIGH | `prices.extend(np.linspace(1070, 1025, 10))` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 27 | `np.linspace` | MEDIUM/HIGH | `prices.extend(np.linspace(1025, 1050, 10))` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 28 | `np.linspace` | MEDIUM/HIGH | `prices.extend(np.linspace(1050, 1045, 10))` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 36 | `np.linspace` | MEDIUM/HIGH | `np.linspace(500000, 300000, 20),` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 37 | `np.linspace` | MEDIUM/HIGH | `np.linspace(300000, 200000, 20),` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 38 | `np.linspace` | MEDIUM/HIGH | `np.linspace(200000, 100000, 20)` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 53 | `np.random` | MEDIUM/HIGH | `close_base = np.random.uniform(1000, 1035, n_base)` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 57 | `np.full` | MEDIUM/HIGH | `vol_base = np.full(n_base, 100000)` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 60 | `55.0` | MEDIUM/HIGH | `close_bo = [1055.0]` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 77 | `FLAT_BASE_BREAKOUT` | MEDIUM/HIGH | `assert res.pattern_type == PatternType.FLAT_BASE_BREAKOUT` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 83 | `np.linspace` | MEDIUM/HIGH | `close = np.linspace(100, 200, 50)` | Remove fallback / Use real data |
| `.\tests\test_patterns.py` | 87 | `np.full` | MEDIUM/HIGH | `volume = np.full(50, 150000)` | Remove fallback / Use real data |
| `.\tests\test_providers.py` | 39 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_providers.py` | 60 | `timedelta(days=` | MEDIUM/HIGH | `"published_at": (datetime.utcnow() - timedelta(days=1)).isof` | Remove fallback / Use real data |
| `.\tests\test_providers.py` | 72 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_providers.py` | 86 | `800000` | MEDIUM/HIGH | `"DELIV_QTY": [900000, 2800000],` | Remove fallback / Use real data |
| `.\tests\test_providers.py` | 97 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 19 | `np.linspace` | MEDIUM/HIGH | `bench_close = pd.Series(np.linspace(100, 110, 60))` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 20 | `np.linspace` | MEDIUM/HIGH | `stock_close = pd.Series(np.linspace(100, 135, 60))` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 34 | `np.linspace` | MEDIUM/HIGH | `nifty_close = np.linspace(20000, 25000, 100)` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 38 | `np.full` | MEDIUM/HIGH | `vol = np.full(100, 5000000)` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 42 | `advance_decline_ratio` | MEDIUM/HIGH | `nifty_df, advance_decline_ratio=1.8, pct_above_50_sma=75.0, ` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 42 | `pct_above_50_sma` | MEDIUM/HIGH | `nifty_df, advance_decline_ratio=1.8, pct_above_50_sma=75.0, ` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 54 | `np.linspace` | MEDIUM/HIGH | `good_close = np.linspace(100, 160, 100)` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 60 | `np.full` | MEDIUM/HIGH | `"volume": np.full(100, 1000000),` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 61 | `np.full` | MEDIUM/HIGH | `"turnover_crores": np.full(100, 15.0),` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 78 | `np.linspace` | MEDIUM/HIGH | `down_close = np.linspace(200, 100, 100)` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 84 | `np.full` | MEDIUM/HIGH | `"volume": np.full(100, 1000000),` | Remove fallback / Use real data |
| `.\tests\test_screener.py` | 85 | `np.full` | MEDIUM/HIGH | `"turnover_crores": np.full(100, 10.0),` | Remove fallback / Use real data |
| `.\tests\test_universe.py` | 48 | `asyncio.run` | MEDIUM/HIGH | `asyncio.run(_run())` | Remove fallback / Use real data |
