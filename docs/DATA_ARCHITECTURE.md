# DATA ARCHITECTURE — NSE SWING AI

## Overview
This document describes the zero-trust, data-integrity-first architecture for the `nse_swing_ai` platform.

## Core Ingestion Principles
1. **Zero Synthetic Data**: No production or backtest path may generate synthetic prices (`np.random`, `np.full`, `np.linspace`). Missing data raises `DATA_UNAVAILABLE` or `DATA_INVALID`.
2. **Real Historical Provider**: `HistoricalDataProvider` (`src/data/historical_provider.py`) fetches genuine date-by-date EOD Bhavcopy and DB records.
3. **Data Quality Validation**: Every dataframe is validated via `validate_ohlcv_dataframe` (`src/data/validation.py`) prior to indicator computation.
4. **No Fallback Prices**: Missing investment values are never replaced with hardcoded numbers (`price or 500` is eliminated).
