# PORTFOLIO BACKTESTING ENGINE — NSE SWING AI

## Overview
`BacktestEngine` (`src/backtest/engine.py`) provides walk-forward event-driven portfolio simulation with realistic Indian market friction.

## Features
- **Real Market Data**: Zero synthetic price series generation.
- **Survivorship-Safe Universe**: Uses date-aware historical universe snapshots (`HistoricalUniverseProvider`).
- **Portfolio Equity Curve**: Calculates true daily mark-to-market portfolio drawdown starting from initial capital.
- **Gap-Through-Stop Handling**: Exits at open price if overnight gap breaches stop loss floor.
- **Partial Target Exits**: T1 (50%), T2 (30%), T3 (20% trail).
