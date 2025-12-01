Quant Research & Trading Platform

A modular, execution-aware, multi-layer quantitative research framework.

Overview

This project is a full-stack quantitative trading research platform modeled after hedge-fund and quant-prop architectures.
It provides a complete pipeline:

data ingestion → feature engineering → alpha signals → execution-aware backtesting → analytics & reporting

all inside a clean Python + Streamlit interface.

The platform enables users to:

Upload their own market data

Configure strategies via YAML / JSON

Run execution-aware backtests

Analyze equity curves, drawdowns, trades, and positions

Extend strategies, models, and execution engines

Although the included example uses a statistical arbitrage (pair trading) workflow, the architecture is generic and designed for multi-asset expansion.

Architecture

The platform follows a production-grade buy-side quant architecture.

1. Data Layer

CSV / Parquet ingestion

MultiIndex formatting (timestamp, symbol)

Input schema validation

Extensible to equities, futures, FX, crypto, and macro datasets

2. Strategy Layer

Modular Strategy base class

Example: StatArbExecutionStrategy

Clean pipeline for spreads, signals, regime filters, hedge ratios

Volatility targeting & adaptive sizing

3. Execution Layer

Institutional-style ExecutionContext

Market & limit order simulation

Latency modeling

Slippage & transaction-cost models

Partial-fill & order-queue logic

Broker ledger + trade logs

Fully pluggable execution models

4. Portfolio Layer

Volatility-targeted position sizing

Dollar-neutral or directional

Hedge-ratio integration

Multi-symbol slicing helpers

5. Backtesting Runner

Orchestrates strategy, data, and execution

Respects random seeds, slippage, cost curves

Produces:

Equity curve

Drawdowns

Trades

Positions

Risk metrics (JSON)

6. UI Layer (Streamlit)

Upload market data

Upload or auto-generate config files

Run full backtest

Interactive charts (PnL, DD, risk metrics)

Positions & trade introspection

Features
✔ Execution-Aware Backtesting

Simulates real market microstructure:

Latency

Slippage models

Transaction costs

Execution queue

Market order routing

Partial fills

✔ Statistical Arbitrage Pipeline

Includes:

Engle–Granger cointegration

OU spread modeling

Regime filtering (HMM-compatible)

Z-score entry/exit logic

Volatility-targeted sizing

✔ Interactive Streamlit Interface

Data upload

Config upload or default injection

Auto symbol detection

Equity curve, drawdowns, risk metrics

Positions & trades visualization

✔ YAML / JSON Config System

Control everything:

Data source

Strategy parameters

Execution models

Saving behavior

Random seeds

How To Use
1. Launch the UI
streamlit run src/quant_platform/ui/app.py

2. Upload Your Market Data (CSV)

Format:

timestamp,symbol,close
2025-01-02,LOW,246.98
2025-01-02,HD,388.46
...

3. Upload Config or Use Default

You may:

Upload your own config.yaml, or

Click Use Default Stat-Arb Config, which injects:

Data source

Symbols

Strategy parameters

Execution settings

4. Run Backtest

Click Run Backtest → The following appear:

Equity curve

Drawdowns

Risk metrics

Positions over time

Project Structure
quant_platform/
    data/
    trading/
        stat_arb/
            pipeline/
            spreads/
            schemas.py
    execution/
        context.py
        models.py
    portfolio/
        position_sizing.py
    runner/
        core.py
        run.py
        config/
            models.py
    ui/
        app.py

Future Roadmap

Multi-pair stat-arb portfolio

Automatic pair selection & cointegration scanning

Multi-strategy framework

Expanded risk engine (factor models, correlation regimes)

Local volatility & advanced derivatives pricing

Cross-sectional ML alpha models

Execution optimizers (TWAP / VWAP / POV / impact models)

Multi-asset support (futures, FX, crypto)