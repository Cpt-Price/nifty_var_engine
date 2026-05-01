# NIFTY 50 Risk Engine

A **machine learning-based Value-at-Risk (VaR) prediction system** for the **NIFTY 50** Indian stock market index. It compares a deep learning model (Temporal Fusion Transformer) against a traditional statistical baseline (GARCH) for financial risk forecasting.

## What It Does

The system predicts **1-day ahead 99% VaR** (the worst expected daily loss with 99% confidence) using two competing models:

| Model | Description |
|-------|-------------|
| **TFT** | Temporal Fusion Transformer — a state-of-the-art deep learning model for time series forecasting |
| **GARCH** | GJR-GARCH with Skew-Student-t distribution — a classical econometric volatility model (used as baseline) |

The pipeline then statistically evaluates whether TFT significantly outperforms GARCH using regulatory and academic tests.

## File Architecture

```
nifty_var_engine/
├── config.py          # Seed management, date ranges, TFT architecture defaults
├── data_loader.py     # Yahoo Finance data ingestion + feature engineering
├── garch_engine.py    # Rolling GARCH baseline VaR computation
├── tft_model.py       # TFT training with PyTorch Forecasting + Lightning
├── hpo.py             # Optuna-based hyperparameter optimization
├── metrics.py         # Backtesting metrics (Kupiec POF, Diebold-Mariano, Basel limits)
├── proof.py           # Granger causality test: US VIX → India VIX spillover
└── main.py            # Master orchestration script
```

## Data Pipeline

The system ingests **6 exogenous features** from Yahoo Finance:

| Ticker | Feature | Transformation |
|--------|---------|---------------|
| `^NSEI` | Nifty50 (target) | Log returns |
| `^VIX` | US VIX | First difference |
| `CL=F` | Crude Oil | Log returns |
| `^TNX` | US 10Y Treasury | First difference |
| `DX=F` | US Dollar Index | Log returns |
| `CPU_index.csv` | Global Computing Power Index | Raw level (monthly, forward-filled) |

## Execution Pipeline (4 Phases)

1. **Data Ingestion** — Downloads historical data from 2007-01-01, computes log returns / differences, runs stationarity (ADF) audits
2. **Rolling GARCH** — Fits ~4,500 rolling GJR-GARCH(1,1) models with 252-day windows to generate baseline VaR predictions (cached in `master_df.csv`)
3. **Hyperparameter Optimization** — Runs 30 Optuna trials searching over `hidden_size`, `dropout`, and `learning_rate` with median pruning
4. **5-Seed Robustness Audit** — Trains the final TFT architecture 5 times with different random seeds, computing:
   - **TFT Failures** vs Basel green zone limit
   - **Kupiec POF** (Probability of Forecast) test p-value
   - **Diebold-Mariano** test comparing TFT vs GARCH tick loss

## How to Run It

### Prerequisites

Install the required Python dependencies:

```bash
pip install yfinance pandas numpy arch pytorch-forecasting lightning optuna optuna-integration scipy statsmodels tqdm torch
```

### Optional: CPU Index Data

Place a `CPU_index.csv` file in the repository root (monthly Global Computing Power Index data). Without it, the system falls back to a placeholder value of `100.0`.

### Main Execution

```bash
python main.py
```

This runs the full 4-phase pipeline. On first run, expect **60-90 minutes** for the GARCH phase and **2-4 hours** for HPO (depending on CPU/GPU availability).

### Standalone Scripts

| Command | Purpose |
|---------|---------|
| `python proof.py` | Run the Granger causality audit (US VIX → India VIX spillover) |
| `python data_loader.py` | Test data ingestion independently |

## Key Design Decisions

1. **`GARCH_Vol` is excluded** from TFT inputs — the model would over-rely on it ("lazy learning"), suppressing exogenous variable contributions
2. **`Global_CPU` uses raw level** (not percentage change) — since the data is monthly and forward-filled to daily, `pct_change()` produces ~95% zeros
3. **Quantile loss** with quantiles `[0.01, 0.5, 0.99]` — the 0.01 quantile directly corresponds to 99% VaR
4. **Deterministic seeding** across 5 seeds ensures results are reproducible and not dependent on a single random initialization
5. **GARCH results are cached** in `master_df.csv` to avoid re-computing ~4,500 model fits on every run