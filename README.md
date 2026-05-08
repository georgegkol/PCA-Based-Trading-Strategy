# PCA-Based Trading Strategy

A quantitative long-only equity strategy that uses Principal Component Analysis (PCA) to extract trading signals from technical indicators and rebalances every 20 trading days.

## How it works

Technical indicators (RSI, MACD, ATR, beta, volatility, etc.) are computed for all NYSE stocks. PCA reduces these into latent factors and per-stock residuals are used as the alpha signal. The top 3 stocks by composite alpha (max 1 per sector) are held with equal weight until the next rebalance.

## Features

- Sector-neutral stock selection
- Beta neutralization and regime scaling
- Transaction cost modeling (0.4375% per trade)
- Benchmarked against S&P 500 with Sharpe ratio comparison
- Live trade execution via Interactive Brokers API

## Requirements

- Python 3.10+ (Anaconda recommended)
- EODHD API key
- Interactive Brokers account + IB Gateway

## Setup

```bash
pip install ib_insync statsmodels python-dotenv tqdm
```

Create a `.env` file in the root folder:

```
API_KEY=your_eodhd_key
```

## Usage

Download latest data and run the strategy manually:

```bash
python live_runner.py
```

Proposed trades are shown for your approval before any orders are placed.

For backtesting and performance analysis:

```bash
python Monthly_re.py
```

## Project Structure

```
├── scheduler_check.py           # Runs daily via Task Scheduler, triggers rebalance after 20 NYSE trading days
├── live_runner.py               # Downloads data, runs PCA strategy, emails proposed trades
├── approve_trades.py            # Run manually to execute approved trades via IBKR
├── ibkr_executor.py             # Interactive Brokers order execution
├── Monthly_re.py                # Backtesting and performance analysis
├── Download_US_prices.py        # Manual data download script
├── functions/
│   ├── Data_download.py         # EODHD API and database utilities
│   ├── Tech_Indicators.py       # RSI, MACD, ATR, beta, etc.
│   ├── Residuals_PCA_function.py# PCA signal generation and portfolio construction
└── datasets/                    # Local data (gitignored)
    ├── stock_data.db
    ├── tickers.csv
    └── sp500_total_return.csv
```

## Notes

- `.env` and `datasets/` are gitignored — never committed to GitHub
- IB Gateway must be running and logged in before executing trades
- Ensure Read-Only API is disabled in IB Gateway settings (Configure → Settings → API)
