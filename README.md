# PCA-Based Trading Strategy

A quantitative long-only equity strategy that uses Principal Component Analysis (PCA) to extract alpha signals from technical indicators across NYSE stocks. Rebalances every 20 NYSE trading days, runs fully automated on AWS EC2.

## How it works

Technical indicators (MACD, ATR, beta, volatility, momentum, etc.) are computed for all NYSE stocks. PCA reduces these into latent factors and the per-stock residuals — the part PCA can't explain — are used as the alpha signal. Residuals are exponentially smoothed across periods to reward consistency over noise. The top 3 stocks by composite alpha (max 1 per sector) are held with equal weight until the next rebalance. A regime detector scales exposure to 50% during weak-signal periods.

On each rebalance, the strategy emails the portfolio manager with the SELL / BUY / HOLD list. Trades are placed manually on Interactive Brokers.

## Automation

Everything runs on an AWS EC2 instance (t3.micro). Three cron jobs:

| Time (UTC) | Job | What it does |
|---|---|---|
| 2am daily | `download_data.py` | Fetches latest prices from EODHD API, appends to DB |
| 9am Mon–Fri | `scheduler_check.py` | Checks if 20 NYSE trading days have passed; if so, runs `live_runner.py` |
| 8am daily | `investor_update.py` | Sends investor portfolio updates every 7 days |

## State persistence

`live_runner.py` runs in two modes:

- **Full mode** (no `portfolio_state.json`): loads the complete price history, runs the full backtest. Used once locally to generate the initial state file.
- **Incremental mode** (`portfolio_state.json` exists): loads only the last ~300 days from DB (~85MB vs 3GB+), runs PCA only for new rebalance periods since the last saved state. This is what runs on AWS every 20 days.

The state file stores `alpha_history` (smoothed per-ticker alpha signals) and `signal_history` (per-period signal strengths for regime detection), so each incremental run picks up exactly where the last one left off.

## Investor emails

`investor_update.py` sends each investor a dark-themed HTML email every 7 days with:
- Current portfolio value and cumulative return since their start date
- Strategy vs S&P 500 comparison (last cycle and since inception)
- Live price performance of current holdings since last rebalance
- Estimated next rebalance date

Investors are listed in `investors.csv` (gitignored).

## Project structure

```
├── live_runner.py               # PCA strategy — full or incremental mode
├── scheduler_check.py           # Daily check: fires live_runner after 20 NYSE trading days
├── download_data.py             # Daily price download, appends to stock_data.db
├── investor_update.py           # Investor email updates (every 7 days)
├── approve_trades.py            # Optional: execute trades via IBKR API
├── ibkr_executor.py             # IBKR order execution utilities
├── functions/
│   ├── Data_download.py         # EODHD API and SQLite utilities
│   ├── Tech_Indicators.py       # RSI, MACD, ATR, beta, ulcer index, etc.
│   └── Residuals_PCA_function.py# PCA, alpha smoothing, regime scaling, portfolio construction
├── datasets/                    # Gitignored
│   ├── stock_data.db            # Full OHLCV history (~668MB, 2000+ NYSE tickers)
│   ├── tickers.csv              # NYSE ticker universe with GICS sectors
│   └── sp500_total_return.csv   # SPY benchmark data
├── portfolio_state.json         # Gitignored — alpha_history, signal_history, holdings
└── investors.csv                # Gitignored — investor names, emails, start dates
```

## Requirements

- Python 3.9+
- EODHD API key
- Gmail account with App Password (for outbound emails)

```bash
pip install pandas numpy requests python-dotenv exchange_calendars sqlalchemy
```

## Environment variables

Create a `.env` file in the root:

```
API_KEY=your_eodhd_key
GMAIL_EMAIL=you@gmail.com
GMAIL_PASSWORD=your_gmail_app_password
```

## First-time setup

1. Run `download_data.py` locally to build the initial `stock_data.db`
2. Run `live_runner.py` locally (full mode, ~2 min) to generate `portfolio_state.json`
3. Upload `stock_data.db`, `portfolio_state.json`, `investors.csv`, and `.env` to the EC2 instance
4. Set up the three cron jobs above — the strategy runs itself from there

## Notes

- Jupyter notebooks (`Download_US_prices.ipynb`, `Monthly_re.ipynb`) are for research and backtesting only — not part of the live pipeline
- `datasets/` and all sensitive files are gitignored
- Trades are executed manually on IBKR after reviewing the rebalance email
