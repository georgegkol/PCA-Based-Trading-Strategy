import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv
import os
from functions.Data_download import get_dataset_split, filter_volume_lowprices_availtickers
from functions.Tech_Indicators import rsi, macd, rolling_beta, ulcer_index, atr
from functions.Residuals_PCA_function import get_residuals, composite_alpha, generate_long_only_signals, filter_fundamentals, build_daily_net_portfolio_returns
from functions.Residuals_PCA_function import calculate_future_returns, Ledoit_pca_factor_loadings, pca_factor_loadings, neutralize_z_residual
from functions.Residuals_PCA_function import regime_scaling, vol_scaling, calculate_turnover_and_cost, calculate_rebalance_trades, calculate_portfolio_return

load_dotenv()
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD") or os.getenv("GMAIL_PASSWORD")

def send_trade_email(sells, buys, holds, rebalance_date):
    subject = f"Rebalance Alert - {rebalance_date}"
    body = f"""PCA Strategy Rebalance

Date: {rebalance_date}

SELL : {', '.join(sells) if sells else 'none'}
BUY  : {', '.join(buys)  if buys  else 'none'}
HOLD : {', '.join(holds) if holds else 'none'}

Trades saved to pending_trades.json.
Run approve_trades.py to execute via IBKR.
"""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = GMAIL_EMAIL
    msg['To'] = GMAIL_EMAIL

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
    print("Email sent.")

# ─── Setup ───────────────────────────────────────────────────────────────────
DB_PATH = 'datasets/stock_data.db'
today = datetime.today().strftime('%Y-%m-%d')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)

# ─── Load and prepare data (same as Monthly_re.py) ──────────────────────────
df_prices = get_dataset_split(split="test")
df_prices = df_prices.sort_values(by=["ticker", "date"]).reset_index(drop=True)
cols = ["ticker"] + [col for col in df_prices.columns if col != "ticker"]
df_prices = df_prices[cols]
df_prices['date'] = df_prices['date'].dt.tz_localize(None)

df_prices, tickers = filter_volume_lowprices_availtickers(df_prices, volume_threshold=800000, remove_price_below=7, volume_window=20)

rolling_df = df_prices.copy()
tickers = pd.read_csv('datasets/tickers.csv')
ticker_to_sector = tickers.set_index('ticker')['gsector'].to_dict()
rolling_df['gsector'] = rolling_df['ticker'].map(ticker_to_sector)
rolling_df = rolling_df.sort_values(['ticker', 'date'])

rolling_df['daily_return'] = rolling_df.groupby('ticker')['adjusted_close'].pct_change()
rolling_df['daily_return'] = rolling_df['daily_return'].clip(-0.3, 0.3)
rolling_df['weekly_return_5d'] = rolling_df.groupby('ticker')['adjusted_close'].pct_change(5)
rolling_df['medium_return_20d'] = rolling_df.groupby('ticker')['adjusted_close'].pct_change(20)
rolling_df['volatility_20d'] = rolling_df.groupby('ticker')['daily_return'].rolling(20).std().reset_index(0, drop=True)
rolling_df['avg_volume_20d'] = rolling_df.groupby('ticker')['volume'].rolling(20).mean().reset_index(0, drop=True)
rolling_df['momentum_volatility'] = rolling_df.groupby('ticker')['weekly_return_5d'].rolling(20).std().reset_index(0, drop=True)
rolling_df['macd_hist'] = rolling_df.groupby('ticker')['adjusted_close'].transform(macd)
rolling_df['sector_return'] = rolling_df.groupby(['gsector', 'date'])['medium_return_20d'].transform('mean')
group = rolling_df.groupby('ticker', group_keys=False)
beta_values = group[['medium_return_20d', 'sector_return']].apply(lambda df: rolling_beta(df['medium_return_20d'], df['sector_return'], window=60))
rolling_df['beta_sector'] = beta_values
rolling_df['range_compression'] = (rolling_df['high'] - rolling_df['low']) / rolling_df['close']
rolling_df['range_compression_5d'] = rolling_df.groupby('ticker')['range_compression'].rolling(5).mean().reset_index(0, drop=True)
rolling_df['ulcer_20d'] = rolling_df.groupby('ticker')['adjusted_close'].transform(lambda x: ulcer_index(x, 20))
atr_input = rolling_df[['ticker', 'high', 'low', 'close']].copy()
rolling_df['atr_20d'] = (atr_input.groupby('ticker', group_keys=False).apply(lambda df: atr(df['high'], df['low'], df['close'], n=20), include_groups=False))
rolling_df['kc_width'] = rolling_df['atr_20d'] / rolling_df['close']

window_span = 50
factor_cols = ['weekly_return_5d', 'volatility_20d', 'avg_volume_20d', 'momentum_volatility', 'macd_hist', 'beta_sector', 'range_compression_5d']
for col in factor_cols:
    rolling_df[col] = rolling_df.groupby('ticker')[col].transform(lambda x: x.ewm(span=window_span, min_periods=1).mean())

rebalance_dates, rolling_df = calculate_future_returns(rolling_df, holding_period_days=20)

portfolio_returns = [{'date': rebalance_dates[4], 'portfolio_return': 0, 'raw_return': 0, 'turnover': 0, 'transaction_cost': 0}]
previous_longs_df = pd.DataFrame(columns=['ticker', 'weight'])
signal_history = []
alpha_history = {}
portfolio_weight_history = []
transaction_cost = 0.004375
n = 5

# ─── Backtest loop ───────────────────────────────────────────────────────────
for rebalance_date in rebalance_dates[n:]:
    df_slice = rolling_df[rolling_df['date'] == rebalance_date].copy()
    df_slice = df_slice[(df_slice['ulcer_20d'] < df_slice['ulcer_20d'].quantile(0.6))]

    factor_loadings = pca_factor_loadings(df_slice, factor_cols, pcs_to_use=slice(None))
    residuals_df = get_residuals(factor_loadings, df_slice[['ticker', 'gsector', 'weekly_return_5d']])
    residuals_df = neutralize_z_residual(residuals_df, df_slice, controls=['beta_sector'])
    residuals_df = composite_alpha(residuals_df, alpha_history)

    longs_df = generate_long_only_signals(residuals_df, top_n=3, skip_bottom_n=0, per_sector_limit=1)
    longs_df = longs_df.reset_index(drop=True)

    regime_scale = regime_scaling(residuals_df, rebalance_date, longs_df, signal_history)
    if regime_scale == 0.5:
        longs_df['weight'] *= 0.5

    turnover, transaction_cost, _ = calculate_turnover_and_cost(previous_longs_df, longs_df, transaction_cost_per_unit=transaction_cost)
    final_trades = calculate_rebalance_trades(longs_df, previous_longs_df)
    portfolio_return, _ = calculate_portfolio_return(longs_df, df_slice[['ticker', 'future_return']], rebalance_date)
    net_portfolio_return = portfolio_return - transaction_cost
    portfolio_returns.append({'date': rebalance_date, 'portfolio_return': net_portfolio_return, 'raw_return': portfolio_return, 'turnover': turnover, 'transaction_cost': transaction_cost})

    previous_longs_df = longs_df.copy()
    longs_df['date'] = rebalance_date
    portfolio_weight_history.append(longs_df.copy())

# ─── Final portfolio: trades to execute ─────────────────────────────────────
sells = final_trades[final_trades['weight_new'] == 0]['ticker'].tolist()
buys  = final_trades[final_trades['weight_old'] == 0]['ticker'].tolist()
holds = final_trades[(final_trades['weight_new'] > 0) & (final_trades['weight_old'] > 0)]['ticker'].tolist()

rebalance_date_str = str(rebalance_dates[-1].date())

print("\n========================================")
print(f"  Rebalance date: {rebalance_date_str}")
print("========================================")
print(f"  SELL : {sells if sells else 'none'}")
print(f"  BUY  : {buys  if buys  else 'none'}")
print(f"  HOLD : {holds if holds else 'none'}")
print("========================================\n")

# ─── Save trades and notify ──────────────────────────────────────────────────
if not sells and not buys:
    print("No trades needed this rebalance.")
else:
    pending = {'date': rebalance_date_str, 'sells': sells, 'buys': buys, 'holds': holds}
    with open('pending_trades.json', 'w') as f:
        json.dump(pending, f)
    print("Trades saved to pending_trades.json.")

    send_trade_email(sells, buys, holds, rebalance_date_str)

# ─── Save last rebalance date for scheduler ──────────────────────────────────
with open('last_rebalance.txt', 'w') as f:
    f.write(rebalance_date_str)

# ─── Save portfolio state for investor updates ───────────────────────────────
state = {
    'last_rebalance': rebalance_date_str,
    'current_holdings': previous_longs_df[['ticker', 'weight']].to_dict(orient='records'),
    'portfolio_returns': [
        {k: (v.strftime('%Y-%m-%d') if hasattr(v, 'strftime') else v) for k, v in r.items()}
        for r in portfolio_returns
    ],
}
with open('portfolio_state.json', 'w') as f:
    json.dump(state, f, indent=2)
print("Portfolio state saved to portfolio_state.json.")
