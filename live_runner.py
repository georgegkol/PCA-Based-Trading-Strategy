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

STATE_FILE = 'portfolio_state.json'
DB_PATH = 'datasets/stock_data.db'
today = datetime.today().strftime('%Y-%m-%d')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)


def send_trade_email(sells, buys, holds, rebalance_date):
    subject = f"Rebalance Alert - {rebalance_date}"
    body = f"""PCA Strategy Rebalance

Date: {rebalance_date}

SELL : {', '.join(sells) if sells else 'none'}
BUY  : {', '.join(buys)  if buys  else 'none'}
HOLD : {', '.join(holds) if holds else 'none'}

Trades saved to pending_trades.json.
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


# ─── Detect mode ─────────────────────────────────────────────────────────────
if os.path.exists(STATE_FILE):
    print("[INCREMENTAL] State file found — loading recent data only.")
    with open(STATE_FILE) as f:
        saved_state = json.load(f)

    alpha_history = saved_state.get('alpha_history', {})
    signal_history = saved_state.get('signal_history', [])
    previous_longs_df = pd.DataFrame(saved_state.get('current_holdings', []))
    portfolio_returns = saved_state.get('portfolio_returns', [])
    last_rebalance_saved = pd.Timestamp(saved_state['last_rebalance'])

    # Load last 300 calendar days (~210 trading days — enough for EWM span=50 to converge)
    lookback_date = (last_rebalance_saved - pd.Timedelta(days=300)).strftime('%Y-%m-%d')
    print(f"[INCREMENTAL] Querying DB from {lookback_date} to {today}...")
    conn = sqlite3.connect(DB_PATH)
    df_prices = pd.read_sql_query(
        "SELECT date, close, open, high, low, adjusted_close, volume, ticker "
        f"FROM stock_prices WHERE date(date) >= date('{lookback_date}')",
        conn, parse_dates=['date'])
    conn.close()
    df_prices['date'] = pd.to_datetime(df_prices['date'])

    # Use ticker universe from tickers.csv (volume/price filter needs full history; skip here)
    tickers_csv = pd.read_csv('datasets/tickers.csv')
    df_prices = df_prices[df_prices['ticker'].isin(tickers_csv['ticker'])].copy()
    df_prices = df_prices.drop_duplicates(subset=['ticker', 'date'], keep='first')
    print(f"[INCREMENTAL] Loaded {len(df_prices):,} rows, {df_prices['ticker'].nunique()} tickers.")

else:
    print("[FULL] No state file — running complete backtest from scratch.")
    alpha_history = {}
    signal_history = []
    previous_longs_df = pd.DataFrame(columns=['ticker', 'weight'])
    portfolio_returns = []
    last_rebalance_saved = None

    df_prices = get_dataset_split(split="test")
    df_prices = df_prices.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    df_prices, tickers_csv = filter_volume_lowprices_availtickers(
        df_prices, volume_threshold=800000, remove_price_below=7, volume_window=20)
    tickers_csv = pd.read_csv('datasets/tickers.csv')


# ─── Compute indicators (identical for both modes) ───────────────────────────
df_prices = df_prices.sort_values(by=["ticker", "date"]).reset_index(drop=True)
cols = ["ticker"] + [col for col in df_prices.columns if col != "ticker"]
df_prices = df_prices[cols]
df_prices['date'] = df_prices['date'].dt.tz_localize(None)

rolling_df = df_prices.copy()
ticker_to_sector = tickers_csv.set_index('ticker')['gsector'].to_dict()
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
beta_values = group[['medium_return_20d', 'sector_return']].apply(
    lambda df: rolling_beta(df['medium_return_20d'], df['sector_return'], window=60))
rolling_df['beta_sector'] = beta_values
rolling_df['range_compression'] = (rolling_df['high'] - rolling_df['low']) / rolling_df['close']
rolling_df['range_compression_5d'] = rolling_df.groupby('ticker')['range_compression'].rolling(5).mean().reset_index(0, drop=True)
rolling_df['ulcer_20d'] = rolling_df.groupby('ticker')['adjusted_close'].transform(lambda x: ulcer_index(x, 20))
atr_input = rolling_df[['ticker', 'high', 'low', 'close']].copy()
rolling_df['atr_20d'] = (atr_input.groupby('ticker', group_keys=False).apply(
    lambda df: atr(df['high'], df['low'], df['close'], n=20), include_groups=False))
rolling_df['kc_width'] = rolling_df['atr_20d'] / rolling_df['close']

window_span = 50
factor_cols = ['weekly_return_5d', 'volatility_20d', 'avg_volume_20d', 'momentum_volatility',
               'macd_hist', 'beta_sector', 'range_compression_5d']
for col in factor_cols:
    rolling_df[col] = rolling_df.groupby('ticker')[col].transform(
        lambda x: x.ewm(span=window_span, min_periods=1).mean())

rebalance_dates, rolling_df = calculate_future_returns(rolling_df, holding_period_days=20)

# ─── Determine which periods to process ──────────────────────────────────────
if last_rebalance_saved is not None:
    # Incremental: only new dates strictly after the saved last rebalance
    dates_to_run = [d for d in rebalance_dates if pd.Timestamp(d) > last_rebalance_saved]
    print(f"[INCREMENTAL] {len(dates_to_run)} new rebalance period(s) to process.")
else:
    # Full mode: first 5 periods used as warmup baseline for regime_scaling
    portfolio_returns = [{'date': str(rebalance_dates[4].date()), 'portfolio_return': 0,
                          'raw_return': 0, 'turnover': 0, 'transaction_cost': 0}]
    dates_to_run = list(rebalance_dates[5:])
    print(f"[FULL] Running {len(dates_to_run)} rebalance periods...")

transaction_cost = 0.004375
portfolio_weight_history = []
final_trades = None

# ─── Main loop ───────────────────────────────────────────────────────────────
for rebalance_date in dates_to_run:
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

    turnover, transaction_cost, _ = calculate_turnover_and_cost(
        previous_longs_df, longs_df, transaction_cost_per_unit=transaction_cost)
    final_trades = calculate_rebalance_trades(longs_df, previous_longs_df)
    portfolio_return, _ = calculate_portfolio_return(
        longs_df, df_slice[['ticker', 'future_return']], rebalance_date)
    net_portfolio_return = portfolio_return - transaction_cost
    portfolio_returns.append({
        'date': str(rebalance_date.date()),
        'portfolio_return': net_portfolio_return,
        'raw_return': portfolio_return,
        'turnover': turnover,
        'transaction_cost': transaction_cost
    })

    previous_longs_df = longs_df.copy()
    longs_df['date'] = rebalance_date
    portfolio_weight_history.append(longs_df.copy())

# ─── Report and notify ───────────────────────────────────────────────────────
if not dates_to_run:
    print("No new rebalance dates — portfolio is already up to date.")
    rebalance_date_str = saved_state['last_rebalance']
else:
    sells = final_trades[final_trades['weight_new'] == 0]['ticker'].tolist()
    buys  = final_trades[final_trades['weight_old'] == 0]['ticker'].tolist()
    holds = final_trades[(final_trades['weight_new'] > 0) & (final_trades['weight_old'] > 0)]['ticker'].tolist()

    rebalance_date_str = str(dates_to_run[-1].date())

    print("\n========================================")
    print(f"  Rebalance date: {rebalance_date_str}")
    print("========================================")
    print(f"  SELL : {sells if sells else 'none'}")
    print(f"  BUY  : {buys  if buys  else 'none'}")
    print(f"  HOLD : {holds if holds else 'none'}")
    print("========================================\n")

    if not sells and not buys:
        print("No trades needed this rebalance.")
    else:
        pending = {'date': rebalance_date_str, 'sells': sells, 'buys': buys, 'holds': holds}
        with open('pending_trades.json', 'w') as f:
            json.dump(pending, f)
        print("Trades saved to pending_trades.json.")
        send_trade_email(sells, buys, holds, rebalance_date_str)

    with open('last_rebalance.txt', 'w') as f:
        f.write(rebalance_date_str)

# ─── Save state — alpha_history + signal_history make future runs fast ────────
state = {
    'last_rebalance': rebalance_date_str,
    'current_holdings': previous_longs_df[['ticker', 'weight']].to_dict(orient='records'),
    'portfolio_returns': [
        {k: (v.strftime('%Y-%m-%d') if hasattr(v, 'strftime') else v) for k, v in r.items()}
        for r in portfolio_returns
    ],
    'alpha_history': {k: float(v) for k, v in alpha_history.items()},
    'signal_history': [float(v) for v in signal_history],
}
with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)
print(f"State saved to {STATE_FILE}.")
print(f"  alpha_history : {len(alpha_history)} tickers")
print(f"  signal_history: {len(signal_history)} entries")
print(f"  portfolio_returns: {len(portfolio_returns)} periods")
