import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from functions.Data_download import get_dataset_split, filter_volume_lowprices_availtickers
from functions.Tech_Indicators import rsi, macd, rolling_beta, ulcer_index, atr
from functions.Residuals_PCA_function import  get_residuals, composite_alpha, generate_long_only_signals, filter_fundamentals, build_daily_net_portfolio_returns
from functions.Residuals_PCA_function import calculate_future_returns, Ledoit_pca_factor_loadings, pca_factor_loadings, neutralize_z_residual
from functions.Residuals_PCA_function import regime_scaling, vol_scaling, calculate_turnover_and_cost, calculate_rebalance_trades, calculate_portfolio_return


pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_colwidth', 100) 
pd.set_option('display.width', 1000)

df_prices = get_dataset_split(split="test")

df_prices = df_prices.sort_values(by=["ticker", "date"]).reset_index(drop=True)

cols = ["ticker"] + [col for col in df_prices.columns if col != "ticker"]
df_prices = df_prices[cols]
df_prices['date'] = df_prices['date'].dt.tz_localize(None) # drop timezone

# Filtering and getting available tickers
df_prices, tickers = filter_volume_lowprices_availtickers(df_prices, volume_threshold=800000, remove_price_below=7, volume_window=20)

rolling_df = df_prices.copy()
tickers = pd.read_csv('datasets/tickers.csv')

# add sector to rolling_df
ticker_to_sector = tickers.set_index('ticker')['gsector'].to_dict()
rolling_df['gsector'] = rolling_df['ticker'].map(ticker_to_sector)



rolling_df = rolling_df.sort_values(['ticker', 'date'])

# Returns
rolling_df['daily_return'] = rolling_df.groupby('ticker')['adjusted_close'].pct_change()
rolling_df['daily_return'] = rolling_df['daily_return'].clip(-0.3, 0.3)
rolling_df['weekly_return_5d'] = rolling_df.groupby('ticker')['adjusted_close'].pct_change(5)
rolling_df['medium_return_20d'] = rolling_df.groupby('ticker')['adjusted_close'].pct_change(20)
# Volatility 
rolling_df['volatility_20d'] = rolling_df.groupby('ticker')['daily_return'].rolling(20).std().reset_index(0,drop=True)
# Volume
rolling_df['avg_volume_20d'] = rolling_df.groupby('ticker')['volume'].rolling(20).mean().reset_index(0,drop=True)
rolling_df['momentum_volatility'] = rolling_df.groupby('ticker')['weekly_return_5d'].rolling(20).std().reset_index(0,drop=True)
rolling_df['macd_hist'] = rolling_df.groupby('ticker')['adjusted_close'].transform(macd)
# Sector beta
rolling_df['sector_return'] = rolling_df.groupby(['gsector', 'date'])['medium_return_20d'].transform('mean')
group = rolling_df.groupby('ticker', group_keys=False)
beta_values = group[['medium_return_20d', 'sector_return']].apply(lambda df: rolling_beta(df['medium_return_20d'], df['sector_return'], window=60))
rolling_df['beta_sector'] = beta_values
# Range compression
rolling_df['range_compression'] = (rolling_df['high'] - rolling_df['low']) / rolling_df['close']
rolling_df['range_compression_5d'] = rolling_df.groupby('ticker')['range_compression'].rolling(5).mean().reset_index(0, drop=True)
# Ulcer
rolling_df['ulcer_20d'] = rolling_df.groupby('ticker')['adjusted_close'].transform(lambda x: ulcer_index(x, 20))
# kc_width
atr_input = rolling_df[['ticker', 'high', 'low', 'close']].copy()
rolling_df['atr_20d'] = ( atr_input.groupby('ticker', group_keys=False).apply(lambda df: atr(df['high'], df['low'], df['close'], n=20), include_groups=False))
rolling_df['kc_width'] = rolling_df['atr_20d'] / rolling_df['close']

# exponential smoothing
window_span = 50  # ~10 weeks
factor_cols = ['weekly_return_5d','volatility_20d','avg_volume_20d', 'momentum_volatility', 'macd_hist', 'beta_sector', 'range_compression_5d'] #+ fundamental_factors

for col in factor_cols:
    rolling_df[col] = rolling_df.groupby('ticker')[col].transform(lambda x: x.ewm(span=window_span, min_periods=1).mean())


rebalance_dates, rolling_df = calculate_future_returns(rolling_df, holding_period_days=20)

portfolio_returns = [{ 'date': rebalance_dates[4], 'portfolio_return': 0, 'raw_return': 0, 'turnover': 0, 'transaction_cost': 0}]
previous_longs_df = pd.DataFrame(columns=['ticker', 'weight'])  # Start empty for first rebalance
signal_history = []
alpha_history = {}  # ticker → smoothed alpha
portfolio_weight_history = []

transaction_cost = 0.004375
n=5
for rebalance_date in rebalance_dates[n:]:

    df_slice = rolling_df[rolling_df['date'] == rebalance_date].copy()
    df_slice = df_slice[(df_slice['ulcer_20d'] < df_slice['ulcer_20d'].quantile(0.6))] #& 
                        #(df_slice['kc_width'] < df_slice['kc_width'].quantile(0.8))]# filter out worst 20% drawdown and volatility risk
    #df_slice = filter_fundamentals(df_slice, drop_bottom=0.2) # filter out companies with low fundamentals

    factor_loadings = pca_factor_loadings(df_slice, factor_cols, pcs_to_use=slice(None))
    residuals_df = get_residuals(factor_loadings, df_slice[['ticker', 'gsector', 'weekly_return_5d']])
    residuals_df = neutralize_z_residual(residuals_df, df_slice, controls = ['beta_sector'])
    residuals_df = composite_alpha(residuals_df, alpha_history)
    
    # Generate longs using smoothed residual
    longs_df = generate_long_only_signals(residuals_df, top_n=3, skip_bottom_n=0, per_sector_limit=1)
    longs_df = longs_df.reset_index(drop=True)

    # Apply regime scaling and volatility scaling
    regime_scale = regime_scaling(residuals_df, rebalance_date, longs_df, signal_history)
    #vol_scale = vol_scaling(portfolio_returns, target_vol = 0.08)
    if regime_scale == 0.5: #or vol_scale<0.5:
        longs_df['weight'] *= 0.5 # reduce exposure during unstable regimes

    # Calculate turnover, trades that need to be executed and portfolio return
    turnover, transaction_cost, _ = calculate_turnover_and_cost(previous_longs_df, longs_df, transaction_cost_per_unit=transaction_cost)
    trades = calculate_rebalance_trades(longs_df, previous_longs_df)

    portfolio_return, _ = calculate_portfolio_return(longs_df, df_slice[['ticker', 'future_return']], rebalance_date)
    net_portfolio_return = portfolio_return - transaction_cost
    portfolio_returns.append({'date': rebalance_date,
                              'portfolio_return': net_portfolio_return,
                              'raw_return': portfolio_return, 
                              'turnover': turnover, 
                              'transaction_cost': transaction_cost})

    # Update previous longs
    previous_longs_df = longs_df.copy()
    # Save weights and rebalance date for forward-filling later
    longs_df['date'] = rebalance_date
    portfolio_weight_history.append(longs_df.copy())

    if rebalance_date == rebalance_dates[-1]: print('Implement new portfolio on the trading day after:',rebalance_date,'\n', trades)


net_result_df = build_daily_net_portfolio_returns(rolling_df,portfolio_weight_history,rebalance_dates, n=n, transaction_cost_per_unit=transaction_cost)


sp500 = pd.read_csv('datasets/sp500_total_return.csv')
sp500['date'] = pd.to_datetime(sp500['date']).dt.tz_localize(None)
sp500 = sp500.sort_values('date')
sp500 = sp500[sp500['date'] >= '2025-06-03']

sp500['return'] = sp500['adjusted_close'].pct_change()
sp500['cumulative_return'] = (1 + sp500['return']).cumprod()

net_result_df['date'] = pd.to_datetime(net_result_df['date']).dt.tz_localize(None)
net_result_df = net_result_df[net_result_df['date'] >= '2025-06-03']

compare_df = pd.DataFrame()
compare_df['date'] = net_result_df['date']
compare_df = compare_df.merge(sp500[['date', 'return', 'cumulative_return']], on='date', how='left')
compare_df['cumulative_return'] = compare_df['cumulative_return'].ffill()
compare_df['pca_cumulative_return'] = net_result_df['cumulative_return'].values

# --- Normalize both to 1 on June 3, 2025 ---
start_sp500 = compare_df.loc[compare_df['date'] >= '2025-06-03', 'cumulative_return'].dropna().iloc[0]
start_pca = compare_df.loc[compare_df['date'] >= '2025-06-03', 'pca_cumulative_return'].dropna().iloc[0]

compare_df['cumulative_return'] /= start_sp500
compare_df['pca_cumulative_return'] /= start_pca

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(compare_df['date'], compare_df['pca_cumulative_return'], label='PCA Long-Only Strategy', linewidth=2)
plt.plot(compare_df['date'], compare_df['cumulative_return'], label='S&P 500', linewidth=2)
plt.title("PCA Factor Long-Only Portfolio vs S&P500 (rebased)")
plt.xlabel("Date")
plt.ylabel("Normalized Cumulative Return (Starting at 1.0)")
plt.legend()
plt.grid(True)
plt.show()


compare_df['pca_return'] = compare_df['pca_cumulative_return'].pct_change()
compare_df['sp500_return'] = compare_df['cumulative_return'].pct_change()

returns = compare_df.dropna(subset=['pca_return', 'sp500_return'])

# Annualize Sharpe ratio assuming 252 trading days
annualization_factor = np.sqrt(252)
pca_sharpe = returns['pca_return'].mean() / returns['pca_return'].std() * annualization_factor
sp500_sharpe = returns['sp500_return'].mean() / returns['sp500_return'].std() * annualization_factor

print(f"PCA Strategy Sharpe:  {pca_sharpe:.2f}")
print(f"S&P 500 Sharpe:       {sp500_sharpe:.2f}")