from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import zscore
import warnings
from numpy.linalg import norm
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def calculate_future_returns(rolling_df, holding_period_days):
    """
    Calculates forward returns from one rebalance date to the next for each ticker.
    Handles duplicate entries, missing prices, and logs dropped tickers.
    """

    rolling_df = rolling_df.copy()
    rolling_df['date'] = pd.to_datetime(rolling_df['date'])
    rolling_df = rolling_df.sort_values(['ticker', 'date'])

    # Deduplicate just in case
    rolling_df = rolling_df.drop_duplicates(subset=['ticker', 'date'])

    # Get every Nth trading day as rebalance date
    all_dates = rolling_df['date'].drop_duplicates().sort_values()
    rebalance_dates = all_dates[::holding_period_days].tolist()

    # Prices on rebalance dates
    price_df = rolling_df[rolling_df['date'].isin(rebalance_dates)][['ticker', 'date', 'adjusted_close']]
    price_df = price_df.drop_duplicates(subset=['ticker', 'date'])

    future_return_rows = []

    for i in range(len(rebalance_dates)-1):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i+1]

        start_df = price_df[price_df['date'] == start_date][['ticker', 'adjusted_close']].rename(columns={'adjusted_close': 'start_price'})
        end_df = price_df[price_df['date'] == end_date][['ticker', 'adjusted_close']].rename(columns={'adjusted_close': 'end_price'})

        merged = pd.merge(start_df, end_df, on='ticker', how='inner')
        merged['date'] = start_date
        merged['future_return'] = (merged['end_price'] / merged['start_price']) - 1

        dropped = len(start_df) - len(merged)
        #if dropped > 0:
        #    print(f"[{start_date.date()} → {end_date.date()}] Dropped {dropped} tickers due to missing prices.")

        future_return_rows.append(merged[['ticker', 'date', 'future_return']])

    # Combine all returns
    future_return_df = pd.concat(future_return_rows, ignore_index=True)

    # Merge into original
    merged_df = pd.merge(rolling_df, future_return_df, on=['ticker', 'date'], how='left')

    return rebalance_dates[:], merged_df




def filter_fundamentals(df, drop_bottom=0.4):
    fundamental_cols = ['bm', 'gprof', 'roe']  # replace with your actual fundamental columns

    # Ensure 'date' is present
    if 'date' not in df.columns:
        raise KeyError("'date' column is missing in the input DataFrame")

    # Z-score each fundamental by date
    for col in fundamental_cols:
        z_col = f"{col}_z"
        df[z_col] = df.groupby('date')[col].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

    z_cols = [f"{col}_z" for col in fundamental_cols]

    # Filter out bottom X% of each z-score individually
    for col in z_cols:
        # Get the percentile rank per date group
        ranks = df.groupby('date')[col].rank(pct=True)
        df = df[ranks > drop_bottom]

    return df


def pca_factor_loadings(df_slice, factor_cols, pcs_to_use=slice(None)):  
    """
    pcs_to_use: slice object to control which PCs to return, e.g. slice(2, None) to skip first 2 PCs.
    """
    X = df_slice[factor_cols].copy()
    X = X.dropna()
    df_slice = df_slice.loc[X.index]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -3, 3)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    loading_cols = [f'pca_{i}' for i in range(X_pca.shape[1])]
    loadings_df = pd.DataFrame(X_pca, columns=loading_cols, index=df_slice.index)
    loadings_df['ticker'] = df_slice['ticker'].values
    # Slice the columns based on pcs_to_use
    selected_cols = loading_cols[pcs_to_use]
    return loadings_df[['ticker'] + selected_cols]


def Ledoit_pca_factor_loadings(df_slice, factor_cols, pcs_to_use=slice(None)):
    X = df_slice[factor_cols].copy()
    X = X.dropna()
    df_slice = df_slice.loc[X.index]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -3, 3)

    lw = LedoitWolf().fit(X_scaled)
    cov_matrix = lw.covariance_

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    X_pca = X_scaled @ eigvecs  # project onto principal directions

    loading_cols = [f'pca_{i}' for i in range(X_pca.shape[1])]
    loadings_df = pd.DataFrame(X_pca[:, pcs_to_use], columns=loading_cols[pcs_to_use], index=df_slice.index)
    loadings_df['ticker'] = df_slice['ticker'].values

    return loadings_df[['ticker'] + loading_cols[pcs_to_use]]


def get_residuals(factor_loadings, returns_df):

    merged = pd.merge(factor_loadings, returns_df, on='ticker', how='inner')

    X = merged[[col for col in merged.columns if col.startswith('pca_')]].copy()
    y = merged['weekly_return_5d']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    merged['residual'] = model.resid

    return merged[['ticker', 'residual']]

def generate_long_only_signals(residuals_df, top_n=25, skip_bottom_n=25):
    #residuals_df['z_residual'] = (residuals_df['residual'] - residuals_df['residual'].mean()) / residuals_df['residual'].std()
    #residuals_df['z_residual'] = residuals_df['z_residual'].clip(-3, 3)

    longs = residuals_df[residuals_df['z_residual'] < 0].copy()
    longs = longs.sort_values('z_residual')

    # Skip the bottom skip_bottom_n (falling knives)
    longs = longs.iloc[skip_bottom_n:]


    # If top_n is set, select up to top_n from the sector-limited list
    if top_n is not None and len(longs) > top_n:
        longs = longs.head(top_n)

    # Assign equal weights
    longs['weight'] = 1.0 / len(longs)

    return longs[['ticker', 'weight']]



def calculate_portfolio_return(weights_df, future_returns_df, rebalance_date):
    merged = pd.merge(weights_df, future_returns_df, on='ticker', how='inner')
    merged['weighted_return'] = merged['weight'] * merged['future_return']
    portfolio_return = merged['weighted_return'].sum()

    return portfolio_return, merged


def calculate_turnover_and_cost(previous_weights_df, current_weights_df, transaction_cost_per_unit=0.001):
    """
    Returns:
    turnover: float, total turnover (sum of absolute weight changes)
    transaction_cost: float, total cost = turnover * cost per unit
    turnover_df: merged DataFrame showing per ticker changes (optional, useful for debugging)
    """

    merged = pd.merge(current_weights_df, previous_weights_df, on='ticker', how='outer', suffixes=('_current', '_prev'))
    # Fill NaN only for numeric columns
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    merged['weight_change'] = (merged['weight_current'] - merged['weight_prev']).abs()
    turnover = merged['weight_change'].sum()
    transaction_cost = turnover * transaction_cost_per_unit

    return turnover, transaction_cost, merged[['ticker', 'weight_prev', 'weight_current', 'weight_change']]

def regime_scaling(residuals_df, rebalance_date, longs_df, signal_history):
    """
    Adjusts the weights of the long positions based on the regime scaling.
    """
    # Calculate the signal strength
    residuals_df['residual'] = residuals_df['residual'].clip(-3, 3)
    signal_strength = residuals_df['residual'].std()
    signal_history.append(signal_strength)

    if len(signal_history) > 50:  # Need some history to calculate percentile
        threshold = np.percentile(signal_history[-50:], 20)  # Bottom 20%
        
        if signal_strength < threshold:
            regime_scaling = 0.5
            print('weak regime', rebalance_date, regime_scaling)
        else:
            regime_scaling = 1.0
    else:
        regime_scaling = 1.0  # Default to normal if no history yet
    
    return regime_scaling

def vol_scaling(portfolio_returns, target_vol = 0.01):
# Calculate realized volatility from past portfolio returns
    if len(portfolio_returns) >= 20:
        recent_returns = pd.Series([p['portfolio_return'] for p in portfolio_returns[-20:]])
        realized_vol = recent_returns.std()
        
        vol_scaler = target_vol / (realized_vol + 1e-9)  # avoid divide by zero
        vol_scaler = min(vol_scaler, 1.0)  # cap scaler to avoid overleverage
    else:
        vol_scaler = 1.0
    return vol_scaler

def composite_alpha(residuals_df, alpha_history):
    # --- Calculate z_residual (cross-sectional z-score of raw residuals) ---
    residuals_df['z_residual'] = (residuals_df['residual'] - residuals_df['residual'].mean()) / residuals_df['residual'].std()
    residuals_df['z_residual'] = residuals_df['z_residual'].clip(-3, 3)

    # --- Smooth z_residual using alpha_history (time-series smoothing) ---
    composite_alpha_list = []

    for _, row in residuals_df.iterrows():
        ticker = row['ticker']
        z_residual = row['z_residual']

        prev_alpha = alpha_history.get(ticker, z_residual)
        smoothed_alpha = 0.3 * prev_alpha + 0.7 * z_residual  # Smoothing factor
        alpha_history[ticker] = smoothed_alpha

        composite_alpha_list.append(smoothed_alpha)

    # Add smoothed alpha as residual (overwriting residual)
    residuals_df['composite_alpha'] = composite_alpha_list
    residuals_df['residual'] = residuals_df['composite_alpha']
    return residuals_df

def calculate_rebalance_trades(new_df, old_df):
    merged = pd.merge(new_df, old_df, on='ticker', how='outer', suffixes=('_new', '_old'))
    num_cols = merged.select_dtypes(include='number').columns
    merged[num_cols] = merged[num_cols].fillna(0)
    merged['weight_change'] = merged['weight_new'] - merged['weight_old']
    return merged[['ticker', 'weight_old', 'weight_new', 'weight_change']]

def neutralize_z_residual(residuals_df, df_slice, controls):
    """
    Regresses z_residual on control exposures (cross-sectionally),
    replaces 'z_residual' column in residuals_df with neutralized values.

    Parameters:
    - residuals_df: pd.DataFrame with columns ['ticker', 'z_residual', ...]
    - df_slice: pd.DataFrame with same-date data including control variables
    - controls: list of str, column names to regress out from z_residual

    Returns:
    - pd.DataFrame (same shape/columns as input), with 'z_residual' replaced
    """

    # Merge in the control exposures
    merged = pd.merge(
        residuals_df[['ticker', 'residual']],
        df_slice[['ticker'] + controls],
        on='ticker',
        how='left'
    )

    # Filter valid rows
    valid = merged['residual'].notna() & merged[controls].notna().all(axis=1)
    y = merged.loc[valid, 'residual']
    X = merged.loc[valid, controls]

    # Regress and get neutralized residuals
    reg = LinearRegression().fit(X, y)
    neutralized = y - reg.predict(X)

    # Create a mapping: ticker → neutralized z_residual
    ticker_to_neutral = dict(zip(merged.loc[valid, 'ticker'], neutralized))

    # Apply back to original residuals_df
    output = residuals_df.copy()
    output['residual'] = output['ticker'].map(ticker_to_neutral)

    return output



def build_daily_net_portfolio_returns(
    rolling_df,
    portfolio_weight_history,
    rebalance_dates,
    n,
    transaction_cost_per_unit=0.001,
    plot=True
):
    """
    Builds daily net portfolio returns by expanding rebalance weights, applying daily returns,
    and subtracting turnover-based transaction costs.
    
    Parameters:
        rolling_df (pd.DataFrame): Must include ['date', 'ticker', 'daily_return']
        portfolio_weight_history (List[pd.DataFrame]): List of rebalance weights (each with 'ticker', 'weight')
        rebalance_dates (List[pd.Timestamp]): All rebalance dates
        n (int): Warmup index (live trading starts from rebalance_dates[n])
        transaction_cost_per_unit (float): Cost per unit of turnover
        plot (bool): Whether to plot the result
        
    Returns:
        pd.DataFrame: daily net portfolio returns with columns ['date', 'net_return', 'cumulative_return']
    """

    # Step 1: Get all trading dates
    all_dates = rolling_df['date'].drop_duplicates().sort_values().reset_index(drop=True)

    # Step 2: Build rebalance windows (start/end per rebalance)
    rebalance_windows = []
    for i in range(n, len(portfolio_weight_history) + n):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else all_dates.max()
        rebalance_windows.append((start_date, end_date))

    # Step 3: Expand weights over all applicable days
    weight_rows = []
    for weights_df, (start_date, end_date) in zip(portfolio_weight_history, rebalance_windows):
        for single_date in all_dates[(all_dates >= start_date) & (all_dates < end_date)]:
            temp = weights_df.copy()
            temp['date'] = single_date
            weight_rows.append(temp)
    weights_daily_df = pd.concat(weight_rows, ignore_index=True)

    # Step 4: Merge with daily returns
    daily_returns = rolling_df[['date', 'ticker', 'daily_return']].copy()
    merged = pd.merge(daily_returns, weights_daily_df, on=['date', 'ticker'], how='left')
    merged['weight'] = merged['weight'].fillna(0)

    # Step 5: Compute raw daily return
    merged['weighted_return'] = merged['weight'] * merged['daily_return']
    raw_daily_return = merged.groupby('date')['weighted_return'].sum().reset_index()
    raw_daily_return.rename(columns={'weighted_return': 'raw_return'}, inplace=True)

    # Step 6: Calculate turnover and transaction cost per rebalance
    cost_log = {}
    previous_weights = pd.DataFrame(columns=['ticker', 'weight'])

    for i, rebalance_date in enumerate(rebalance_dates[n : n + len(portfolio_weight_history)]):
        current_weights = portfolio_weight_history[i]
        turnover, cost, _ = calculate_turnover_and_cost(previous_weights, current_weights, transaction_cost_per_unit)
        cost_log[rebalance_date] = cost
        previous_weights = current_weights.copy()

    # Step 7: Map cost to daily dates
    raw_daily_return['transaction_cost'] = raw_daily_return['date'].map(cost_log).fillna(0.0)
    raw_daily_return['net_return'] = raw_daily_return['raw_return'] - raw_daily_return['transaction_cost']

    # Step 8: Trim to live period
    start_date = rebalance_dates[n]
    result = raw_daily_return[raw_daily_return['date'] >= start_date].copy()
    result['cumulative_return'] = (1 + result['net_return']).cumprod()
    result['cumulative_return'] /= result['cumulative_return'].iloc[0]

    # Step 9: Plot (optional)
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(result['date'], result['cumulative_return'], label='Net Portfolio Return')
        plt.title("PCA Factor Portfolio - Net Daily Cumulative Return (After Turnover Costs)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.legend()
        plt.show()

    return result[['date', 'net_return', 'cumulative_return']]
