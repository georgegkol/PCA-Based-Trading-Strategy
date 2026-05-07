import pandas as pd

def rsi(series, span=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    avg_gain = up.ewm(span=span, min_periods=1).mean()
    avg_loss = down.ewm(span=span, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-9)  # avoid division by zero
    rsi = rs / (1 + rs)
    return rsi

def macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return macd_line - signal_line

def rolling_beta(x, y, window=60):
    return x.rolling(window).cov(y) / y.rolling(window).var()

def ulcer_index(series, window=20):
    """
    Compute rolling Ulcer Index over a 1D pandas Series.
    Returns a pandas Series.
    computes how far the price has fallen from its highest point over the last n days.
    """
    drawdowns = 100 * (series / series.rolling(window).max() - 1)
    squared_drawdowns = drawdowns ** 2
    ulcer = squared_drawdowns.rolling(window).mean() ** 0.5
    return ulcer

def atr(high, low, close, n=14):
    """
    Calculate the Average True Range (ATR) over a rolling window.

    Parameters:
    high (pd.Series): Series of high prices
    low (pd.Series): Series of low prices
    close (pd.Series): Series of closing prices
    n (int): Rolling window length (default is 14)

    Returns:
    pd.Series: ATR values, same length as input, with NaN for the first n-1 periods

    Notes:
    - ATR is a volatility indicator that measures the average range between
      high and low prices, adjusted for gaps from the previous close.
    - It is commonly used in risk management, stop placement, and volatility analysis.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=n).mean()



