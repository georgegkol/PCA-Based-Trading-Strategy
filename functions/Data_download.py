import pandas as pd
import requests
import sqlite3
from datetime import datetime
import time

from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

API_KEY = os.getenv("API_KEY")


data_split = {"Train": {"Years": 10,"Period": "2006–2015"},
              "Validation": {"Years": 4,"Period": "2016–2019"},
              "Test": {"Years": 4,"Period": "2020–2025"}}



def us_source_tickers(tickers_df,):
    nyse_tickers_df = tickers_df[
        (tickers_df['Exchange'] == 'NYSE') &
        (tickers_df['Type'] == 'Common Stock')
    ]
    tickers = nyse_tickers_df['Code'].tolist()
    clean_tickers = []

    for t in tickers:
        t = str(t)
        if "-" in t or "." in t:
            continue
        if t[0].isdigit() or t[-1].isdigit():
            continue
        if t.endswith("Q"):
            continue
        if any(c in t for c in "()/" ):
            continue
        if any(x in t for x in ["-P-", "-U", "-WS", "-W"]):
            continue
        clean_tickers.append(t)

    return clean_tickers

def gr_source_tickers(tickers_df):
        stocks_df = tickers_df[tickers_df['Type'] == 'Common Stock']
        tickers = stocks_df['Code'].tolist()
        clean_tickers = []

        for t in tickers:
            t = str(t)
            if "-" in t or "." in t:
                continue
            if t[0].isdigit() or t[-1].isdigit():
                continue
            if t.endswith("Q"):
                continue
            if any(c in t for c in "()/" ):
                continue
            if any(x in t for x in ["-P-", "-U", "-WS", "-W"]):
                continue
            clean_tickers.append(t)

        return clean_tickers



# === EODHD Download Function ===
def get_eodhd_adjusted_prices(ticker, start_date, end_date, market=".US"):
    full_ticker = ticker + market
    url = (
        f"https://eodhd.com/api/eod/{full_ticker}"
        f"?from={start_date}&to={end_date}&period=d&api_token={API_KEY}&fmt=json"
    )

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed: {full_ticker} — {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    if not data:
        print(f"No data for {full_ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['ticker'] = ticker

    try:
        df = df[[
            'date', 'ticker',
            'open', 'high', 'low', 'close',
            'adjusted_close',
            'volume'
        ]]
    except KeyError:
        print(f"Missing expected columns for {ticker}")
        return pd.DataFrame()

    return df


def eodhd_download_prices(start_date='2000-01-01', end_date=None, db_path="datasets/stock_data.db", market=".US", rate_limit=0.5):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch full US symbol list
    if market == ".US":
        url = f"https://eodhd.com/api/exchange-symbol-list/US?api_token={API_KEY}&fmt=json"
    elif market==".AT":
        url = f"https://eodhd.com/api/exchange-symbol-list/AT?api_token={API_KEY}&fmt=json"

    resp = requests.get(url)

    # Debugging output to understand failure
    print("📡 Status Code:", resp.status_code)
    print("📄 Content Preview:", resp.text[:300])

    if resp.status_code == 200:
        try:
            data = resp.json()
            symbols_df = pd.DataFrame(data)
        except ValueError:
            print("❌ JSON decode error. Raw response:")
            print(resp.text[:300])  # limit to 300 chars
            return
    else:
        print(f"❌ Failed to fetch symbol list: {resp.status_code}")
        print(resp.text)
        return


    if market == ".US":
        nyse_tickers = us_source_tickers(symbols_df)
    elif market==".AT":
        nyse_tickers = gr_source_tickers(symbols_df)
    print("Number of common stocks:", len(nyse_tickers))

    all_data = []

    for i, ticker in enumerate(nyse_tickers, start=1):
        print(f"[{i}/{len(nyse_tickers)}] Downloading {ticker}")
        df = get_eodhd_adjusted_prices(ticker, start_date, end_date, market)
        if not df.empty:
            all_data.append(df)
        time.sleep(rate_limit)

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)

        conn = sqlite3.connect(db_path)
        result_df.to_sql("stock_prices", conn, if_exists="append", index=False)
        conn.close()

        print(f"Appended {len(result_df)} rows to database: {db_path}")
    else:
        print("No data downloaded.")

    print("DONE.")


def download_sp500_total_return(end_date, start_date='2005-01-01', output_csv="datasets/sp500_total_return.csv"):
    symbol = "SPY.US" # S&P 500 Total Return Index symbol in EODHD
    url = (
        f"https://eodhd.com/api/eod/{symbol}"
        f"?from={start_date}&to={end_date}&period=d&api_token={API_KEY}&fmt=json"
    )

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return

    data = response.json()
    if not data:
        print("No data received.")
        return

    df = pd.DataFrame(data)
    df = df[['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
    df.to_csv(output_csv, index=False)
    print(f"S&P 500 Total Return saved to {output_csv}")




def get_dataset_split(split, database = "datasets/stock_data.db"):
    # Define split year ranges
    split_years = {
        "train": ("2005-01-01", "2015-12-31"),
        "validation": ("2016-01-01", "2019-12-31"),
        "test": ("2020-01-01", "2027-12-31")
    }

    # Validate input
    if split not in split_years:
        raise ValueError("split must be 'train', 'validation', or 'test'")

    # Connect to the database
    conn = sqlite3.connect(database)

    start_date, end_date = split_years[split]

    # Query the relevant date range
    query = f"""
        SELECT date, close, open, high, low, adjusted_close, volume, ticker
        FROM stock_prices
        WHERE date(date) BETWEEN date('{start_date}') AND date('{end_date}')
    """

    df = pd.read_sql_query(query, conn, parse_dates=["date"])
    conn.close()
    df['date'] = pd.to_datetime(df['date'])

    return df



def filter_volume_lowprices_availtickers(df_prices, volume_threshold=600000, remove_price_below=5, volume_window=20, database="datasets/stock_data.db"):

    df_prices = df_prices.sort_values(by=['ticker', 'date'])
    df_prices = df_prices.drop_duplicates(subset=['ticker', 'date'], keep='first')
    print("Duplicates removed. New shape:", df_prices.shape)

    # Rolling average volume (only past data)
    df_prices['avg_volume'] = (
        df_prices.groupby('ticker')['volume']
        .transform(lambda x: x.shift(1).rolling(volume_window, min_periods=1).mean())
    )

    # Mark dates where avg_volume or price drops below threshold
    df_prices['volume_flag'] = df_prices['avg_volume'] < volume_threshold
    df_prices['price_flag'] = df_prices['close'] < remove_price_below

    # Find first date where each ticker violates a rule
    ticker_cutoff = df_prices[df_prices['volume_flag'] | df_prices['price_flag']].groupby('ticker')['date'].min()

    # Merge cutoff dates back
    df_prices = df_prices.merge(ticker_cutoff.rename('cutoff_date'), on='ticker', how='left')

    # Remove rows on or after the violation date
    df_prices = df_prices[
        (df_prices['cutoff_date'].isna()) | (df_prices['date'] < df_prices['cutoff_date'])
    ].copy()

    print(f"Number of tickers after time-aware filters: {df_prices['ticker'].nunique()}")

    if database=="datasets/stock_data.db":
        # Filter by available ticker-sector info
        tickers = pd.read_csv('datasets/tickers.csv')
        df_prices = df_prices[df_prices['ticker'].isin(tickers['ticker'])].copy()

        print(f"Number of tickers after ticker-sector pairs filter: {df_prices['ticker'].nunique()}")
    
        return df_prices, tickers

    return df_prices, None
