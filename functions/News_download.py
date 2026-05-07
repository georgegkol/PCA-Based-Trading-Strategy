import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import sqlite3

TIINGO_API_KEY = "0f90d2d96f9bc2a770d3e42b010a7efd1d394b83"

last_date = '2025-05-24'
today= '2025-05-29'

def get_tiingo_news(ticker, start_date=None, end_date=None):
    url = "https://api.tiingo.com/tiingo/news"
    headers = {'Authorization': f'Token {TIINGO_API_KEY}'}
    params = {
        'tickers': ticker,
        'startDate': start_date,
        'endDate': end_date,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        print("Error:", response.status_code, response.text)
        return pd.DataFrame()
    


def download_and_store_news(start_date, end_date, db_path="datasets/news_data.db"):
    # Get unique tickers from your stock price database
    conn_prices = sqlite3.connect("datasets/stock_data.db")
    df_tickers = pd.read_sql("SELECT DISTINCT ticker FROM stock_prices", conn_prices)
    conn_prices.close()

    unique_tickers = df_tickers['ticker'].unique()
    news_dataframes = []

    for ticker in unique_tickers:
        print(f"Fetching news for {ticker} from {start_date} to {end_date}...")
        news_df = get_tiingo_news(ticker, start_date=start_date, end_date=end_date)

        if not news_df.empty:
            news_df['ticker'] = ticker

            # Drop unsupported columns
            news_df = news_df.drop(columns=["tags", "tickers"], errors="ignore")

            # Parse and validate publishedDate
            news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'], format='ISO8601', errors='coerce', utc=True)
            news_df = news_df.dropna(subset=['publishedDate'])  # Optional: drop bad rows
            news_df['publishedDate'] = news_df['publishedDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

            # Optional: Format crawlDate
            if 'crawlDate' in news_df.columns:
                news_df['crawlDate'] = pd.to_datetime(news_df['crawlDate'], format='ISO8601', errors='coerce', utc=True)
                news_df['crawlDate'] = news_df['crawlDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

            news_dataframes.append(news_df)

    if news_dataframes:
        all_news = pd.concat(news_dataframes, ignore_index=True)

        # Save to database
        conn = sqlite3.connect(db_path)
        all_news.to_sql("news_articles", conn, if_exists="append", index=False)
        conn.close()

        print(f"Appended {len(all_news)} news articles to {db_path}")
    else:
        print("No news data found.")

    print("DONE.")

