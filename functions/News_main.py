import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import timedelta
import re
from tqdm import tqdm 

from News_download import download_and_store_news
from Data_download import tiingo_download_prices, get_sp500_index_data
from News_preprocessing import preprocess_prices, get_prices, get_news
from News_core_functions import build_full_scored_news_df, build_finbert_pipeline, run_backtest_cumulative, plot_strategy_vs_sp500

last_date = '2025-05-24'
today= '2025-05-29'

def run_agent(previous_date, today):
    # Download and store news articles
    download_and_store_news(previous_date, today)
    print("News data downloaded and stored successfully.")
    df_sp500 = get_sp500_index_data('2005-01-01', today)
    df_sp500.to_csv('datasets/sp500_total_return.csv', index=False)
    tiingo_download_prices(previous_date, today, db_path="datasets/stock_data.db")
    print("Stock prices downloaded and stored successfully.")

    # Preprocess prices and news
    df_2025 = get_prices(previous_date, today)
    df_2025 = preprocess_prices(df_2025, volume_upper=1e6, volume_lower=400000, date_cutoff='2025-02-24')
    news_df = get_news(previous_date, today)
    news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'], format='mixed', utc=True)
    news_df['publishedDate'] = news_df['publishedDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    news_df['ticker'] = news_df['ticker'].str.upper()
    valid_tickers = df_2025['ticker'].str.upper().unique()
    news_df = news_df[news_df['ticker'].isin(valid_tickers)].copy()
    print("News and prices processed successfully.")

    
    # Score news articles
    news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'], utc=True)
    trading_dates = df_2025['date'].drop_duplicates()
    finbert = build_finbert_pipeline()
    top_scored_df = build_full_scored_news_df(news_df, finbert, trading_dates)

    # Backtest
    result_df = run_backtest_cumulative(df_2025, top_scored_df, score_threshold=4, top_n=2, rebalance_interval=5)
    plot_strategy_vs_sp500(result_df)


if __name__ == "__main__":
    run_agent()




