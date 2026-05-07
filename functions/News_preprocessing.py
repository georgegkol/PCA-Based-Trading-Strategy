import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import timedelta
import sqlite3

TIINGO_API_KEY = "0f90d2d96f9bc2a770d3e42b010a7efd1d394b83"

def get_prices(start_date, end_date):

    conn = sqlite3.connect("datasets/stock_data.db")

    # Query the relevant date range
    query = f"""
        SELECT date, close, adjClose, adjHigh, adjLow, adjOpen, adjVolume, ticker
        FROM stock_prices
        WHERE date(date) BETWEEN date('{start_date}') AND date('{end_date}')
    """

    df = pd.read_sql_query(query, conn, parse_dates=["date"])
    conn.close()
    df['date'] = pd.to_datetime(df['date'])

    return df


def get_news(start_date, end_date):
    conn = sqlite3.connect("datasets/news_data.db")

    query = f"""
        SELECT *
        FROM news_articles
        WHERE date(publishedDate) BETWEEN date('{start_date}') AND date('{end_date}')
    """

    df = pd.read_sql_query(query, conn, parse_dates=["publishedDate"])
    conn.close()
    df['publishedDate'] = pd.to_datetime(df['publishedDate'])
    return df

def preprocess_prices(df, volume_upper, volume_lower, date_cutoff):
    # Standardize datetime
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.sort_values(['ticker', 'date'])

    # Compute rolling average of adjVolume
    df['rolling_avg_vol'] = df.groupby('ticker')['adjVolume'].transform(lambda x: x.rolling(window=60, min_periods=1).mean())
    df = df[df['date'] > date_cutoff]

    # Get earliest violation per ticker: either too high or too low
    high_vol_cutoffs = df[df['rolling_avg_vol'] > volume_upper].groupby('ticker')['date'].min()
    low_vol_cutoffs = df[df['rolling_avg_vol'] < volume_lower].groupby('ticker')['date'].min()

    # Combine and get earliest of the two
    cutoff_dates = pd.concat([high_vol_cutoffs, low_vol_cutoffs], axis=1)
    cutoff_dates.columns = ['high_vol_cutoff', 'low_vol_cutoff']
    cutoff_dates['cutoff'] = cutoff_dates.min(axis=1)

    # Filter function to keep only rows before the cutoff
    def keep_row(row):
        ticker = row['ticker']
        if ticker in cutoff_dates.index:
            return row['date'] < cutoff_dates.loc[ticker, 'cutoff']
        return True

    # Apply filter
    df = df[df.apply(keep_row, axis=1)]
    df = df.drop(columns='rolling_avg_vol')
    df = df.drop_duplicates(subset=['ticker', 'date'], keep='first')
    return df