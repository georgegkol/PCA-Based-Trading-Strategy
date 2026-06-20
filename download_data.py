import sqlite3
import pandas as pd
from datetime import datetime
from functions.Data_download import eodhd_download_prices, download_sp500_total_return

DB_PATH = 'datasets/stock_data.db'
today = datetime.today().strftime('%Y-%m-%d')

conn = sqlite3.connect(DB_PATH)
latest_date = pd.read_sql("SELECT MAX(date) as max_date FROM stock_prices", conn).iloc[0]['max_date']
conn.close()

print(f"Database last updated: {latest_date}")
print(f"Downloading new data up to: {today}")

eodhd_download_prices(start_date=latest_date, end_date=today, db_path=DB_PATH, market=".US")
download_sp500_total_return(end_date=today)

print("Download complete.")
