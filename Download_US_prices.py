import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from functions.Data_download import eodhd_download_prices, download_sp500_total_return

pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_colwidth', 100) 
pd.set_option('display.width', 1000)

previous= '2025-04-19' #still start from here cause it wasnt completed before
today='2026-05-06'

eodhd_download_prices(start_date=previous, end_date=today, db_path='datasets/stock_data.db', market=".US")

download_sp500_total_return(end_date = today)


conn = sqlite3.connect("datasets/stock_data.db")

df = pd.read_sql("""
    SELECT *
    FROM stock_prices
    WHERE date >= '2025-06-29' AND date <= '2030-12-31'
""", conn)

conn.close()