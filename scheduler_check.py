import os
import subprocess
import pandas_market_calendars as mcal
import pandas as pd
from datetime import datetime

LAST_REBALANCE_FILE = 'last_rebalance.txt'
TRADING_DAYS_THRESHOLD = 20
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(os.path.dirname(os.__file__), '..', 'python.exe')  # Anaconda python

# ─── Read last rebalance date ────────────────────────────────────────────────
if not os.path.exists(LAST_REBALANCE_FILE):
    print("No last_rebalance.txt found. Running live_runner.py for the first time.")
    subprocess.run([PYTHON, os.path.join(PROJECT_DIR, 'live_runner.py')], cwd=PROJECT_DIR)
    exit()

with open(LAST_REBALANCE_FILE, 'r') as f:
    last_rebalance = pd.Timestamp(f.read().strip())

today = pd.Timestamp(datetime.today().date())

# ─── Count NYSE trading days since last rebalance ────────────────────────────
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=last_rebalance, end_date=today)
trading_days_elapsed = len(schedule) - 1  # exclude the rebalance day itself

print(f"Last rebalance : {last_rebalance.date()}")
print(f"Today          : {today.date()}")
print(f"Trading days elapsed: {trading_days_elapsed}")

# ─── Run live_runner if 20 trading days have passed ──────────────────────────
if trading_days_elapsed >= TRADING_DAYS_THRESHOLD:
    print("20 trading days reached. Running live_runner.py...")
    subprocess.run([PYTHON, os.path.join(PROJECT_DIR, 'live_runner.py')], cwd=PROJECT_DIR)
else:
    remaining = TRADING_DAYS_THRESHOLD - trading_days_elapsed
    print(f"Next rebalance in {remaining} trading day(s). Nothing to do.")
