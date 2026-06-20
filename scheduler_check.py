import os
import sys
import subprocess
import pandas as pd
import exchange_calendars as xcals
from datetime import datetime

LAST_REBALANCE_FILE = 'last_rebalance.txt'
TRADING_DAYS_THRESHOLD = 20
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# ─── Read last rebalance date ────────────────────────────────────────────────
if not os.path.exists(LAST_REBALANCE_FILE):
    print("No last_rebalance.txt found. Running live_runner.py for the first time.")
    subprocess.run([PYTHON, os.path.join(PROJECT_DIR, 'live_runner.py')], cwd=PROJECT_DIR)
    exit()

with open(LAST_REBALANCE_FILE, 'r') as f:
    last_rebalance = pd.Timestamp(f.read().strip())

today = pd.Timestamp(datetime.today().date())

# ─── Count exact NYSE trading days since last rebalance ─────────────────────
nyse = xcals.get_calendar('XNYS')
trading_days_elapsed = len(nyse.sessions_in_range(last_rebalance, today)) - 1

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
