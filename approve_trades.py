import json
import os
from ibkr_executor import execute_rebalance

TRADES_FILE = 'pending_trades.json'

if not os.path.exists(TRADES_FILE):
    print("No pending trades found. Run live_runner.py first.")
    exit()

with open(TRADES_FILE, 'r') as f:
    pending = json.load(f)

sells = pending['sells']
buys  = pending['buys']
holds = pending['holds']
date  = pending['date']

print(f"\n========================================")
print(f"  Pending trades from: {date}")
print(f"========================================")
print(f"  SELL : {', '.join(sells) if sells else 'none'}")
print(f"  BUY  : {', '.join(buys)  if buys  else 'none'}")
print(f"  HOLD : {', '.join(holds) if holds else 'none'}")
print(f"========================================\n")

approval = input("Execute these trades via IBKR? (y/n): ").strip().lower()
if approval == 'y':
    execute_rebalance(sells, buys)
    os.remove(TRADES_FILE)
    print("Trades executed and pending file cleared.")
else:
    print("Cancelled. No orders placed. pending_trades.json kept for next time.")
