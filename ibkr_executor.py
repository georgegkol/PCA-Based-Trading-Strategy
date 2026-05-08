from ib_insync import IB, Stock, MarketOrder
import time


def execute_rebalance(sells, buys):
    """
    Sells previous holdings and buys new ones with equal cash allocation.
    Requires IB Gateway running on localhost:4001 (live account).
    """
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1)
    print("\n Connected to IB Gateway.")

    # --- Get current positions from IBKR ---
    positions = ib.positions()
    pos_dict = {p.contract.symbol: int(p.position) for p in positions}

    # --- Step 1: Place sell orders ---
    sell_trades = []
    print("\n--- SELLS ---")
    for ticker in sells:
        qty = pos_dict.get(ticker, 0)
        if qty <= 0:
            print(f"  {ticker}: no position found in IBKR, skipping")
            continue
        contract = Stock(ticker, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        order = MarketOrder('SELL', qty)
        trade = ib.placeOrder(contract, order)
        sell_trades.append((ticker, qty, trade))
        print(f"  Sell {qty} shares of {ticker}")

    # --- Step 2: Wait for sell fills (up to 60 seconds) ---
    print("\nWaiting for sell orders to fill...")
    timeout = 60
    start = time.time()
    while time.time() - start < timeout:
        ib.sleep(2)
        filled = [t for _, _, t in sell_trades if t.orderStatus.status == 'Filled']
        if len(filled) == len(sell_trades):
            break
    unfilled = [tk for tk, _, t in sell_trades if t.orderStatus.status != 'Filled']
    if unfilled:
        print(f"  WARNING: These sell orders did not fill within timeout: {unfilled}")

    # --- Step 3: Calculate total proceeds from filled sells ---
    total_proceeds = sum(
        qty * t.orderStatus.avgFillPrice
        for _, qty, t in sell_trades
        if t.orderStatus.status == 'Filled'
    )
    print(f"\n  Total proceeds from sells: ${total_proceeds:,.2f}")

    if not buys:
        print("No new stocks to buy.")
        ib.disconnect()
        return

    cash_per_stock = total_proceeds / len(buys)
    print(f"  Cash per new position: ${cash_per_stock:,.2f}")

    # --- Step 4: Buy new positions ---
    print("\n--- BUYS ---")
    for ticker in buys:
        contract = Stock(ticker, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        [ticker_data] = ib.reqTickers(contract)
        price = ticker_data.marketPrice()

        if not price or price != price:  # guard against NaN
            print(f"  {ticker}: could not get market price, skipping")
            continue

        shares = int(cash_per_stock / price)
        if shares <= 0:
            print(f"  {ticker}: calculated 0 shares at ${price:.2f}, skipping")
            continue

        order = MarketOrder('BUY', shares)
        ib.placeOrder(contract, order)
        print(f"  Buy {shares} shares of {ticker} @ ~${price:.2f}  (${shares * price:,.2f})")

    ib.sleep(5)
    ib.disconnect()
    print("\n Done. Disconnected from IB Gateway.")
