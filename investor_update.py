import json
import os
import smtplib
import csv
import requests
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD") or os.getenv("GMAIL_PASSWORD")
EODHD_API_KEY = os.getenv("API_KEY")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(PROJECT_DIR, 'portfolio_state.json')
INVESTORS_FILE = os.path.join(PROJECT_DIR, 'investors.csv')
SP500_FILE = os.path.join(PROJECT_DIR, 'datasets', 'sp500_total_return.csv')
LAST_EMAIL_FILE = os.path.join(PROJECT_DIR, 'last_investor_email.txt')
EMAIL_INTERVAL_DAYS = 3


def load_state():
    if not os.path.exists(STATE_FILE):
        raise FileNotFoundError("portfolio_state.json not found. Run live_runner.py first.")
    with open(STATE_FILE) as f:
        return json.load(f)


def load_investors():
    if not os.path.exists(INVESTORS_FILE):
        raise FileNotFoundError("investors.csv not found.")
    investors = []
    with open(INVESTORS_FILE, newline='') as f:
        for row in csv.DictReader(f):
            name = row.get('name', '').strip()
            email = row.get('email', '').strip()
            initial = float(row.get('initial_investment', 0))
            start = row.get('start_date', '').strip()
            if email and initial > 0 and start:
                investors.append({'name': name, 'email': email, 'initial_investment': initial, 'start_date': start})
    return investors


def cumulative_return(returns):
    result = 1.0
    for r in returns:
        result *= (1 + r)
    return result - 1


def sp500_return_for_period(start_date, end_date):
    df = pd.read_csv(SP500_FILE, parse_dates=['date'])
    df = df.sort_values('date')
    df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
    if len(df) < 2:
        return None
    return (df.iloc[-1]['adjusted_close'] - df.iloc[0]['adjusted_close']) / df.iloc[0]['adjusted_close']


def fetch_live_prices(tickers):
    today = datetime.today().strftime('%Y-%m-%d')
    prices = {}
    for ticker in tickers:
        url = (f"https://eodhd.com/api/eod/{ticker}.US"
               f"?from={today}&to={today}&period=d&api_token={EODHD_API_KEY}&fmt=json")
        resp = requests.get(url)
        if resp.status_code == 200 and resp.json():
            prices[ticker] = resp.json()[-1]['adjusted_close']
    return prices


def holding_return_since_rebalance(tickers, last_rebalance_str):
    start = last_rebalance_str
    today = datetime.today().strftime('%Y-%m-%d')
    returns = {}
    for ticker in tickers:
        url = (f"https://eodhd.com/api/eod/{ticker}.US"
               f"?from={start}&to={today}&period=d&api_token={EODHD_API_KEY}&fmt=json")
        resp = requests.get(url)
        if resp.status_code == 200 and len(resp.json()) >= 2:
            data = resp.json()
            start_price = data[0]['adjusted_close']
            end_price = data[-1]['adjusted_close']
            returns[ticker] = (end_price - start_price) / start_price
    return returns


def next_rebalance_date(last_rebalance_str, trading_days=20):
    nyse = mcal.get_calendar('NYSE')
    last = pd.Timestamp(last_rebalance_str)
    today = pd.Timestamp(datetime.today().date())
    schedule = nyse.schedule(start_date=last, end_date=today)
    elapsed = len(schedule) - 1
    remaining = max(0, trading_days - elapsed)
    if remaining == 0:
        return "Due now"
    future = nyse.schedule(start_date=today, end_date=today + pd.Timedelta(days=remaining * 2 + 10))
    days = future.index.tolist()
    if remaining <= len(days):
        return days[remaining - 1].strftime('%B %d, %Y')
    return "Soon"


def build_email(investor, state, live_returns):
    all_returns = state['portfolio_returns']
    last_rebalance = state['last_rebalance']
    start_date = investor['start_date']
    initial = investor['initial_investment']
    name = investor['name'] or 'Investor'

    investor_returns = [
        r for r in all_returns
        if r['portfolio_return'] != 0 and r['date'] >= start_date
    ]

    cum_return = cumulative_return([r['portfolio_return'] for r in investor_returns])
    current_value = initial * (1 + cum_return)

    recent_return = investor_returns[-1]['portfolio_return'] if investor_returns else 0
    recent_date = investor_returns[-2]['date'] if len(investor_returns) >= 2 else start_date

    sp500_since_inception = sp500_return_for_period(start_date, last_rebalance)
    sp500_recent = sp500_return_for_period(recent_date, last_rebalance)

    next_reb = next_rebalance_date(last_rebalance)

    holdings_rows = ''.join(
        f"<tr>"
        f"<td style='padding:8px 14px;border-bottom:1px solid #2a2a2a'>{h['ticker']}</td>"
        f"<td style='padding:8px 14px;border-bottom:1px solid #2a2a2a;text-align:right'>{h['weight']*100:.1f}%</td>"
        f"<td style='padding:8px 14px;border-bottom:1px solid #2a2a2a;text-align:right'>{fmt_pct_plain(live_returns.get(h['ticker']))}</td>"
        f"</tr>"
        for h in state['current_holdings']
    )

    def fmt_pct(val):
        if val is None:
            return '<span style="color:#6b7280">N/A</span>'
        color = '#4ade80' if val >= 0 else '#f87171'
        sign = '+' if val >= 0 else ''
        return f"<span style='color:{color};font-weight:700'>{sign}{val*100:.2f}%</span>"

    def fmt_pct_plain(val):
        if val is None:
            return '<span style="color:#6b7280">—</span>'
        color = '#4ade80' if val >= 0 else '#f87171'
        sign = '+' if val >= 0 else ''
        return f"<span style='color:{color}'>{sign}{val*100:.2f}%</span>"

    def fmt_money(val):
        return f"${val:,.2f}"

    import hashlib
    compliments = [
        "You have exceptional taste. In investments and, we can only assume, in life.",
        "Objectively one of the best-looking investors we've ever had. We don't make the rules.",
        "Smart, sophisticated, and frankly very attractive. The numbers below are almost a distraction.",
        "Not to be weird about it, but you are genuinely good-looking. Anyway, here are your returns.",
        "Whoever said money can't buy good looks clearly never met you. Here's your update.",
        "We ran the numbers. You're both a great investor and ridiculously good-looking. Coincidence? Probably.",
    ]
    idx = int(hashlib.md5(f"{name}{datetime.today().strftime('%Y-%m-%d')}".encode()).hexdigest(), 16) % len(compliments)
    compliment = compliments[idx]

    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0f0f0f;font-family:Arial,sans-serif;color:#e5e7eb">
  <div style="max-width:580px;margin:32px auto;background:#1a1a1a;border-radius:10px;overflow:hidden;border:1px solid #2a2a2a">

    <div style="background:#111827;padding:24px 28px;border-bottom:1px solid #2a2a2a">
      <p style="margin:0 0 4px;font-size:12px;color:#6b7280;letter-spacing:1px;text-transform:uppercase">PCA Trading Strategy</p>
      <h2 style="margin:0;font-size:22px;color:#f9fafb">Your Money. Working Hard.</h2>
      <p style="margin:6px 0 0;font-size:13px;color:#6b7280">Update as of {last_rebalance}</p>
    </div>

    <div style="padding:24px 28px">
      <p style="margin:0 0 8px;color:#d1d5db;font-size:15px">Hey {name},</p>
      <p style="margin:0 0 24px;color:#9ca3af;font-size:14px;line-height:1.6">{compliment}</p>

      <h3 style="margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#9ca3af">Your Portfolio</h3>
      <div style="background:#111827;border-radius:8px;padding:20px;margin-bottom:24px;border:1px solid #2a2a2a">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <span style="color:#9ca3af;font-size:14px">You put in</span>
          <span style="font-size:16px;font-weight:600;color:#f9fafb">{fmt_money(initial)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="color:#9ca3af;font-size:14px">It's now worth</span>
          <span style="font-size:22px;font-weight:700;color:#f9fafb">{fmt_money(current_value)}</span>
        </div>
      </div>

      <h3 style="margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#9ca3af">How We're Doing vs. The Market</h3>
      <table style="width:100%;border-collapse:collapse;background:#111827;border-radius:8px;overflow:hidden;border:1px solid #2a2a2a;margin-bottom:24px">
        <thead>
          <tr style="border-bottom:1px solid #2a2a2a">
            <th style="padding:10px 14px;text-align:left;font-size:12px;color:#6b7280;font-weight:500">Period</th>
            <th style="padding:10px 14px;text-align:right;font-size:12px;color:#6b7280;font-weight:500">Strategy</th>
            <th style="padding:10px 14px;text-align:right;font-size:12px;color:#6b7280;font-weight:500">S&amp;P 500</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid #2a2a2a">
            <td style="padding:10px 14px;font-size:14px">Last cycle</td>
            <td style="padding:10px 14px;text-align:right">{fmt_pct(recent_return)}</td>
            <td style="padding:10px 14px;text-align:right">{fmt_pct(sp500_recent)}</td>
          </tr>
          <tr>
            <td style="padding:10px 14px;font-size:14px">Since you joined ({start_date})</td>
            <td style="padding:10px 14px;text-align:right">{fmt_pct(cum_return)}</td>
            <td style="padding:10px 14px;text-align:right">{fmt_pct(sp500_since_inception)}</td>
          </tr>
        </tbody>
      </table>

      <h3 style="margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#9ca3af">What We're Holding Right Now</h3>
      <table style="width:100%;border-collapse:collapse;background:#111827;border-radius:8px;overflow:hidden;border:1px solid #2a2a2a;margin-bottom:24px">
        <thead>
          <tr style="border-bottom:1px solid #2a2a2a">
            <th style="padding:10px 14px;text-align:left;font-size:12px;color:#6b7280;font-weight:500">Ticker</th>
            <th style="padding:10px 14px;text-align:right;font-size:12px;color:#6b7280;font-weight:500">Weight</th>
            <th style="padding:10px 14px;text-align:right;font-size:12px;color:#6b7280;font-weight:500">Since rebalance</th>
          </tr>
        </thead>
        <tbody>{holdings_rows}</tbody>
      </table>

      <h3 style="margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#9ca3af">Next Move</h3>
      <div style="background:#111827;border-radius:8px;padding:16px;border:1px solid #2a2a2a;margin-bottom:24px">
        <span style="font-size:16px;font-weight:600;color:#f9fafb">{next_reb}</span>
        <span style="margin-left:12px;font-size:12px;color:#6b7280">— that's when the model picks again. Last rebalance: {last_rebalance}</span>
      </div>

      <p style="margin:0;font-size:11px;color:#4b5563;border-top:1px solid #2a2a2a;padding-top:16px;line-height:1.6">
        Returns are net of estimated transaction costs. Past performance is not indicative of future results.
        You're receiving this because you're an investor in the PCA strategy. Updates go out every few days.
      </p>
    </div>
  </div>
</body>
</html>"""


def send_email(investor, html_body):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Strategy Update — {datetime.today().strftime('%B %d, %Y')}"
    msg['From'] = GMAIL_EMAIL
    msg['To'] = investor['email']
    msg.attach(MIMEText(html_body, 'html'))
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)


def should_send_today():
    if not os.path.exists(LAST_EMAIL_FILE):
        return True
    with open(LAST_EMAIL_FILE) as f:
        last_sent = datetime.strptime(f.read().strip(), '%Y-%m-%d').date()
    return (datetime.today().date() - last_sent).days >= EMAIL_INTERVAL_DAYS


def mark_sent():
    with open(LAST_EMAIL_FILE, 'w') as f:
        f.write(datetime.today().strftime('%Y-%m-%d'))


def main():
    if not should_send_today():
        print("Less than 3 days since last email. Skipping.")
        return

    state = load_state()
    investors = load_investors()

    if not investors:
        print("No investors found in investors.csv.")
        return

    tickers = [h['ticker'] for h in state['current_holdings']]
    live_returns = holding_return_since_rebalance(tickers, state['last_rebalance'])
    print(f"Live returns fetched: {live_returns}")

    print(f"Sending updates to {len(investors)} investor(s)...")
    for investor in investors:
        html = build_email(investor, state, live_returns)
        send_email(investor, html)
        print(f"  Sent to {investor['name']} <{investor['email']}>")

    mark_sent()
    print("Done.")


if __name__ == '__main__':
    main()
