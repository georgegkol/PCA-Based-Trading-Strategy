import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import timedelta
import re
from tqdm import tqdm 


def build_finbert_pipeline():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    return finbert

def apply_sentiment_rules(titles, rules):
    results = []
    fallback_needed = [True] * len(titles)

    for i, title in enumerate(titles):
        title_lower = title.lower()
        sentiment_label = None
        sentiment_points = 0

        for keywords, label, points in rules:
            if all(kw in title_lower for kw in keywords):
                sentiment_label = label
                sentiment_points = points
                fallback_needed[i] = False
                break

        results.append({
            "index": i,
            "title": title,
            "sentiment_label": sentiment_label,
            "sentiment_points": sentiment_points,
            "fallback_needed": fallback_needed[i]
        })

    return results

def apply_finbert_fallback(results, finbert_pipeline):
    fallback_titles = [r["title"] for r in results if r["fallback_needed"]]
    fallback_indices = [r["index"] for r in results if r["fallback_needed"]]

    if fallback_titles:
        sentiments = finbert_pipeline(fallback_titles)
        for idx, sentiment in zip(fallback_indices, sentiments):
            label = sentiment["label"].upper()
            points = {"POSITIVE": 2, "NEUTRAL": 0, "NEGATIVE": 0}.get(label, 0)
            results[idx]["sentiment_label"] = label
            results[idx]["sentiment_points"] = points
            results[idx]["fallback_needed"] = False

    return results

def compute_final_scores(news_df, recent_news, older_news, results):
    output = []

    for i, (_, row) in enumerate(recent_news.iterrows()):
        r = results[i]
        score = r["sentiment_points"]
        ticker = row["ticker"]
        title = r["title"]

        if older_news[older_news['ticker'] == ticker].empty and score > 0:
            score += 1  # Freshness bonus

        if len(recent_news[recent_news['ticker'] == ticker]) > 1 and score > 0:
            score += 0  # Clustering bonus

        if score > 0 and ticker.lower() in title.lower():
            score += 1  # Ticker in title

        output.append({
            "ticker": ticker,
            "publishedDate": row["publishedDate"],
            "title": title,
            "sentiment": r["sentiment_label"],
            "score": score
        })

    return pd.DataFrame(output)

def score_news_stories(news_df, finbert_pipeline, lookback_days=5):
    if news_df.empty:
        return pd.DataFrame()

    news_df = news_df.copy()
    news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'], utc=True)
    news_df['ticker'] = news_df['ticker'].astype(str).str.upper()

    most_recent_date = news_df['publishedDate'].max().normalize()
    cutoff_date = most_recent_date - timedelta(days=lookback_days)

    recent_news = news_df[news_df['publishedDate'] >= most_recent_date]
    older_news = news_df[(news_df['publishedDate'] < most_recent_date) & (news_df['publishedDate'] >= cutoff_date)]

    if recent_news.empty: return pd.DataFrame()

    titles = recent_news['title'].tolist()
    sentiment_rules = [
        (["beats", "earnings"], "POSITIVE", 3), (["misses", "earnings"], "NEGATIVE", 0),
        (["raises", "guidance"], "POSITIVE", 3), (["cuts", "forecast"], "NEGATIVE", 0),
        (["takes", "position"], "POSITIVE", 3), (["purchases", "shares"], "POSITIVE", 3),
        (["declares", "dividend"], "POSITIVE", 3),
        (["announces", "buyback"], "POSITIVE", 3),
        (["stock", "surges"], "POSITIVE", 3), (["stock", "plunges"], "NEGATIVE", 0),
        (["soars"], "POSITIVE", 3), (["skyrockets"], "POSITIVE", 3), (["increases"], "POSITIVE", 3),
        (["tumbles"], "NEGATIVE", 0), (["decreases"], "NEGATIVE", 0),
        (["strong"], "POSITIVE", 3), (["weak"], "NEGATIVE", 0), 
        (["slashes", "dividend"], "NEGATIVE", 0),
    ]

    results = apply_sentiment_rules(titles, sentiment_rules)
    results = apply_finbert_fallback(results, finbert_pipeline)
    return compute_final_scores(news_df, recent_news, older_news, results)


def build_full_scored_news_df(news_df, finbert_pipeline, trading_dates):
    results = []

    for trading_date in tqdm(sorted(trading_dates.unique())):
        # Define window: from t-1 08:00 to t 08:00 UTC
        window_end = pd.Timestamp(trading_date).replace(hour=8, minute=0, tzinfo=pd.Timestamp.utcnow().tz)
        window_start = window_end - pd.Timedelta(hours=24)

        news_window = news_df[
            (news_df['publishedDate'] >= window_start) &
            (news_df['publishedDate'] < window_end)
        ]

        if news_window.empty:
            continue

        # Score news in this window
        scored_window = score_news_stories(news_window, finbert_pipeline=finbert_pipeline, lookback_days=5)

        if scored_window.empty:
            continue

        # Sort and get top score per ticker
        top_scores = (
            scored_window.sort_values(['ticker', 'score'], ascending=[True, False])
                         .groupby('ticker')
                         .first()
                         .reset_index()
        )
        top_scores['date'] = trading_date
        results.append(top_scores)

    final_df = pd.concat(results, ignore_index=True)
    return final_df[['date', 'ticker', 'score', 'title', 'publishedDate', 'sentiment']]



def run_backtest_cumulative(df_prices, scored_df, score_threshold=4, top_n=5, rebalance_interval=5):
    # Preprocess price data
    price_df = df_prices[["date", "ticker", "adjClose"]].copy()
    price_df = price_df.sort_values(["ticker", "date"])
    price_pivot = price_df.pivot(index="date", columns="ticker", values="adjClose").ffill()
    price_returns = price_pivot.pct_change().fillna(0)  # Daily returns

    rebalance_dates = price_returns.index[::rebalance_interval]

    portfolio_returns = []
    holdings = {}

    for i, date in enumerate(price_returns.index):
        if date in rebalance_dates:
            # Rebalance portfolio
            candidates = scored_df[scored_df["date"] == date]
            candidates = candidates[candidates["score"] >= score_threshold]

            # Exclude tickers with >8% daily return in any of the previous 5 days
            date_index = price_returns.index.get_loc(date)
            if date_index >= 5:
                lookback_dates = price_returns.index[date_index - 5:date_index]
                filtered_tickers = []
                for ticker in candidates["ticker"]:
                    if ticker in price_returns.columns:
                        if (price_returns.loc[lookback_dates, ticker] > 0.08).any():
                            continue
                        filtered_tickers.append(ticker)
                candidates = candidates[candidates["ticker"].isin(filtered_tickers)]

            # Select top N tickers
            candidates = candidates.sort_values("score", ascending=False).head(top_n)
            print(candidates)
            tickers = candidates["ticker"].tolist()

            if tickers:
                holdings = {ticker: 1 / len(tickers) for ticker in tickers}
            # else: keep previous holdings (do nothing)

        # Calculate weighted daily return with current holdings
        daily_return = 0
        for ticker, weight in holdings.items():
            if ticker in price_returns.columns:
                daily_return += weight * price_returns.loc[date, ticker] - 0.001  # transaction cost

        portfolio_returns.append((date, daily_return))

    # Create result DataFrame
    result_df = pd.DataFrame(portfolio_returns, columns=["date", "daily_return"])
    result_df["cumulative_return"] = (1 + result_df["daily_return"]).cumprod()
    return result_df


def plot_strategy_vs_sp500(result_df):
    # Load and prepare S&P 500 data
    sp500 = pd.read_csv('datasets/sp500_total_return.csv')
    sp500['date'] = pd.to_datetime(sp500['date'])
    sp500 = sp500.sort_values('date')
    sp500['return'] = sp500['adjClose'].pct_change()
    sp500['sp500_cumulative_return'] = (1 + sp500['return']).cumprod()
    sp500['date'] = sp500['date'].dt.tz_localize(None)  # Ensure datetime compatibility

    # Prepare strategy results
    result_df['date'] = pd.to_datetime(result_df['date']).dt.tz_localize(None)

    # Merge and compare
    compare_df = result_df.merge(sp500[['date', 'sp500_cumulative_return']], on='date', how='left')
    compare_df['sp500_cumulative_return'] = compare_df['sp500_cumulative_return'].ffill()

    # Normalize returns
    compare_df['strategy_cumulative_return_norm'] = compare_df['cumulative_return'] / compare_df['cumulative_return'].iloc[0]
    compare_df['sp500_cumulative_return_norm'] = compare_df['sp500_cumulative_return'] / compare_df['sp500_cumulative_return'].iloc[0]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(compare_df['date'], compare_df['strategy_cumulative_return_norm'], label='Sentiment-Based Strategy', linewidth=2)
    plt.plot(compare_df['date'], compare_df['sp500_cumulative_return_norm'], label='S&P 500', linewidth=2)
    plt.title("Strategy vs. S&P 500 (Normalized Cumulative Returns)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Cumulative Return (Starting at 1.0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("news_portfolio_return.png", bbox_inches="tight")
    plt.show()