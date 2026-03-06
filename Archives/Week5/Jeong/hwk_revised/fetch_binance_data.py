#!/usr/bin/env python3
"""
Fetch real BTC/USDT 1-min klines and trade data from Binance public API.

Training period: 14 days
Test period: 7 days (completely separate, unseen during training)

No API key required — uses public endpoints only.
"""
import requests
import time
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BINANCE_BASE = "https://api.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_klines(symbol, interval, start_ms, end_ms, limit=1000):
    """Fetch klines from Binance API (max 1000 per request)."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"  Retry {attempt+1}/5: {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"  Failed after 5 retries, skipping chunk")
            break

        if not data:
            break

        all_klines.extend(data)
        current_start = data[-1][0] + 1  # next ms after last candle
        time.sleep(0.1)  # rate limiting

        if len(data) < limit:
            break

    return all_klines


def fetch_agg_trades(symbol, start_ms, end_ms, limit=1000):
    """Fetch aggregate trades from Binance API."""
    url = f"{BINANCE_BASE}/api/v3/aggTrades"
    all_trades = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": min(current_start + 3600000, end_ms),  # 1hr chunks
            "limit": limit,
        }
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"  Retry {attempt+1}/5: {e}")
                time.sleep(2 ** attempt)
        else:
            break

        if not data:
            current_start += 3600000
            continue

        all_trades.extend(data)
        current_start = data[-1]["T"] + 1
        time.sleep(0.15)

        if len(all_trades) % 10000 < limit:
            print(f"  Fetched {len(all_trades)} trades so far...")

    return all_trades


def klines_to_df(klines):
    """Convert raw klines to DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_vol", "taker_buy_quote_vol"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["n_trades"] = df["n_trades"].astype(int)

    # Mid-price = (high + low) / 2 as intra-bar estimate
    df["mid_price"] = (df["high"] + df["low"]) / 2.0
    # Approximate half-spread from high-low range
    df["half_spread"] = (df["high"] - df["low"]) / 2.0
    # Trade intensity (trades per minute)
    df["trade_intensity"] = df["n_trades"]
    return df


def main():
    print("=" * 70)
    print("Fetching real BTC/USDT data from Binance")
    print("=" * 70)

    # Define time periods
    # Training: 14 days ending 30 days ago
    # Test: 7 days ending 16 days ago (completely separate)
    now = datetime.utcnow()

    # Training period: ~45 to ~31 days ago (14 days)
    train_end = now - timedelta(days=31)
    train_start = train_end - timedelta(days=14)

    # Test period: ~30 to ~23 days ago (7 days) — unseen
    test_start = train_end + timedelta(days=1)
    test_end = test_start + timedelta(days=7)

    train_start_ms = int(train_start.timestamp() * 1000)
    train_end_ms = int(train_end.timestamp() * 1000)
    test_start_ms = int(test_start.timestamp() * 1000)
    test_end_ms = int(test_end.timestamp() * 1000)

    print(f"\nTraining period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} (14 days)")
    print(f"Test period:     {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} (7 days)")
    print(f"(Test data is COMPLETELY UNSEEN during training)\n")

    # ---- Fetch Training Klines ----
    print("Fetching TRAINING klines (1-min BTC/USDT)...")
    train_klines = fetch_klines(SYMBOL, INTERVAL, train_start_ms, train_end_ms)
    train_df = klines_to_df(train_klines)
    print(f"  Got {len(train_df)} 1-min candles for training")
    print(f"  Price range: ${train_df['mid_price'].min():.0f} - ${train_df['mid_price'].max():.0f}")

    # ---- Fetch Test Klines ----
    print("\nFetching TEST klines (1-min BTC/USDT)...")
    test_klines = fetch_klines(SYMBOL, INTERVAL, test_start_ms, test_end_ms)
    test_df = klines_to_df(test_klines)
    print(f"  Got {len(test_df)} 1-min candles for testing")
    print(f"  Price range: ${test_df['mid_price'].min():.0f} - ${test_df['mid_price'].max():.0f}")

    # ---- Fetch sample trades for fill-rate estimation (6 hours from training period) ----
    print("\nFetching sample trades for fill-rate estimation (6 hours)...")
    trade_start_ms = train_start_ms
    trade_end_ms = trade_start_ms + 6 * 3600 * 1000  # 6 hours
    trades = fetch_agg_trades(SYMBOL, trade_start_ms, trade_end_ms)

    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["price"] = trade_df["p"].astype(float)
        trade_df["qty"] = trade_df["q"].astype(float)
        trade_df["time"] = pd.to_datetime(trade_df["T"], unit="ms")
        trade_df["is_buyer_maker"] = trade_df["m"]
        print(f"  Got {len(trade_df)} aggregate trades")
        trade_df.to_parquet(os.path.join(SAVE_DIR, "train_trades.parquet"), index=False)
    else:
        print("  Warning: Could not fetch trades, will estimate fill-rate from klines")
        trade_df = None

    # ---- Save data ----
    train_df.to_parquet(os.path.join(SAVE_DIR, "train_klines.parquet"), index=False)
    test_df.to_parquet(os.path.join(SAVE_DIR, "test_klines.parquet"), index=False)

    print(f"\nSaved:")
    print(f"  train_klines.parquet: {len(train_df)} rows")
    print(f"  test_klines.parquet:  {len(test_df)} rows")
    if trade_df is not None:
        print(f"  train_trades.parquet: {len(trade_df)} rows")

    # Quick summary stats
    print(f"\n=== Training Data Summary ===")
    print(f"  Period: {train_df['open_time'].iloc[0]} to {train_df['open_time'].iloc[-1]}")
    print(f"  Mid-price: mean=${train_df['mid_price'].mean():.0f}, std=${train_df['mid_price'].std():.0f}")
    print(f"  Half-spread: mean=${train_df['half_spread'].mean():.2f}")
    print(f"  Trades/min: mean={train_df['trade_intensity'].mean():.1f}")

    print(f"\n=== Test Data Summary ===")
    print(f"  Period: {test_df['open_time'].iloc[0]} to {test_df['open_time'].iloc[-1]}")
    print(f"  Mid-price: mean=${test_df['mid_price'].mean():.0f}, std=${test_df['mid_price'].std():.0f}")
    print(f"  Half-spread: mean=${test_df['half_spread'].mean():.2f}")
    print(f"  Trades/min: mean={test_df['trade_intensity'].mean():.1f}")


if __name__ == "__main__":
    main()
