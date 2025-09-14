import time
from datetime import datetime
from typing import List

import pandas as pd
import requests

from . import features as feat

MOEX_BASE = "https://iss.moex.com/iss"


def _fetch_paginated(
    url: str, params: dict, key: str, limit: int = 500, pause: float = 0.2
) -> pd.DataFrame:
    params = params.copy()
    params.setdefault("start", 0)
    params.setdefault("limit", limit)
    rows = []
    columns = None
    with requests.Session() as s:
        while True:
            r = s.get(url, params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
            container = j.get(key, {})
            batch = container.get("data", [])
            cols = container.get("columns", [])
            if cols and columns is None:
                columns = cols
            if not batch:
                break
            rows.extend(batch)
            print(f"GET {url} start={params.get('start')} -> {len(batch)} rows")
            if len(batch) < limit:
                break
            params["start"] = params.get("start", 0) + len(batch)
            time.sleep(pause)
    if not columns:
        raise ValueError("no columns parsed from response")
    df = pd.DataFrame(rows, columns=columns)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_daily(symbol: str, board: str, start: datetime, end: datetime) -> pd.DataFrame:
    url = f"{MOEX_BASE}/history/engines/stock/markets/shares/boards/{board}/securities/{symbol}.json"
    params = {
        "from": start.strftime("%Y-%m-%d"),
        "till": end.strftime("%Y-%m-%d"),
        "start": 0,
        "limit": 500,
    }
    df = _fetch_paginated(url, params, key="history")
    # normalize columns
    if "tradedate" in df.columns:
        df["date"] = pd.to_datetime(df["tradedate"]).dt.date
    elif "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"]).dt.date
    else:
        # try to find any date-like column
        for c in df.columns:
            if "date" in c:
                df["date"] = pd.to_datetime(df[c]).dt.date
                break

    # rename value->volume if needed
    if "value" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"value": "volume"})

    df["ticker"] = symbol

    keep = [
        c
        for c in ["date", "open", "high", "low", "close", "volume", "ticker"]
        if c in df.columns
    ]
    return df[keep].copy()


def load_minutes(
    symbol: str, board: str, start: datetime, end: datetime, interval: int = 1
) -> pd.DataFrame:
    url = f"{MOEX_BASE}/engines/stock/markets/shares/securities/{symbol}/candles.json"
    params = {
        "from": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "till": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "interval": interval,
        "start": 0,
        "limit": 500,
    }
    df = _fetch_paginated(url, params, key="candles")
    if "begin" in df.columns:
        df["datetime"] = pd.to_datetime(df["begin"])
    elif "begininterval" in df.columns:
        df["datetime"] = pd.to_datetime(df["begininterval"])
    else:
        for c in df.columns:
            if "time" in c or "date" in c or "begin" in c:
                df["datetime"] = pd.to_datetime(df[c])
                break
    if "value" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"value": "volume"})
    df["ticker"] = symbol
    keep = [
        c
        for c in ["datetime", "open", "high", "low", "close", "volume", "ticker"]
        if c in df.columns
    ]
    return df[keep].copy()


def _add_indicators_grouped(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _sym, g in df.groupby("ticker"):
        tmp = g.sort_values("date").reset_index(drop=True)
        tmp = feat.add_bollinger_bands(
            tmp.rename(columns={"close": "close"}), window=20, num_std=2
        )
        tmp = feat.calculate_macd(tmp, fast=12, slow=26, signal=9)
        tmp = feat.calculate_rsi(tmp, window=14)
        # rename to explicit
        tmp = tmp.rename(
            columns={
                "sma": "sma20",
                "bollinger_upper": "boll_upper20",
                "bollinger_lower": "boll_lower20",
                "macd": "macd",
                "macd_signal": "macd_signal",
                "rsi": "rsi14",
            }
        )
        parts.append(tmp)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def load_daily_multi(
    symbols: List[str], board: str, start: datetime, end: datetime
) -> pd.DataFrame:
    parts = []
    for s in symbols:
        print(f"Loading daily {s}...")
        try:
            df = load_daily(s, board, start, end)
            if df.empty:
                print(f"warning: empty daily for {s}")
            else:
                parts.append(df)
        except Exception as e:
            print(f"error loading {s}: {e}")
    if not parts:
        return pd.DataFrame()
    df_all = pd.concat(parts, ignore_index=True)
    # ensure date is datetime.date
    if not pd.api.types.is_datetime64_any_dtype(df_all["date"]):
        df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    # add indicators per ticker
    df_all = _add_indicators_grouped(df_all)
    return df_all


def merge_minute_with_daily(min_df: pd.DataFrame, eod_df: pd.DataFrame) -> pd.DataFrame:
    min_df = min_df.copy()
    min_df["date"] = pd.to_datetime(min_df["datetime"]).dt.date
    merged = pd.merge(
        min_df, eod_df, on=["date", "ticker"], how="left", suffixes=("_min", "_eod")
    )
    return merged
