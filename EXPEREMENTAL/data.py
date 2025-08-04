import asyncio

import aiohttp
import pandas as pd


async def fetch_data(session, url, params, key):
    data = []
    start = 0
    while True:
        params["start"] = start
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            json_data = await response.json()
            batch = json_data.get(key, {}).get("data", [])
            if not batch:
                break

            columns = json_data[key]["columns"]
            data.extend(batch)
            if len(batch) < 500:
                break
            start += len(batch)
    return pd.DataFrame(data, columns=columns)


async def fetch_ticker_data(session, ticker, date_from, date_to):
    base_url = "https://iss.moex.com/iss"

    # Дневные данные
    eod_params = {"from": date_from, "till": date_to, "start": 0}
    eod_url = f"{base_url}/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    eod_df = await fetch_data(session, eod_url, eod_params, "history")

    # Минутные данные
    min_params = {
        "from": f"{date_from}T10:00:00",
        "till": f"{date_to}T23:59:59",
        "interval": 1,
        "start": 0,
    }
    min_url = (
        f"{base_url}/engines/stock/markets/shares/securities/{ticker}/candles.json"
    )
    min_df = await fetch_data(session, min_url, min_params, "candles")

    # Обработка данных
    eod_df = process_eod(eod_df, ticker)
    min_df = process_min(min_df, ticker)

    # Объединение данных
    return merge_data(eod_df, min_df, ticker)


def process_eod(df, ticker):
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["tradedate"]).dt.date
    df["ticker"] = ticker
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "open_eod",
            "high": "high_eod",
            "low": "low_eod",
            "close": "close_eod",
            "volume": "volume_eod",
        }
    )


def process_min(df, ticker):
    df.columns = [c.lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["begin"])
    df["date"] = df["datetime"].dt.date
    df["ticker"] = ticker
    df = df[["datetime", "ticker", "open", "high", "low", "close", "value"]].rename(
        columns={
            "open": "open_min",
            "high": "high_min",
            "low": "low_min",
            "close": "close_min",
            "value": "volume_min",
        }
    )
    return df.set_index("datetime")


def merge_data(eod_df, min_df, ticker):
    # Объединение с forward fill для дневных данных
    merged = min_df.merge(eod_df, on=["date", "ticker"], how="left")
    merged = merged.sort_index()

    # Добавление индикаторов
    return add_technical_indicators(merged)


def add_technical_indicators(df):
    # Минутные индикаторы
    df["sma20_min"] = df.groupby("ticker")["close_min"].transform(
        lambda x: x.rolling(20).mean()
    )
    df["rsi_min"] = calculate_rsi(df.groupby("ticker")["close_min"], 14)

    # Дневные индикаторы
    df["sma5_eod"] = df.groupby("ticker")["close_eod"].transform(
        lambda x: x.rolling(5).mean()
    )
    df["daily_volatility"] = df.groupby("ticker")["close_eod"].transform(
        lambda x: x.pct_change().rolling(5).std()
    )

    # Комбинированные индикаторы
    df["premium_to_daily"] = df["close_min"] / df["close_eod"].shift(1) - 1
    return df.dropna()


def calculate_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


async def main(tickers, date_from, date_to):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ticker_data(session, t, date_from, date_to) for t in tickers]
        results = await asyncio.gather(*tasks)
        return pd.concat(results).sort_index()
