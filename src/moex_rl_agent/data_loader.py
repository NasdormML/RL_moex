from datetime import datetime

import numpy as np
import pandas as pd
import requests


def add_bollinger_bands(df, window=20, num_std=2):
    """Добавление полос Боллинджера."""
    df["SMA"] = df["close"].rolling(window=window).mean()
    df["std"] = df["close"].rolling(window=window).std()
    df["bollinger_upper"] = df["SMA"] + num_std * df["std"]
    df["bollinger_lower"] = df["SMA"] - num_std * df["std"]
    return df


def add_additional_indicators(df):
    """Добавление дополнительных индикаторов."""
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["ATR"] = (
        df["high"] - df["low"]
    )  # Простой пример, можно использовать более сложные вычисления
    df["OBV"] = (df["volume"] * np.sign(df["close"].diff())).cumsum()
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Добавление MACD и сигнальной линии."""
    exp1 = df["close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    return df


def load_daily_multi(
    symbols: list[str], board: str, start: datetime, end: datetime
) -> pd.DataFrame:
    dfs = []
    for symbol in symbols:
        try:
            url = (
                "https://iss.moex.com/iss/history/engines/stock/markets/shares"
                f"/boards/{board}/securities/{symbol}.json"
            )
            params = {
                "from": start.strftime("%Y-%m-%d"),
                "till": end.strftime("%Y-%m-%d"),
            }
            r = requests.get(url, params=params)
            r.raise_for_status()

            data = r.json()["history"]
            df = pd.DataFrame(data["data"], columns=data["columns"])
            df["date"] = pd.to_datetime(df["TRADEDATE"])
            df.rename(
                columns={
                    "OPEN": "open",
                    "HIGH": "high",
                    "LOW": "low",
                    "CLOSE": "close",
                    "VALUE": "volume",
                },
                inplace=True,
            )
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df["ticker"] = symbol

            # Индикаторы
            df["SMA20"] = df["close"].rolling(20, min_periods=1).mean()
            df["RSI14"] = (
                df["close"]
                .diff()
                .apply(lambda x: max(x, 0) if x > 0 else 0)
                .rolling(14)
                .mean()
            )
            df["MOM5"] = df["close"].diff(5)
            df = add_bollinger_bands(df)
            df = add_additional_indicators(df)
            df = calculate_macd(df)

            dfs.append(df.dropna().reset_index(drop=True))

        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке данных для {symbol}: {e}")
            continue

    multi = pd.concat(dfs, ignore_index=True)
    multi.sort_values(["date", "ticker"], inplace=True)
    return multi.reset_index(drop=True)
