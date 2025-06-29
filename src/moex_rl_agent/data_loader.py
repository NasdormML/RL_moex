from datetime import datetime

import pandas as pd
import requests


def load_daily_multi(
    symbols: list[str], board: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """
    Загрузить дневные свечи для списка тикеров и добавить индикаторы:
    SMA20, RSI14, MOM5.
    Возвращает DataFrame с колонками
    ['date','ticker','open','high','low','close','volume','SMA20','RSI14','MOM5'].
    """
    dfs = []
    for symbol in symbols:
        url = (
            f"https://iss.moex.com/iss/history/engines/stock/markets/shares/"
            f"boards/{board}/securities/{symbol}.json"
        )
        params = {"from": start.strftime("%Y-%m-%d"), "till": end.strftime("%Y-%m-%d")}
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

        # индикаторы
        df["SMA20"] = df["close"].rolling(20, min_periods=1).mean()
        delta = df["close"].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.ewm(span=14, adjust=False).mean()
        roll_down = down.ewm(span=14, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-9)
        df["RSI14"] = 100 - 100 / (1 + rs)
        df["MOM5"] = df["close"].diff(5)

        dfs.append(df.dropna().reset_index(drop=True))

    multi = pd.concat(dfs, ignore_index=True)
    multi.sort_values(["date", "ticker"], inplace=True)
    return multi.reset_index(drop=True)
