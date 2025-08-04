import asyncio
import logging
from typing import Any, Dict, List

import aiohttp
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def fetch_paginated(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict[str, Any],
    key: str,
    page_size: int = 500,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> pd.DataFrame:
    start = params.get("start", 0)
    all_rows: List[List[Any]] = []
    columns: List[str] = []

    while True:
        attempt = 0
        while attempt <= max_retries:
            try:
                async with session.get(url, params={**params, "start": start}) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
                break
            except Exception as exc:
                wait = backoff_factor * (2**attempt)
                logger.warning(
                    "Request failed (attempt %d/%d). Retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    exc,
                )
                await asyncio.sleep(wait)
                attempt += 1
        else:
            logger.error("Max retries exceeded for %s with params %s", url, params)
            break

        page = payload.get(key, {})
        batch = page.get("data", [])
        if not columns:
            columns = page.get("columns", [])

        if not batch:
            logger.info("No more data returned; ending pagination.")
            break

        logger.info("Fetched %d rows (start=%d)", len(batch), start)
        all_rows.extend(batch)

        if len(batch) < page_size:
            break
        start += len(batch)

    if not all_rows or not columns:
        logger.error("Empty dataset for key '%s'", key)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=columns)
    return df


def prepare_eod(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["tradedate"]).dt.date
    return df.rename(
        columns={
            "open": "open_eod",
            "high": "high_eod",
            "low": "low_eod",
            "close": "close_eod",
            "value": "volume_eod",
        }
    )


def prepare_minute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    df["datetime"] = pd.to_datetime(df["begin"]).dt.floor("min")
    df["date"] = df["datetime"].dt.date
    return df.rename(
        columns={
            "open": "open_min",
            "high": "high_min",
            "low": "low_min",
            "close": "close_min",
            "value": "volume_min",
        }
    )


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    df["sma20"] = df["close_min"].rolling(20).mean()
    df["std20"] = df["close_min"].rolling(20).std()
    df["boll_upper"] = df["sma20"] + 2 * df["std20"]
    df["boll_lower"] = df["sma20"] - 2 * df["std20"]

    e12 = df["close_min"].ewm(span=12).mean()
    e26 = df["close_min"].ewm(span=26).mean()
    df["macd"] = e12 - e26
    df["signal"] = df["macd"].ewm(span=9).mean()
    return df


async def load_all_data(
    from_date: str,
    till_date: str,
    start_time: str = None,
    end_time: str = None,
) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        eod_url = (
            "https://iss.moex.com/iss/history/engines/stock/markets/shares"
            "/boards/TQBR/securities/SBER.json"
        )
        min_url = (
            "https://iss.moex.com/iss/engines/stock/markets/shares"
            "/securities/SBER/candles.json"
        )

        params_eod = {"from": from_date, "till": till_date, "start": 0}
        params_min = (
            {
                "from": f"{from_date}T{start_time}",
                "till": f"{till_date}T{end_time}",
                "interval": 1,
                "start": 0,
            }
            if start_time and end_time
            else {"from": from_date, "till": till_date, "interval": 1, "start": 0}
        )

        eod_task = fetch_paginated(session, eod_url, params_eod, key="history")
        min_task = fetch_paginated(session, min_url, params_min, key="candles")

        eod_df, min_df = await asyncio.gather(eod_task, min_task)

    if eod_df.empty or min_df.empty:
        raise ValueError("Failed to load EOD or minute data")

    eod_clean = prepare_eod(eod_df)
    min_clean = prepare_minute(min_df)

    merged = min_clean.merge(
        eod_clean[
            ["date", "open_eod", "high_eod", "low_eod", "close_eod", "volume_eod"]
        ],
        on="date",
        how="left",
    )

    result = add_technical_indicators(merged)
    return result


if __name__ == "__main__":
    # Example usage
    from_date = "2025-07-08"
    till_date = "2025-07-10"
    start_t = "10:00:00"
    end_t = "18:45:00"

    df = asyncio.run(load_all_data(from_date, till_date, start_t, end_t))
    print(df.head())
    print(df.describe())
