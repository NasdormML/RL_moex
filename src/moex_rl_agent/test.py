import asyncio

import aiohttp
import pandas as pd


async def fetch_paginated_data(session, url, params, key):
    data = []
    while True:
        async with session.get(url, params=params) as response:
            try:
                response.raise_for_status()
                response_data = await response.json()
            except Exception as e:
                print(f"Ошибка при запросе {url} с параметрами {params}: {e}")
                break

            # Логируем, сколько записей было получено на текущей итерации
            data_batch = response_data.get(key, {}).get("data", [])
            columns = response_data.get(key, {}).get("columns", [])

            if not columns or not data_batch:
                print(f"Ошибка: пустой ответ от API (url: {url}, params: {params})")
                break

            # Логируем сколько записей получено
            print(f"Получено {len(data_batch)} записей")

            data.extend(data_batch)

            if len(data_batch) < 500:
                break

            params["start"] = len(data)

    return pd.DataFrame(data, columns=columns)


# Основная асинхронная функция для загрузки данных
async def main():
    async with aiohttp.ClientSession() as session:
        # Дневные данные с пагинацией
        url_eod = (
            "https://iss.moex.com/iss/history/engines/stock/markets/shares"
            "/boards/TQBR/securities/SBER.json"
        )
        params_eod = {"from": "2025-07-08", "till": "2025-07-10", "start": 0}
        eod_df = await fetch_paginated_data(session, url_eod, params_eod, "history")

        if eod_df.empty:
            print("Ошибка: нет данных для дневных котировок.")
        else:
            print("Дневные данные успешно загружены.")

        eod_df.columns = [c.lower() for c in eod_df.columns]

        # Минутные данные с пагинацией
        url_min = (
            "https://iss.moex.com/iss/engines/stock/markets/shares"
            "/securities/SBER/candles.json"
        )
        params_min = {
            "from": "2025-07-08T10:00:00",
            "till": "2025-07-10T18:45:00",
            "interval": 1,
            "start": 0,
        }
        min_df = await fetch_paginated_data(session, url_min, params_min, "candles")

        if min_df.empty:
            print("Ошибка: нет данных для минутных котировок.")
        else:
            print("Минутные данные успешно загружены.")

        # Приводим столбцы к нижнему регистру
        min_df.columns = [c.lower() for c in min_df.columns]

        # Для дневных данных:
        eod_df["date"] = pd.to_datetime(eod_df["tradedate"]).dt.date
        eod_df.rename(
            columns={
                "open": "open_eod",
                "high": "high_eod",
                "low": "low_eod",
                "close": "close_eod",
                "value": "volume_eod",
            },
            inplace=True,
        )

        # Для минутных данных:
        min_df["datetime"] = pd.to_datetime(min_df["begin"])
        min_df["date"] = min_df["datetime"].dt.date
        min_df.rename(
            columns={
                "open": "open_min",
                "high": "high_min",
                "low": "low_min",
                "close": "close_min",
                "value": "volume_min",
            },
            inplace=True,
        )

        min_df["datetime"] = min_df["datetime"].dt.floor("min")  # Группируем по минутам

        # Объединение дневных и минутных данных по дате
        merged = pd.merge(
            min_df,
            eod_df[
                ["date", "open_eod", "close_eod", "high_eod", "low_eod", "volume_eod"]
            ],
            on="date",
            how="left",
        )

        # Добавление технических индикаторов для минутных данных
        def add_indicators(df):
            df = df.sort_values("datetime").reset_index(drop=True)
            df["sma20_min"] = df["close_min"].rolling(20).mean()
            df["std20_min"] = df["close_min"].rolling(20).std()
            df["boll_upper_min"] = (
                df["sma20_min"] + 2 * df["std20_min"]
            )  # Верхняя полоса Боллинджера.
            df["boll_lower_min"] = (
                df["sma20_min"] - 2 * df["std20_min"]
            )  # Нижняя полоса Боллинджера
            df["ema12_min"] = (
                df["close_min"].ewm(span=12).mean()
            )  # Экспоненциальная скользящая средняя (EMA) с периодом 12
            df["ema26_min"] = df["close_min"].ewm(span=26).mean()  # EMA с периодом 26
            df["macd_min"] = df["ema12_min"] - df["ema26_min"]  # Индикатор MACD
            df["signal_min"] = (
                df["macd_min"].ewm(span=9).mean()
            )  # Линия сигнала для MACD
            return df

        data = add_indicators(merged)

        print(
            data[
                [
                    "datetime",
                    "date",
                    "open_min",
                    "close_min",
                    "boll_upper_min",
                    "boll_lower_min",
                    "macd_min",
                    "signal_min",
                ]
            ].head()
        )
        print(data.describe())


asyncio.run(main())
