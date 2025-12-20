import time
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import features as feat

# === КОНФИГУРАЦИЯ ===
MOEX_BASE = "https://iss.moex.com/iss"
CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def create_session() -> requests.Session:
    """Создание сессии с автоматическим переподключением"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_cache_path(ticker: str, board: str, start: datetime, end: datetime) -> Path:
    """Генерация пути к кэш-файлу для тикера"""
    key = f"{ticker}_{board}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
    return CACHE_DIR / f"{ticker}_{hash_key[:8]}.pkl"


def fetch_paginated(
    url: str,
    params: Dict[str, Any],
    key: str,
    limit: int = 100,
    pause: float = 0.3
) -> pd.DataFrame:
    session = create_session()
    all_rows = []
    columns = None
    start_param = 0
    
    print(f"   Fetching {key} data...")
    
    while True:
        current_params = params.copy()
        current_params.update({
            "start": start_param,
            "limit": limit,
            "iss.meta": "off",
            "iss.only": key
        })
        
        try:
            response = session.get(url, params=current_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            container = data.get(key, {})
            rows = container.get("data", [])
            
            if columns is None:
                columns = container.get("columns", [])
            
            if not rows:
                print(f"      No more data at start={start_param}")
                break
            
            all_rows.extend(rows)
            print(f"      Got {len(rows)} rows (start={start_param})")
            
            start_param += len(rows)
            if len(rows) < limit:
                break
            
            time.sleep(pause)
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Network error: {e}, retrying...")
            time.sleep(pause * 2)
            if start_param > 10000:
                print("   ❌ Too many retries, aborting")
                break
            continue
    
    if not columns:
        raise ValueError(f"❌ No columns found in response for {key}")
    
    df = pd.DataFrame(all_rows, columns=columns)
    df.columns = [c.lower() for c in df.columns]
    
    return df


def load_daily_ticker(
    ticker: str,
    board: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> pd.DataFrame:  # УДАЛЕН НЕИСПОЛЬЗУЕМЫЙ ПАРАМЕТР fill_nan
    cache_path = get_cache_path(ticker, board, start, end)
    
    # Попытка загрузки из кэша
    if use_cache and cache_path.exists():
        try:
            print(f"   💾 Loading from cache: {cache_path.name}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"   ⚠️  Cache load failed: {e}, reloading from API")
    
    # ИСПРАВЛЕН: ДОБАВЛЕН /history/ ДЛЯ ФЬЮЧЕРСОВ
    if board == "RFUD":
        url = f"{MOEX_BASE}/history/engines/futures/markets/forts/boards/{board}/securities/{ticker}.json"
    else:  # Акции и другие
        url = f"{MOEX_BASE}/history/engines/stock/markets/shares/boards/{board}/securities/{ticker}.json"
    
    params = {
        "from": start.strftime("%Y-%m-%d"),
        "till": end.strftime("%Y-%m-%d"),
    }
    
    # Загрузка данных
    df = fetch_paginated(url, params, key="history")
    
    if df.empty:
        print(f"   ⚠️  No data received for {ticker}")
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'ticker'])
    
    # === ОБРАБОТКА ДАННЫХ ===
    
    # Нормализация колонок
    if "tradedate" in df.columns:
        df["date"] = pd.to_datetime(df["tradedate"]).dt.date
        df.drop(columns=["tradedate"], inplace=True)
    elif "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"]).dt.date
        df.drop(columns=["trade_date"], inplace=True)
    else:
        # Ищем колонку с датой
        date_col = None
        for col in df.columns:
            if "date" in col.lower() and col != "date":
                date_col = col
                break
        if date_col:
            df["date"] = pd.to_datetime(df[date_col]).dt.date
            df.drop(columns=[date_col], inplace=True)
        else:
            raise ValueError(f"❌ No date column found for {ticker}")
    
    # Переименование value -> volume
    if "value" in df.columns and "volume" not in df.columns:
        df.rename(columns={"value": "volume"}, inplace=True)
    
    # Добавление тикера
    df["ticker"] = ticker
    
    # Фильтрация по датам
    df = df[(df["date"] >= start.date()) & (df["date"] <= end.date())]
    
    # Удаление дубликатов
    df = df.drop_duplicates(subset=["date", "ticker"])
    
    # Выбор нужных колонок
    required_cols = ["date", "open", "high", "low", "close", "volume", "ticker"]
    available_cols = [c for c in required_cols if c in df.columns]
    
    # Если чего-то не хватает - заполняем нулями
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"   ⚠️  Missing columns for {ticker}: {missing_cols}, filling with 0")
        for col in missing_cols:
            df[col] = 0.0
    
    df = df[required_cols].copy()
    
    # Сохранение в кэш
    if use_cache:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            print(f"   💾 Saved to cache: {cache_path.name}")
        except Exception as e:
            print(f"   ⚠️  Cache save failed: {e}")
    
    return df


def load_daily_multi(
    symbols: List[str],
    board: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> pd.DataFrame:
    parts = []
    global_min_date = end.date()
    global_max_date = start.date()
    
    print(f"\n📥 Loading data for {len(symbols)} symbols...")
    
    for ticker in symbols:
        try:
            print(f"\n   Loading {ticker}...")
            df = load_daily_ticker(ticker, board, start, end, use_cache)
            
            if df.empty:
                print(f"   ⚠️  Empty data for {ticker}, skipping")
                continue
            
            # Обновляем диапазон дат
            global_min_date = min(global_min_date, df["date"].min())
            global_max_date = max(global_max_date, df["date"].max())
            
            parts.append(df)
            print(f"   ✅ {ticker}: {df.shape[0]} rows")
            
        except Exception as e:
            print(f"   ❌ Error loading {ticker}: {e}")
            continue
    
    if not parts:
        print("❌ No data loaded for any symbol!")
        return pd.DataFrame()
    
    # Объединение всех тикеров
    print(f"\n🔀 Merging {len(parts)} tickers...")
    df_all = pd.concat(parts, ignore_index=True)
    
    # === СОЗДАНИЕ ПОЛНОГО ДАТАСЕТА ===
    
    # Создаем полный календарь торговых дней
    print("   Creating full date range...")
    full_dates = pd.date_range(
        start=global_min_date,
        end=global_max_date,
        freq='B'
    ).date
    
    # Создаем DataFrame date + ticker
    print("   Creating base grid...")
    df_pivot = df_all.pivot(index="date", columns="ticker", values="close")
    df_pivot = df_pivot.reindex(full_dates)
    
    df_melted = df_pivot.rename_axis('date').reset_index().melt(
        id_vars=['date'],
        var_name='ticker',
        value_name='close'
    )
    
    other_cols = ['open', 'high', 'low', 'volume']
    for col in other_cols:
        if col in df_all.columns:
            pivot_col = df_all.pivot(index="date", columns="ticker", values=col)
            pivot_col = pivot_col.reindex(full_dates)
            
            melted_col = pivot_col.rename_axis('date').reset_index().melt(
                id_vars=['date'],
                var_name='ticker',
                value_name=col
            )
            
            df_melted = df_melted.merge(
                melted_col[['date', 'ticker', col]],
                on=['date', 'ticker'],
                how='left'
            )
    
    # Заполняем пропуски
    print("   Filling NaN values...")
    
    # Для цен: forward-fill
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_melted.columns:
            df_melted[col] = df_melted.groupby('ticker')[col].fillna(method='ffill')
    
    # Для объема: 0
    if 'volume' in df_melted.columns:
        df_melted['volume'] = df_melted['volume'].fillna(0)
    
    # Удаляем строки где все цены NaN
    df_melted = df_melted.dropna(subset=price_cols, how='all')
    
    # Добавляем индикаторы
    print("📊 Adding technical indicators...")
    df_melted = add_indicators_grouped(df_melted)
    
    # Сортировка
    df_melted = df_melted.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    print(f"✅ Final dataset: {df_melted.shape[0]} rows, {len(df_melted['ticker'].unique())} tickers")
    print(f"   Date range: {df_melted['date'].min()} → {df_melted['date'].max()}")
    print(f"   Columns: {list(df_melted.columns)}")
    
    return df_melted


def add_indicators_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление технических индикаторов для каждого тикера отдельно
    """
    if df.empty:
        return df
    
    results = []
    
    for ticker, group in df.groupby("ticker"):
        print(f"      Processing indicators for {ticker}...")
        tmp = group.sort_values("date").reset_index(drop=True)
        
        # Проверка минимального количества данных
        if len(tmp) < 26:
            print(f"         ⚠️  Not enough data for {ticker} ({len(tmp)} rows), skipping indicators")
            results.append(tmp)
            continue
        
        # Добавляем индикаторы
        try:
            tmp = feat.add_bollinger_bands(tmp, window=20, num_std=2.0)
            tmp = feat.calculate_macd(tmp, fast=12, slow=26, signal=9)
            tmp = feat.calculate_rsi(tmp, window=14)
            
            # Переименование
            tmp.rename(columns={
                "sma": "sma20",
                "bollinger_upper": "boll_upper20",
                "bollinger_lower": "boll_lower20",
                "rsi": "rsi14"
            }, inplace=True)
            
            results.append(tmp)
            
        except Exception as e:
            print(f"         ❌ Error calculating indicators for {ticker}: {e}")
            results.append(group)
    
    # Объединение результатов
    df_enriched = pd.concat(results, ignore_index=True)
    df_enriched = df_enriched.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Заполняем NaN в индикаторах (для первых 20 дней)
    indicator_cols = ['sma20', 'boll_upper20', 'boll_lower20', 'macd', 'macd_signal', 'rsi14']
    for col in indicator_cols:
        if col in df_enriched.columns:
            df_enriched[col] = df_enriched.groupby('ticker')[col].fillna(method='ffill').fillna(0)
    
    return df_enriched


def demo_load_data():
    """Демонстрация работы загрузчика данных"""
    print("=" * 80)
    print("ДЕМОНСТРАЦИЯ ЗАГРУЗКИ ДАННЫХ С MOEX")
    print("=" * 80)
    
    # Для акций
    print("\n📈 Пример 1: Акции MOEX (TQBR)")
    symbols = ["SBER", "GAZP", "LKOH"]
    board = "TQBR"
    start = datetime(2019, 1, 1)
    end = datetime(2022, 12, 31)
    
    print(f"Тикеры: {symbols}")
    print(f"Период: {start.date()} - {end.date()}")
    print(f"Борд: {board}")
    
    df_stocks = load_daily_multi(symbols, board, start, end, use_cache=True)
    
    if not df_stocks.empty:
        print("\n✅ Результат:")
        print(f"Shape: {df_stocks.shape}")
        print(f"Колонки: {list(df_stocks.columns)}")
        print(f"Диапазон дат: {df_stocks['date'].min()} - {df_stocks['date'].max()}")
        print(f"Количество тикеров: {df_stocks['ticker'].nunique()}")
        print(f"\nПервые 5 строк:")
        print(df_stocks.head().to_string())
        print(f"\nПоследние 5 строк:")
        print(df_stocks.tail().to_string())
        
        # Сохранение
        output_file = Path(__file__).parent / "data" / "demo_stocks.csv"
        output_file.parent.mkdir(exist_ok=True)
        df_stocks.to_csv(output_file, index=False)
        print(f"\n💾 Данные сохранены в: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ Демонстрация завершена")


if __name__ == "__main__":
    demo_load_data()