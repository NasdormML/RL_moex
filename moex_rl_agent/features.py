import pandas as pd


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: int = 2
) -> pd.DataFrame:
    s = df.copy()
    s["sma"] = s["close"].rolling(window=window, min_periods=1).mean()
    s["std"] = s["close"].rolling(window=window, min_periods=1).std().fillna(0.0)
    s["bollinger_upper"] = s["sma"] + num_std * s["std"]
    s["bollinger_lower"] = s["sma"] - num_std * s["std"]
    return s


def calculate_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    s = df.copy()
    exp1 = s["close"].ewm(span=fast, adjust=False).mean()
    exp2 = s["close"].ewm(span=slow, adjust=False).mean()
    s["macd"] = exp1 - exp2
    s["macd_signal"] = s["macd"].ewm(span=signal, adjust=False).mean()
    return s


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    s = df.copy()
    delta = s["close"].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.rolling(window=window, min_periods=1).mean()
    ma_down = down.rolling(window=window, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    s["rsi"] = 100 - 100 / (1 + rs)
    return s
