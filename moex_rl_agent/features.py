import pandas as pd


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands со стандартной популяции"""
    s = df.copy()
    
    s["sma"] = s["close"].rolling(window=window, min_periods=window).mean()
    s["std"] = s["close"].rolling(window=window, min_periods=window).std(ddof=0)
    
    s["bollinger_upper"] = s["sma"] + num_std * s["std"]
    s["bollinger_lower"] = s["sma"] - num_std * s["std"]
    
    s["bollinger_upper"] = s["bollinger_upper"].fillna(method='ffill').fillna(s["close"])
    s["bollinger_lower"] = s["bollinger_lower"].fillna(method='ffill').fillna(s["close"])
    
    return s


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """MACD с обработкой NaN"""
    s = df.copy()
    
    exp1 = s["close"].ewm(span=fast, adjust=False).mean()
    exp2 = s["close"].ewm(span=slow, adjust=False).mean()
    
    s["macd"] = exp1 - exp2
    s["macd_signal"] = s["macd"].ewm(span=signal, adjust=False).mean()
    
    s["macd"] = s["macd"].fillna(0)
    s["macd_signal"] = s["macd_signal"].fillna(0)
    
    return s


def calculate_rsi(
    df: pd.DataFrame,
    window: int = 14
) -> pd.DataFrame:
    """RSI с Wilder smoothing (EMA)"""
    s = df.copy()
    
    delta = s["close"].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    
    roll_up = up.ewm(alpha=1/window, min_periods=window).mean()
    roll_down = down.ewm(alpha=1/window, min_periods=window).mean()
    
    rs = roll_up / (roll_down + 1e-9)
    s["rsi"] = 100 - 100 / (1 + rs)
    s["rsi"] = s["rsi"].fillna(50)
    
    return s
