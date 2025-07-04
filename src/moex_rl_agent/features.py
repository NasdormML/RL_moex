def add_bollinger_bands(df, window=20, num_std=2):
    df["SMA"] = df["close"].rolling(window=window).mean()
    df["std"] = df["close"].rolling(window=window).std()
    df["bollinger_upper"] = df["SMA"] + num_std * df["std"]
    df["bollinger_lower"] = df["SMA"] - num_std * df["std"]
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df["close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    return df


def calculate_rsi(df, window=14):
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(window=window, min_periods=1).mean()
    roll_down = down.rolling(window=window, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI14"] = 100 - 100 / (1 + rs)
    return df
