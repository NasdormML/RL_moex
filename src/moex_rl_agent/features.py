def macd(df, fast=12, slow=26, signal=9):
    exp1 = df["close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


macd.feature_dim = 2
