import pandas as pd


def check_dataframe_for_env(df: pd.DataFrame):
    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    return True


def save_df_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def load_df_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
