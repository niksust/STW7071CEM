import pandas as pd

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Empty dataset: {path}")
    return df
