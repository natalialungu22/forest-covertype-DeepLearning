import pandas as pd
from .paths import DATA_RAW

def load_raw_csv(filename: str) -> pd.DataFrame:
    # Load dataset from data/raw
    path = DATA_RAW / filename
    return pd.read_csv(path)
