"""Module for preprocessing raw pollution data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_encode(csv_path: str) -> pd.DataFrame:
    """Load CSV, one-hot encode wnd_dir, and place pollution as the first column."""
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df = pd.get_dummies(df, columns=["wnd_dir"], prefix="wnd_dir", dtype=float)

    # Keep pollution as the first column for easy target extraction
    cols = ["pollution"] + [c for c in df.columns if c != "pollution"]
    return df[cols]


def split_data(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split the dataframe into train, validation, and test sets."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def fit_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[MinMaxScaler, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a MinMaxScaler on training data and transform all three splits."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)
    return scaler, train_scaled, val_scaled, test_scaled


def make_sequences(
    data: np.ndarray, seq_len: int, target_col: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences (X) and next-step targets (y)."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def inverse_scale_pollution(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """Inverse-transform scaled pollution predictions back to original PM2.5 units."""
    dummy = np.zeros((len(values), scaler.n_features_in_))
    dummy[:, 0] = values
    return scaler.inverse_transform(dummy)[:, 0]
