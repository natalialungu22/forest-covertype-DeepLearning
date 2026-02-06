from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessArtifacts:
    num_cols: list[str]
    soil_cols: list[str]
    wild_cols: list[str]
    scaler: StandardScaler


def get_feature_groups(df: pd.DataFrame, target_col: str = "class") -> tuple[list[str], list[str], list[str]]:
    # One-hot columns (binary)
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]

    # Numeric columns (everything else except target)
    num_cols = [c for c in df.columns if c not in soil_cols + wild_cols + [target_col]]
    return num_cols, soil_cols, wild_cols


def split_data(
    df: pd.DataFrame,
    target_col: str = "class",
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # Separate features/target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Convert 1..7 -> 0..6 for Keras sparse labels
    y = y - 1

    # First split: train vs temp
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: val vs test (split temp in half if equal sizes)
    val_ratio_of_temp = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_of_temp),
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_and_stack(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()

    # Fit scaler on TRAIN only (avoid leakage)
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_val_num = scaler.transform(X_val[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    # One-hot columns are already 0/1 (no scaling needed)
    X_train_cat = X_train[cat_cols].to_numpy()
    X_val_cat = X_val[cat_cols].to_numpy()
    X_test_cat = X_test[cat_cols].to_numpy()

    # Combine numeric + one-hot
    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_val_final = np.hstack([X_val_num, X_val_cat])
    X_test_final = np.hstack([X_test_num, X_test_cat])

    return X_train_final, X_val_final, X_test_final, scaler


def save_processed(
    out_dir: Path,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "X_test.npy", X_test)

    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "y_test.npy", y_test)

    joblib.dump(scaler, out_dir / "scaler.joblib")
