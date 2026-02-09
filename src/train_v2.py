from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

from .modeling import build_mlp_v2
from .config import (
    RANDOM_STATE,
    BATCH_SIZE,
    V2_EPOCHS,
    EARLY_STOPPING_PATIENCE_V2,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
)

# Load processed NumPy arrays created by the preprocessing pipeline.
def load_processed(processed_dir: Path):
    X_train = np.load(processed_dir / "X_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")

    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    return X_train, X_val, X_test, y_train, y_val, y_test

# Compute class weights to reduce bias from class imbalance.
def make_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}

#Save training history JSON and learning-curve plots (loss + accuracy).
def save_history_and_plots(ROOT: Path, history: tf.keras.callbacks.History) -> None:
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    history_dir = reports_dir / "history"
    figures_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    # Save history as JSON
    history_path = history_dir / "history_v2.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print("Saved training history to:", history_path.resolve(strict=False))

    # Loss curve
    plt.figure()
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("Training vs Validation Loss (V2)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(figures_dir / "training_history_v2_loss.png", dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(history.history.get("acc", []))
    plt.plot(history.history.get("val_acc", []))
    plt.title("Training vs Validation Accuracy (V2)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(figures_dir / "training_history_v2_acc.png", dpi=200)
    plt.close()

    print("Saved plots to:", figures_dir.resolve(strict=False))

# Create loss/accuracy plots from Keras CSVLogger output.
def plot_history_from_csv(csv_path: Path, figures_dir: Path) -> None:
    """Create loss/accuracy plots from Keras CSVLogger output."""
    if not csv_path.exists():
        print("No CSV history found at:", csv_path)
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV history is empty (no completed epochs).")
        return

    # Loss
    plt.figure()
    plt.plot(df["loss"])
    if "val_loss" in df.columns:
        plt.plot(df["val_loss"])
    plt.title("Training vs Validation Loss (V2)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(figures_dir / "training_history_v2_loss.png", dpi=200)
    plt.close()

    # Accuracy
    plt.figure()
    if "acc" in df.columns:
        plt.plot(df["acc"])
    if "val_acc" in df.columns:
        plt.plot(df["val_acc"])
    plt.title("Training vs Validation Accuracy (V2)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(figures_dir / "training_history_v2_acc.png", dpi=200)
    plt.close()

    print("Saved plots to:", figures_dir.resolve(strict=False))


def main():
    ROOT = Path(__file__).resolve().parents[1]
    processed_dir = ROOT / "data" / "processed"

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # Load processed arrays
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(processed_dir)

    # Build model
    model = build_mlp_v2(input_dim=X_train.shape[1], num_classes=7)
    model.summary()

    # Handle class imbalance
    class_weights = make_class_weights(y_train)

    # Paths for saving history and plots
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    history_dir = reports_dir / "history"
    figures_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = history_dir / "history_v2.csv"

    # Training callbacks
    callbacks = [
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE_V2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "best_model_v2.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = None

    try:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=V2_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
    finally:
        plot_history_from_csv(csv_log_path, figures_dir)


    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nV2 Test loss: {test_loss:.4f} | V2 Test acc: {test_acc:.4f}")

    # Save last model snapshot
    model.save(models_dir / "last_model_v2.keras")
    print("Saved v2 models to:", models_dir.resolve(strict=False))


if __name__ == "__main__":
    main()
