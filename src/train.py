from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .modeling import build_mlp

# Load processed NumPy arrays created by preprocess.py.
def load_processed(processed_dir: Path):
    X_train = np.load(processed_dir / "X_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")

    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    return X_train, X_val, X_test, y_train, y_val, y_test

# Compute class weights to handle class imbalance.
#This increases the loss contribution of rare classes (e.g., class 3 = original cover type 4).
def make_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def main():
    # Paths
    ROOT = Path(__file__).resolve().parents[1]
    processed_dir = ROOT / "data" / "processed"

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility (same randomness each run)
    tf.keras.utils.set_random_seed(42)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(processed_dir)

    # Build model
    model = build_mlp(input_dim=X_train.shape[1], num_classes=7)
    model.summary()

    # Handle imbalance
    class_weights = make_class_weights(y_train)

    # Callbacks:
    # - EarlyStopping: stop when val loss stops improving
    # - ReduceLROnPlateau: lower lr if training plateaus
    # - ModelCheckpoint: save best model on val loss
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=1024,  # large dataset -> larger batch is ok
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set (final unbiased estimate)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    # Save the final model (best on val loss) for later inference.
    model.save(models_dir / "last_model.keras")
    print("Saved models to:", models_dir.resolve())


if __name__ == "__main__":
    main()
