from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .modeling import build_mlp_v2


def load_processed(processed_dir: Path):
    X_train = np.load(processed_dir / "X_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")

    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def main():
    ROOT = Path(__file__).resolve().parents[1]
    processed_dir = ROOT / "data" / "processed"

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(42)

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(processed_dir)

    model = build_mlp_v2(input_dim=X_train.shape[1], num_classes=7)
    model.summary()

    class_weights = make_class_weights(y_train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "best_model_v2.keras"),
            monitor="val_loss",
            save_best_only=True
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=1024,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nV2 Test loss: {test_loss:.4f} | V2 Test acc: {test_acc:.4f}")

    model.save(models_dir / "last_model_v2.keras")
    print("Saved v2 models to:", models_dir.resolve())


if __name__ == "__main__":
    main()
