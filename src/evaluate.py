from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]

    processed_dir = ROOT / "data" / "processed"
    model_path = ROOT / "models" / "best_model.keras"

    reports_dir = ROOT / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Predict logits -> class ids (0..6)
    logits = model.predict(X_test, batch_size=2048, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    # Print per-class metrics
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)

    plt.figure(figsize=(8, 8))
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()

    out_path = reports_dir / "confusion_matrix_test.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("\nSaved confusion matrix to:", out_path.resolve())


if __name__ == "__main__":
    main()
