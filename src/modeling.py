from __future__ import annotations

import tensorflow as tf


def build_mlp(input_dim: int, num_classes: int = 7) -> tf.keras.Model:
   
    inputs = tf.keras.Input(shape=(input_dim,), name="features")

    # Dense layers learn non-linear combinations of your tabular features
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)  # stabilizes training
    x = tf.keras.layers.Dropout(0.30)(x)         # reduces overfitting

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)

   #  no softmax here because loss uses from_logits=True
    outputs = tf.keras.layers.Dense(num_classes, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp_baseline")

    # Sparse labels (0..6) -> use SparseCategoricalCrossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        ],
    )
    return model
