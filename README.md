# Forest Cover Type Classifier (Deep Learning)

Predict forest cover type (7 classes) from cartographic variables using TensorFlow/Keras.

## Dataset

Forest CoverType dataset (Roosevelt National Forest, Colorado).  
Features include numeric cartographic variables plus one-hot encoded wilderness areas (4) and soil types (40).  
Target column: `class` with values 1..7.

## Project goals

- Build a multi-class classifier with TensorFlow/Keras
- Apply good preprocessing (no data leakage)
- Tune hyperparameters and evaluate performance
- Keep code clean, modular, and reproducible

## Repository structure

- `notebooks/` — EDA and experimentation
- `src/` — reusable code (data loading, preprocessing, training, evaluation)
- `reports/figures/` — plots and saved figures

## How to run

1. Create venv and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
