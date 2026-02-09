---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
Notebook:
- `notebooks/01_eda_preprocess.ipynb`

Steps included:
- dataset inspection
- missing value check
- duplicate row check
- class distribution analysis
- one-hot validation for soil and wilderness columns

---

### 2. Preprocessing Pipeline

Implemented in `src/preprocess.py`.

Key preprocessing steps:

- stratified train/validation/test split
- scaling numerical features using `StandardScaler`
- combining scaled numerical + one-hot categorical features
- saving processed datasets as `.npy` files

Outputs saved in `data/processed/`:

- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `scaler.joblib`

---

### 3. Baseline Deep Learning Model (MLP)

Implemented in:

- `src/modeling.py`
- `src/train.py`

Architecture:

- Dense layers with Batch Normalization and Dropout
- output layer with 7 logits

Baseline Results:

- **Test Accuracy:** 0.7971
- **Weighted F1-score:** 0.8049
- Confusion Matrix saved to:
  - `reports/figures/confusion_matrix_test.png`

---

### 4. Improved Model (MLP V2)

Implemented in:

- `src/train_v2.py`

Improvements:

- larger network architecture (512 → 256 → 128)
- L2 regularization
- AdamW optimizer (weight decay)
- ReduceLROnPlateau + EarlyStopping
- class weights to reduce imbalance impact

V2 Results:

- **Test Accuracy:** 0.8674
- **Weighted F1-score:** 0.8713
- **Macro F1-score:** 0.8142
- Confusion Matrix saved to:
  - `reports/figures/confusion_matrix_test_v2.png`

---

## Results Summary

| Model           | Test Accuracy | Weighted F1 | Macro F1   |
| --------------- | ------------- | ----------- | ---------- |
| Baseline MLP    | 0.7971        | 0.8049      | 0.7358     |
| Improved MLP V2 | **0.8674**    | **0.8713**  | **0.8142** |

---

### Confusion Matrix (Baseline)

![Baseline Confusion Matrix](reports/figures/confusion_matrix_test.png)

### Confusion Matrix (V2)

![V2 Confusion Matrix](reports/figures/confusion_matrix_test_v2.png)

---

### Key Insights

- Model V2 achieved **86.7% test accuracy**, significantly improving over the baseline model (~79.7%).

- Validation accuracy increased steadily and plateaued after ~40 epochs, indicating stable convergence.

- Most misclassifications occurred between classes **0 and 1**, suggesting these cover types share similar environmental features.

- Minority classes (such as class 3 and class 4) achieved high recall but lower precision, meaning the model detects them well but sometimes confuses them with other classes.

- Using **class weights** improved performance on underrepresented classes.

---

## Detailed Model Report

A detailed analysis of model performance, comparison between baseline and improved models, confusion matrix interpretation, and recommendations for future improvements can be found here:

📄 **[Model Performance Report](REPORT.md)**

This report includes:

- comparison of baseline vs improved models
- explanation of why the improved model performs better
- analysis of misclassifications
- performance metrics (accuracy, precision, recall, F1)
- suggestions for further improvements

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/natalialungu22/forest-covertype-DeepLearning.git
cd forest-covertype-DeepLearning
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment (Mac/Linux)

```bash
source .venv/bin/activate
```

You should now see something like:

```bash
(.venv) ➜ forest-covertype-DeepLearning
```

## Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

## How to Run the Project

1. Add the dataset

Place the dataset CSV file inside:

```bash
data/raw/covertype.csv
```

2. Run preprocessing

This will create the processed .npy datasets inside data/processed/:

```bash
python3 -m src.preprocess
```

3. Train the baseline model

```bash
python3 -m src.train
```

This will save the model in:

```bash
models/best_model.keras
models/last_model.keras
```

4. Train the improved V2 model

```bash
python3 -m src.train_v2
```

This will save the model in:

```bash
models/best_model_v2.keras
models/last_model_v2.keras
```

5. Evaluate the model

The evaluation script prints a classification report and saves a confusion matrix.

To evaluate the baseline model, set inside src/evaluate.py:

```bash
model_path = ROOT / "models" / "best_model.keras"
```

To evaluate the V2 model, set:

```bash
model_path = ROOT / "models" / "best_model_v2.keras"
```

Then run:

```bash
python3 -m src.evaluate
```

Confusion matrix images will be saved in:

```bash
reports/figures/
```

---

## Technologies Used

- Python

- NumPy

- Pandas

- Scikit-learn

- TensorFlow / Keras

- Matplotlib
