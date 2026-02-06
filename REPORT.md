# Model Performance Report — Forest Cover Type Classification

## 1. Project Goal

The objective of this project was to build a **multi-class classification model** that predicts the forest cover type (7 classes) using only cartographic and environmental variables such as elevation, slope, distances, hillshade, wilderness area, and soil type.

This dataset represents 30m × 30m land cells in Roosevelt National Forest (Colorado, USA).  
The task is challenging due to **class imbalance** and similarities between certain forest types.

---

## 2. Dataset Summary

- Total samples: **581,012**
- Total features: **54**
- Target: `class` (7 cover types)

The dataset includes:

- Continuous numerical features (elevation, distances, hillshade, etc.)
- One-hot encoded features:
  - 4 wilderness areas
  - 40 soil types

### Class Imbalance

The dataset is highly imbalanced. Some classes appear much more frequently than others.

Example counts:

- Class 2: 283,301
- Class 1: 211,840
- Class 4: 2,747 (very rare)

This imbalance affects model performance, especially for minority classes.

---

## 3. Preprocessing Summary

To ensure a clean and reproducible ML pipeline, the dataset was processed using the following steps:

### Train/Validation/Test Split

- 80% training
- 10% validation
- 10% testing

Splitting was performed using **stratified sampling** to preserve class distribution across splits.

### Scaling

- Numerical features were scaled using `StandardScaler`
- One-hot encoded features (soil type and wilderness area) were kept unchanged (0/1 values)

### Data Leakage Prevention

The scaler was fitted only on the training set to prevent leakage into validation and test data.

---

## 4. Models Developed

Two deep learning models were built and compared.

---

# Model 1 — Baseline MLP

## Architecture

The baseline model was a Multi-Layer Perceptron (MLP) consisting of:

- Dense layers (256 → 128 → 64)
- Batch Normalization
- Dropout regularization
- Adam optimizer

Class weights were applied to reduce the impact of class imbalance.

## Results (Baseline)

- **Test Accuracy:** 0.7971
- **Weighted F1-score:** 0.8049
- **Macro F1-score:** 0.7358

This model performed reasonably well, but struggled with minority classes and showed confusion between similar forest types.

---

# Model 2 — Improved MLP V2

## Improvements Made

The second model introduced several hyperparameter and architecture improvements:

### 1. Larger Network Capacity

The model was expanded to:

- Dense layers (512 → 256 → 128)

This increased the model’s ability to learn more complex patterns.

### 2. L2 Regularization

L2 regularization was added to prevent weights from becoming too large, improving generalization.

### 3. AdamW Optimizer

AdamW was used instead of Adam.  
AdamW introduces proper weight decay, helping reduce overfitting and improving performance.

### 4. Training Enhancements

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint to save the best model

Class weights were still used to address imbalance.

---

## Results (V2)

- **Test Accuracy:** 0.8674
- **Weighted F1-score:** 0.8713
- **Macro F1-score:** 0.8142

This model achieved a significant improvement over the baseline.

---

# 5. Model Comparison

| Model           | Test Accuracy | Weighted F1 | Macro F1   |
| --------------- | ------------- | ----------- | ---------- |
| Baseline MLP    | 0.7971        | 0.8049      | 0.7358     |
| Improved MLP V2 | **0.8674**    | **0.8713**  | **0.8142** |

The V2 model improves overall accuracy by approximately **7%**, and also performs better on minority classes as shown by the increase in Macro F1-score.

---

# 6. Confusion Matrix Analysis

Confusion matrix evaluation shows that most predictions are correct (strong diagonal).

### Main Misclassification Pattern

The largest confusion occurs between:

- **Class 0 (Spruce/Fir)**
- **Class 1 (Lodgepole Pine)**

This is expected because these cover types share similar environmental conditions and appear in overlapping regions.

### Minority Class Challenges

Some minority classes (such as Class 4) show weaker precision, meaning the model sometimes predicts that class incorrectly.

However, recall for rare classes improved substantially after applying class weights and using the stronger model architecture.

---

# 7. Why Model V2 Performed Better

The improved model achieved better performance due to:

- **Higher capacity** (more neurons and deeper layers)
- **Better regularization** (L2 + dropout)
- **Better optimization strategy** (AdamW)
- **Dynamic learning rate adjustment** (ReduceLROnPlateau)

These improvements helped the model generalize better and reduced overfitting, resulting in stronger test performance.

---

# 8. Performance Metrics

The evaluation used the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **Weighted and Macro Averages**

Macro metrics were important because the dataset is imbalanced and accuracy alone can hide poor performance on minority classes.

---

# 9. Recommendations for Further Improvements

To further improve classification performance, especially for minority classes, the following techniques could be tested:

### 1. Focal Loss

This loss function penalizes hard-to-classify examples more strongly and can improve minority class prediction.

### 2. Oversampling / SMOTE

Balancing the dataset through oversampling could help rare classes learn more distinct patterns.

### 3. Feature Engineering

Additional engineered features could be created, such as:

- ratios of distances
- log transforms of distance-based variables
- interaction features (e.g., elevation × slope)

### 4. Model Architectures

Try alternative models such as:

- gradient boosting models (XGBoost / LightGBM)
- tabular transformer architectures
- embedding layers for categorical one-hot features

### 5. Hyperparameter Search

Using GridSearch or RandomSearch on:

- dropout rates
- learning rate
- number of layers
- batch size
- weight decay

---

# 10. Conclusion

This project successfully developed a deep learning pipeline for forest cover type classification.

The final improved model (MLP V2) achieved strong predictive performance:

- **Test Accuracy:** 0.8674
- **Weighted F1-score:** 0.8713

The project demonstrates strong understanding of:

- deep learning for tabular classification
- preprocessing and leakage prevention
- model regularization and optimizer selection
- evaluation using classification reports and confusion matrices
- handling imbalanced datasets

This makes the project suitable as a portfolio-ready machine learning and deep learning project.
