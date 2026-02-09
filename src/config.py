"""
Central configuration file for model training and preprocessing settings.

Keeping hyperparameters here makes experiments easier to manage and improves reproducibility.
"""

# Random seed for reproducibility
RANDOM_STATE = 42

# Dataset split ratios
TEST_SIZE = 0.10
VAL_SIZE = 0.10

# Training settings
BATCH_SIZE = 1024
BASELINE_EPOCHS = 30
V2_EPOCHS = 50

# Baseline model hyperparameters
BASELINE_LR = 1e-3

# V2 model hyperparameters
V2_LR = 7e-4
V2_WEIGHT_DECAY = 1e-4
V2_L2_REG = 1e-4

# Callback settings
EARLY_STOPPING_PATIENCE_BASELINE = 5
EARLY_STOPPING_PATIENCE_V2 = 6
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6
