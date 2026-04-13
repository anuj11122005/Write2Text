"""Configuration for CRNN Handwriting Recognition."""

import os

# Image dimensions (must match model architecture)
# Larger images = better accuracy but slower training
IMG_WIDTH = 128
IMG_HEIGHT = 32

# Training - HIGHER ACCURACY SETTINGS
BATCH_SIZE = 64  # Reduce to 32 if out of memory
NUM_EPOCHS = 100  # More epochs = better convergence
LEARNING_RATE = 0.001  # Initial learning rate

# Learning rate schedule - better for convergence
LR_SCHEDULER = 'plateau'  # Options: 'plateau', 'step', 'onecycle'
LR_PATIENCE = 10  # Epochs before reducing LR
LR_FACTOR = 0.5  # Multiply LR by this when plateau

# Early stopping - prevents overfitting
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20  # Stop if no improvement for N epochs

# Gradient clipping
GRAD_CLIP = 5.0

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_RAW = os.path.join(PROJECT_ROOT, "dataset", "raw", "iam_words")
DATASET_PROCESSED = os.path.join(PROJECT_ROOT, "dataset", "processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# CTC
BLANK_IDX = 0  # CTC blank token index

# Create directories if needed
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(DATASET_PROCESSED, exist_ok=True)
