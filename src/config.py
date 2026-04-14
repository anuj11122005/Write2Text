"""
=============================================================================
Configuration for Write2Text - CRNN Handwriting Recognition
=============================================================================
Centralized configuration for all hyperparameters, paths, and settings.
Supports both local development and Google Colab environments.
"""

import os

# ── Auto-detect environment ──────────────────────────────────────────────────
# Detect if running in Google Colab
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if IN_COLAB:
    # Google Colab paths (mount Google Drive first)
    DRIVE_ROOT = "/content/drive/MyDrive/DL_Project"
    DATASET_RAW = os.path.join(DRIVE_ROOT, "dataset", "raw", "iam_words")
    DATASET_PROCESSED = os.path.join(DRIVE_ROOT, "dataset", "processed")
    CHECKPOINTS_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
    OUTPUTS_DIR = os.path.join(DRIVE_ROOT, "outputs")
else:
    # Local paths
    DATASET_RAW = os.path.join(PROJECT_ROOT, "dataset", "raw", "iam_words")
    DATASET_PROCESSED = os.path.join(PROJECT_ROOT, "dataset", "processed")
    CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ── Image Dimensions ────────────────────────────────────────────────────────
# Width x Height — wider images preserve more horizontal detail for text
IMG_WIDTH = 128     # Horizontal (sequence dimension after CNN)
IMG_HEIGHT = 32     # Vertical

# ── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 64             # Reduce to 32 if GPU runs out of memory
NUM_EPOCHS = 50             # Total training epochs
LEARNING_RATE = 0.001       # Initial learning rate (Adam)
WEIGHT_DECAY = 1e-4         # L2 regularization

# ── Learning Rate Schedule ──────────────────────────────────────────────────
LR_SCHEDULER = "plateau"    # Options: "plateau", "cosine", "step"
LR_PATIENCE = 5             # Epochs to wait before reducing LR (plateau)
LR_FACTOR = 0.5             # Multiply LR by this factor when plateau

# ── Early Stopping ──────────────────────────────────────────────────────────
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15  # Stop if no val_loss improvement for N epochs

# ── Gradient Clipping ───────────────────────────────────────────────────────
GRAD_CLIP = 5.0

# ── CTC Settings ────────────────────────────────────────────────────────────
BLANK_IDX = 0               # CTC blank token is always index 0

# ── Model Architecture ──────────────────────────────────────────────────────
CNN_CHANNELS = [64, 128, 256, 512]   # CNN channel progression
RNN_HIDDEN = 256                      # LSTM hidden size (per direction)
RNN_LAYERS = 2                        # Number of LSTM layers
DROPOUT = 0.3                         # Dropout rate in RNN

# ── Data Pipeline ───────────────────────────────────────────────────────────
MAX_LABEL_LENGTH = 32       # Maximum word length to keep
MIN_LABEL_LENGTH = 1        # Minimum word length to keep
TRAIN_SPLIT = 0.8           # Fraction for training
VAL_SPLIT = 0.1             # Fraction for validation
TEST_SPLIT = 0.1            # Fraction for testing
NUM_WORKERS = 0             # DataLoader workers (0 for Colab compatibility)

# ── Data Augmentation ──────────────────────────────────────────────────────
AUGMENT_TRAIN = True        # Enable augmentation during training
AUG_ROTATION = 2            # Max rotation degrees
AUG_NOISE_STD = 0.02        # Gaussian noise standard deviation

# ── Beam Search ─────────────────────────────────────────────────────────────
BEAM_WIDTH = 10             # Beam width for beam search decoding

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_INTERVAL = 5            # Show predictions every N epochs
SAVE_BEST_ONLY = True       # Only save model when val_loss improves

# ── Create directories ──────────────────────────────────────────────────────
for _dir in [CHECKPOINTS_DIR, DATASET_PROCESSED, OUTPUTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Print config summary ────────────────────────────────────────────────────
def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("  Write2Text Configuration")
    print("=" * 60)
    print(f"  Environment:    {'Google Colab' if IN_COLAB else 'Local'}")
    print(f"  Image size:     {IMG_WIDTH} x {IMG_HEIGHT}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  Learning rate:  {LEARNING_RATE}")
    print(f"  CNN channels:   {CNN_CHANNELS}")
    print(f"  RNN hidden:     {RNN_HIDDEN} (x2 bidirectional)")
    print(f"  RNN layers:     {RNN_LAYERS}")
    print(f"  Dropout:        {DROPOUT}")
    print(f"  Grad clip:      {GRAD_CLIP}")
    print(f"  Early stopping: {EARLY_STOPPING} (patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Augmentation:   {AUGMENT_TRAIN}")
    print(f"  Dataset raw:    {DATASET_RAW}")
    print(f"  Checkpoints:    {CHECKPOINTS_DIR}")
    print("=" * 60)
