"""
=============================================================================
Run in Google Colab — Complete Pipeline
=============================================================================
This script is designed to be run in a Google Colab notebook cell.
It handles:
    1. Google Drive mounting
    2. Dependency installation
    3. Data preparation
    4. Model training
    5. Evaluation and prediction

Copy this entire file into a Colab cell, or run it with:
    !python run_colab.py

Prerequisites:
    Upload your IAM dataset to Google Drive at:
        My Drive/DL_Project/dataset/raw/iam_words/
"""

import os
import sys
import subprocess


def setup_colab():
    """Set up Google Colab environment."""

    # ── Step 1: Mount Google Drive ───────────────────────────────────────
    print("=" * 60)
    print("  Step 1: Mounting Google Drive")
    print("=" * 60)

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("  ✓ Google Drive mounted")
    except ImportError:
        print("  [INFO] Not running in Colab, skipping drive mount")

    # ── Step 2: Install Dependencies ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2: Installing Dependencies")
    print("=" * 60)

    packages = [
        "torch", "torchvision", "torchaudio",
        "pandas", "numpy", "opencv-python",
        "scikit-learn", "matplotlib", "tqdm"
    ]

    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✓ {pkg} already installed")
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            print(f"  ✓ {pkg} installed")

    # ── Step 3: Set Working Directory ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3: Setting Up Project")
    print("=" * 60)

    # Check if project exists on Drive
    drive_project = "/content/drive/MyDrive/DL_Project"
    if os.path.exists(drive_project):
        os.chdir(drive_project)
        print(f"  ✓ Working directory: {drive_project}")
    else:
        # Try cloning or creating in /content
        local_project = "/content/DL_Project"
        if not os.path.exists(local_project):
            os.makedirs(local_project, exist_ok=True)
        os.chdir(local_project)
        print(f"  ✓ Working directory: {local_project}")

    # Add project root to Python path
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    return os.getcwd()


def check_dataset():
    """Verify the IAM dataset is available."""
    from src import config

    words_file = os.path.join(config.DATASET_RAW, "words.txt")
    words_dir = os.path.join(config.DATASET_RAW, "words")

    print("\n" + "=" * 60)
    print("  Step 4: Checking Dataset")
    print("=" * 60)

    if not os.path.exists(words_file):
        print(f"  ✗ words.txt not found at: {words_file}")
        print(f"\n  Please upload the IAM dataset to:")
        print(f"    {config.DATASET_RAW}/")
        print(f"    ├── words.txt")
        print(f"    └── words/")
        print(f"        ├── a01/")
        print(f"        │   ├── a01-000u/")
        print(f"        │   │   ├── a01-000u-00-00.png")
        print(f"        │   │   └── ...")
        print(f"        └── ...")
        return False

    if not os.path.isdir(words_dir):
        print(f"  ✗ words/ directory not found at: {words_dir}")
        return False

    # Count images
    img_count = 0
    for root, dirs, files in os.walk(words_dir):
        img_count += sum(1 for f in files if f.endswith(".png"))
        if img_count > 100:
            break  # Just verify some exist

    print(f"  ✓ words.txt found")
    print(f"  ✓ words/ directory found ({img_count}+ images)")
    return True


def run_preparation():
    """Run data preparation."""
    print("\n" + "=" * 60)
    print("  Step 5: Preparing Dataset")
    print("=" * 60)

    from src.data_loader import prepare_dataset
    return prepare_dataset()


def run_training(num_epochs=None):
    """Run model training."""
    print("\n" + "=" * 60)
    print("  Step 6: Training Model")
    print("=" * 60)

    from src import config

    if num_epochs is not None:
        config.NUM_EPOCHS = num_epochs

    # Import and run training
    import importlib
    import train as train_module
    importlib.reload(train_module)

    # Override sys.argv to avoid argparse issues in Colab
    original_argv = sys.argv
    sys.argv = ["train.py"]

    try:
        train_module.main()
    finally:
        sys.argv = original_argv


def run_evaluation():
    """Run evaluation on test set."""
    print("\n" + "=" * 60)
    print("  Step 7: Evaluating Model")
    print("=" * 60)

    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from src import config
    from src.model import CRNN
    from src.dataset import IAMDataset, collate_fn
    from src.utils import (
        load_checkpoint, evaluate_model, show_sample_predictions
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best checkpoint
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("  [ERROR] No trained model found. Run training first.")
        return

    checkpoint = load_checkpoint(ckpt_path, device)
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    num_classes = checkpoint.get("num_classes", len(char_to_idx) + 1)

    model = CRNN(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load test set
    test_path = os.path.join(config.DATASET_PROCESSED, "test.csv")
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        test_dataset = IAMDataset(test_df, char_to_idx, augment=False)
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn
        )

        # Greedy decoding evaluation
        print("\n  Greedy Decoding:")
        cer, wer, word_acc = evaluate_model(
            model, test_loader, idx_to_char, device, use_beam_search=False
        )
        print(f"    CER:          {cer:.4f}")
        print(f"    WER:          {wer:.4f}")
        print(f"    Word Accuracy: {word_acc:.2%}")

        # Beam search evaluation
        print("\n  Beam Search Decoding (width=10):")
        cer_beam, wer_beam, word_acc_beam = evaluate_model(
            model, test_loader, idx_to_char, device,
            use_beam_search=True, beam_width=10
        )
        print(f"    CER:          {cer_beam:.4f}")
        print(f"    WER:          {wer_beam:.4f}")
        print(f"    Word Accuracy: {word_acc_beam:.2%}")

        # Show sample predictions
        print("\n  Sample Predictions (Test Set):")
        show_sample_predictions(
            model, test_loader, idx_to_char, device, num_samples=20
        )


def main():
    """Complete Colab pipeline."""
    print("╔" + "═" * 58 + "╗")
    print("║  Write2Text — Handwritten Word Recognition               ║")
    print("║  CRNN + CTC Loss on IAM Dataset                          ║")
    print("║  Google Colab Runner                                      ║")
    print("╚" + "═" * 58 + "╝")

    # Setup
    project_dir = setup_colab()

    # Check dataset
    if not check_dataset():
        print("\n[STOPPED] Please upload the IAM dataset and re-run.")
        return

    # Prepare data
    run_preparation()

    # Train (adjust epochs as needed)
    run_training(num_epochs=50)

    # Evaluate
    run_evaluation()

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("  Best model saved at: checkpoints/best_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
