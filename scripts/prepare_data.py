"""
=============================================================================
Prepare Data — Wrapper Script for Data Preparation
=============================================================================
This is a convenience wrapper that calls src/data_loader.py's prepare_dataset().
Run this before training to parse the IAM dataset and create train/val/test splits.

Usage:
    python scripts/prepare_data.py

Prerequisites:
    1. Download IAM Handwriting Dataset
    2. Place in dataset/raw/iam_words/
    3. Structure:
        dataset/raw/iam_words/
        ├── words.txt            # Annotations file
        └── words/               # Image directory
            ├── a01/
            │   ├── a01-000u/
            │   │   ├── a01-000u-00-00.png
            │   │   ├── a01-000u-00-01.png
            │   │   └── ...
            │   └── ...
            └── ...
"""

import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import prepare_dataset


if __name__ == "__main__":
    print("=" * 60)
    print("  Write2Text — Data Preparation Script")
    print("=" * 60)

    try:
        train_df, val_df, test_df, char_to_idx, idx_to_char, num_classes = prepare_dataset()

        print("\n" + "=" * 60)
        print("  Data preparation complete!")
        print("  You can now train the model:")
        print("    python train.py")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure the IAM dataset is in the correct location.")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
