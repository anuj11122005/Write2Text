"""Prepare IAM Words dataset for training."""

import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def prepare_dataset():
    """Parse words.txt and create train/val/test splits."""
    labels = []
    words_file = os.path.join(config.DATASET_RAW, "words.txt")

    with open(words_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split()
            image_id = parts[0]
            word = parts[-1]

            # Skip invalid words
            if word == "err" or not word.isalpha():
                continue

            folder1 = image_id.split("-")[0]
            folder2 = "-".join(image_id.split("-")[:2])
            image_path = os.path.join(config.DATASET_RAW, "words", folder1, folder2, f"{image_id}.png")

            if os.path.exists(image_path):
                labels.append([image_path, word])

    df = pd.DataFrame(labels, columns=["image_path", "label"])
    print(f"Total samples: {len(df)}")

    # Split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Build character vocabulary from ALL splits (ensures consistency)
    all_labels = pd.concat([train_df, val_df, test_df])['label'].tolist()
    chars = sorted(list(set("".join(all_labels))))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 reserved for CTC blank
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}

    print(f"Vocabulary size: {len(chars)}")
    print(f"Characters: {''.join(chars)}")

    # Save
    os.makedirs(config.DATASET_PROCESSED, exist_ok=True)
    train_df.to_csv(os.path.join(config.DATASET_PROCESSED, "train.csv"), index=False)
    val_df.to_csv(os.path.join(config.DATASET_PROCESSED, "val.csv"), index=False)
    test_df.to_csv(os.path.join(config.DATASET_PROCESSED, "test.csv"), index=False)

    # Save character mapping
    mapping = {
        'char_to_idx': char_to_idx,
        'idx_to_char': {str(k): v for k, v in idx_to_char.items()},  # JSON keys must be strings
        'vocab_size': len(chars),
        'num_classes': len(chars) + 1  # +1 for CTC blank
    }
    with open(os.path.join(config.DATASET_PROCESSED, "char_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Saved to {config.DATASET_PROCESSED}")


if __name__ == "__main__":
    prepare_dataset()
