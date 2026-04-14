"""
=============================================================================
Data Loader — IAM Handwriting Words Dataset Parser
=============================================================================
Parses the IAM words.txt file, validates image paths, builds vocabulary,
and creates train/val/test splits saved as CSV files.

IAM words.txt format (each non-comment line):
    image_id ok/err gray_level num_components x y w h transcription
    Example: a01-000u-00-00 ok 154 408 768 27 51 AT

Image path structure:
    words/{folder1}/{folder2}/{image_id}.png
    e.g. words/a01/a01-000u/a01-000u-00-00.png
"""

import os
import sys
import json
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def parse_words_txt(words_file_path):
    """
    Parse the IAM words.txt annotation file.

    Each line format:
        image_id segmentation_status graylevel #components x y w h transcription

    We extract: image_id, segmentation_status, and transcription (last field).
    Lines starting with '#' are comments and are skipped.

    Args:
        words_file_path: Path to words.txt file

    Returns:
        List of dicts with keys: image_id, status, label
    """
    if not os.path.exists(words_file_path):
        raise FileNotFoundError(
            f"words.txt not found at: {words_file_path}\n"
            f"Please download the IAM dataset and place it in: {config.DATASET_RAW}"
        )

    entries = []
    skipped_lines = 0

    with open(words_file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Need at least 9 parts: id + status + graylevel + components + x,y,w,h + word
            if len(parts) < 9:
                skipped_lines += 1
                continue

            image_id = parts[0]
            status = parts[1]           # "ok" or "err"
            transcription = parts[-1]   # Last field is the word

            # The transcription field uses "|" for spaces in multi-word entries
            # Replace "|" with space for rare cases
            transcription = transcription.replace("|", " ")

            entries.append({
                "image_id": image_id,
                "status": status,
                "label": transcription
            })

    print(f"[DataLoader] Parsed {len(entries)} entries from words.txt "
          f"(skipped {skipped_lines} malformed lines)")

    return entries


def resolve_image_path(image_id, base_dir):
    """
    Construct the full image path from an image_id.

    IAM path structure: words/{folder1}/{folder2}/{image_id}.png
    Example: a01-000u-00-00 → words/a01/a01-000u/a01-000u-00-00.png

    Args:
        image_id: e.g. "a01-000u-00-00"
        base_dir: Root of IAM dataset (contains words/ directory)

    Returns:
        Full path string, or None if file doesn't exist
    """
    parts = image_id.split("-")
    if len(parts) < 3:
        return None

    folder1 = parts[0]                           # e.g. "a01"
    folder2 = f"{parts[0]}-{parts[1]}"           # e.g. "a01-000u"
    filename = f"{image_id}.png"

    full_path = os.path.join(base_dir, "words", folder1, folder2, filename)

    if os.path.exists(full_path):
        return full_path

    return None


def validate_image(image_path):
    """
    Check if an image file can be loaded and is not corrupted.

    Args:
        image_path: Path to image file

    Returns:
        True if image is valid, False otherwise
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        h, w = img.shape
        # Reject images that are too small or too large
        if h < 5 or w < 5 or h > 2000 or w > 2000:
            return False
        return True
    except Exception:
        return False


def filter_entries(entries, base_dir, max_label_len=None, min_label_len=None):
    """
    Filter dataset entries:
    - Skip entries with status "err"
    - Skip entries where image file is missing
    - Skip entries with labels too long or too short
    - Skip entries with empty labels

    Args:
        entries: List of dicts from parse_words_txt()
        base_dir: Root of IAM dataset
        max_label_len: Maximum label length (default from config)
        min_label_len: Minimum label length (default from config)

    Returns:
        Filtered list of dicts with added "image_path" key
    """
    if max_label_len is None:
        max_label_len = config.MAX_LABEL_LENGTH
    if min_label_len is None:
        min_label_len = config.MIN_LABEL_LENGTH

    filtered = []
    stats = Counter()

    for entry in entries:
        label = entry["label"]

        # Skip segmentation errors
        if entry["status"] == "err":
            stats["segmentation_error"] += 1
            continue

        # Skip empty labels
        if not label or len(label.strip()) == 0:
            stats["empty_label"] += 1
            continue

        # Skip labels that are too long or too short
        if len(label) > max_label_len:
            stats["label_too_long"] += 1
            continue
        if len(label) < min_label_len:
            stats["label_too_short"] += 1
            continue

        # Resolve and validate image path
        image_path = resolve_image_path(entry["image_id"], base_dir)
        if image_path is None:
            stats["missing_image"] += 1
            continue

        # Validate image can be loaded
        if not validate_image(image_path):
            stats["corrupted_image"] += 1
            continue

        filtered.append({
            "image_path": image_path,
            "label": label
        })

    # Print filtering statistics
    print(f"[DataLoader] Filtering results:")
    print(f"  [OK]   Valid samples:        {len(filtered)}")
    for reason, count in stats.most_common():
        print(f"  [SKIP] {reason:22s}: {count}")

    return filtered


def build_vocabulary(labels):
    """
    Build character vocabulary from all labels.

    Index 0 is reserved for CTC blank token.
    Characters are sorted alphabetically for reproducibility.

    Args:
        labels: List of label strings

    Returns:
        char_to_idx: dict mapping character → index (1-indexed)
        idx_to_char: dict mapping index → character
        num_classes: total classes including CTC blank
    """
    # Collect all unique characters
    all_chars = set()
    for label in labels:
        all_chars.update(label)

    # Sort for reproducibility
    chars = sorted(list(all_chars))

    # Build mappings (index 0 = CTC blank)
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    num_classes = len(chars) + 1  # +1 for blank

    print(f"[DataLoader] Vocabulary: {len(chars)} unique characters")
    print(f"  Characters: {''.join(chars)}")
    print(f"  Num classes (with blank): {num_classes}")

    return char_to_idx, idx_to_char, num_classes


def save_char_mapping(char_to_idx, idx_to_char, num_classes, output_dir=None):
    """Save character mapping to JSON file."""
    if output_dir is None:
        output_dir = config.DATASET_PROCESSED

    mapping = {
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "vocab_size": len(char_to_idx),
        "num_classes": num_classes
    }

    path = os.path.join(output_dir, "char_mapping.json")
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[DataLoader] Saved character mapping to: {path}")
    return path


def prepare_dataset(dataset_raw=None, output_dir=None):
    """
    Complete data preparation pipeline:
    1. Parse words.txt
    2. Filter invalid entries
    3. Build vocabulary
    4. Split into train/val/test
    5. Save CSVs and character mapping

    Args:
        dataset_raw: Path to raw IAM dataset (default from config)
        output_dir: Output directory for processed files (default from config)

    Returns:
        train_df, val_df, test_df, char_to_idx, idx_to_char, num_classes
    """
    if dataset_raw is None:
        dataset_raw = config.DATASET_RAW
    if output_dir is None:
        output_dir = config.DATASET_PROCESSED

    print("=" * 60)
    print("  Write2Text — Data Preparation")
    print("=" * 60)

    # Step 1: Parse words.txt
    words_file = os.path.join(dataset_raw, "words.txt")
    entries = parse_words_txt(words_file)

    # Step 2: Filter entries
    filtered = filter_entries(entries, dataset_raw)

    if len(filtered) == 0:
        raise ValueError(
            "No valid samples found! Check:\n"
            f"  1. words.txt exists at: {words_file}\n"
            f"  2. Image files exist in: {os.path.join(dataset_raw, 'words')}\n"
            f"  3. Directory structure: words/a01/a01-000u/a01-000u-00-00.png"
        )

    # Step 3: Create DataFrame
    df = pd.DataFrame(filtered)

    # Step 4: Build vocabulary from ALL data before splitting
    all_labels = df["label"].tolist()
    char_to_idx, idx_to_char, num_classes = build_vocabulary(all_labels)

    # Step 5: Split into train / val / test
    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - config.TRAIN_SPLIT), random_state=42, shuffle=True
    )
    relative_val = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - relative_val), random_state=42
    )

    print(f"\n[DataLoader] Dataset splits:")
    print(f"  Train:      {len(train_df):,} samples")
    print(f"  Validation: {len(val_df):,} samples")
    print(f"  Test:       {len(test_df):,} samples")
    print(f"  Total:      {len(df):,} samples")

    # Step 6: Save to disk
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    save_char_mapping(char_to_idx, idx_to_char, num_classes, output_dir)

    # Save dataset statistics
    stats = {
        "total_samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "vocab_size": int(len(char_to_idx)),
        "num_classes": int(num_classes),
        "max_label_length": int(df["label"].str.len().max()),
        "avg_label_length": float(round(df["label"].str.len().mean(), 2)),
        "min_label_length": int(df["label"].str.len().min()),
    }
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[DataLoader] All files saved to: {output_dir}")
    print(f"  Label length: min={stats['min_label_length']}, "
          f"avg={stats['avg_label_length']}, max={stats['max_label_length']}")

    return train_df, val_df, test_df, char_to_idx, idx_to_char, num_classes


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prepare_dataset()
