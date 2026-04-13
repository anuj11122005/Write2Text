"""Check if project is set up correctly before training."""

import os
import sys
import pandas as pd
import torch

from src import config
from src.model import CRNN
from src.dataset import IAMDataset
from src.utils import build_char_mapping, decode_predictions


def check_files():
    """Check if required files exist."""
    print("Checking files...")

    required_files = [
        os.path.join(config.DATASET_PROCESSED, "train_clean.csv"),
        os.path.join(config.DATASET_PROCESSED, "val.csv"),
    ]

    all_good = True
    for f in required_files:
        if os.path.exists(f):
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            all_good = False

    return all_good


def check_model():
    """Check model dimensions."""
    print("\nChecking model dimensions...")

    device = torch.device("cpu")
    dummy_classes = 80
    model = CRNN(dummy_classes).to(device)

    # Test forward pass
    x = torch.randn(2, 1, config.IMG_HEIGHT, config.IMG_WIDTH).to(device)
    y = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    expected_time_steps = config.IMG_WIDTH // 4  # 2 pools of stride 2
    expected_shape = (2, expected_time_steps, dummy_classes)

    if y.shape == expected_shape:
        print(f"  [OK] Output shape correct")
        return True
    else:
        print(f"  [FAIL] Expected {expected_shape}, got {y.shape}")
        return False


def check_dataset():
    """Check dataset loading."""
    print("\nChecking dataset...")

    train_path = os.path.join(config.DATASET_PROCESSED, "train_clean.csv")
    if not os.path.exists(train_path):
        print(f"  [FAIL] Train file not found")
        return False

    train_df = pd.read_csv(train_path)
    print(f"  Train samples: {len(train_df)}")

    if len(train_df) == 0:
        print(f"  [FAIL] No training samples")
        return False

    # Check character mapping
    char_to_idx, idx_to_char, num_classes = build_char_mapping(train_df['label'].tolist())
    print(f"  Vocabulary size: {len(char_to_idx)}")
    print(f"  Num classes (with blank): {num_classes}")

    # Check dataset loading
    dataset = IAMDataset(train_df.iloc[:5], char_to_idx, config.IMG_WIDTH, config.IMG_HEIGHT)
    img, label = dataset[0]

    print(f"  Sample image shape: {img.shape}")
    print(f"  Sample label length: {len(label)}")

    # Check first image file exists (try loading through dataset)
    try:
        _ = dataset[0]
        print(f"  [OK] Image files accessible")
    except Exception as e:
        print(f"  [FAIL] Cannot load image: {e}")
        return False

    return True


def check_training_step():
    """Check if a single training step works."""
    print("\nChecking training step...")

    device = torch.device("cpu")

    # Load minimal data
    train_path = os.path.join(config.DATASET_PROCESSED, "train_clean.csv")
    train_df = pd.read_csv(train_path).iloc[:10]

    char_to_idx, idx_to_char, num_classes = build_char_mapping(train_df['label'].tolist())
    dataset = IAMDataset(train_df, char_to_idx, config.IMG_WIDTH, config.IMG_HEIGHT)

    # Create single batch
    images, labels = [], []
    for i in range(4):
        img, lbl = dataset[i]
        images.append(img)
        labels.append(lbl)

    imgs = torch.stack(images, 0).to(device)
    labels = [l.to(device) for l in labels]

    # Forward pass
    model = CRNN(num_classes).to(device)
    preds = model(imgs)

    # CTC loss
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    log_probs = torch.log_softmax(preds, dim=2).permute(1, 0, 2)
    targets = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    input_lengths = torch.full((4,), log_probs.size(0), dtype=torch.long)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    # Backward pass
    loss.backward()

    print(f"  Loss computed: {loss.item():.4f}")
    print(f"  [OK] Training step works")

    # Check prediction
    pred_texts = decode_predictions(preds, idx_to_char)
    true_texts = [''.join([idx_to_char[i.item()] for i in l]) for l in labels]

    print(f"  Sample prediction: '{pred_texts[0]}' | True: '{true_texts[0]}'")
    print(f"  (Random at start - should improve after training)")

    return True


def main():
    print("=" * 60)
    print("CRNN Setup Diagnostic")
    print("=" * 60)

    checks = [
        ("Files", check_files),
        ("Model", check_model),
        ("Dataset", check_dataset),
        ("Training Step", check_training_step),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] Error in {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n[OK] All checks passed! Ready to train.")
        print("\nRun: python train.py")
    else:
        print("\n[FAIL] Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
