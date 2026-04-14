"""
=============================================================================
Train.py — Main Training Script for Write2Text CRNN
=============================================================================
Complete training pipeline:
    1. Load and validate data
    2. Build model
    3. Train with CTC loss, Adam optimizer, LR scheduling
    4. Early stopping based on validation loss
    5. Periodic evaluation with CER/WER metrics
    6. Save best model checkpoint
    7. Final evaluation with sample predictions

Usage:
    python train.py                    # Train with default config
    python train.py --epochs 100       # Override epochs
    python train.py --batch_size 32    # Override batch size
    python train.py --resume           # Resume from checkpoint
"""

import os
import sys
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src import config
from src.dataset import IAMDataset, collate_fn
from src.model import build_model
from src.utils import (
    load_char_mapping, save_checkpoint, load_checkpoint,
    show_sample_predictions
)
from src.train import (
    train_one_epoch, validate_one_epoch, compute_epoch_metrics, EarlyStopping
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Write2Text CRNN model for handwriting recognition"
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of epochs (default: {config.NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=None,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {config.LEARNING_RATE})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for resuming")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Override config with CLI args
    num_epochs = args.epochs or config.NUM_EPOCHS
    batch_size = args.batch_size or config.BATCH_SIZE
    learning_rate = args.lr or config.LEARNING_RATE

    # ── Print Configuration ──────────────────────────────────────────────
    config.print_config()

    # ── Device Setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")

    # ── Load Processed Data ──────────────────────────────────────────────
    train_path = os.path.join(config.DATASET_PROCESSED, "train.csv")
    val_path = os.path.join(config.DATASET_PROCESSED, "val.csv")

    if not os.path.exists(train_path):
        print(f"\n[ERROR] Processed data not found: {train_path}")
        print("Run data preparation first:")
        print("  python src/data_loader.py")
        print("  OR python scripts/prepare_data.py")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"\n[Train] Dataset loaded:")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Val samples:   {len(val_df):,}")

    # ── Character Mapping ────────────────────────────────────────────────
    char_to_idx, idx_to_char, num_classes = load_char_mapping()
    print(f"  Vocabulary:    {len(char_to_idx)} characters")
    print(f"  Num classes:   {num_classes} (including CTC blank)")

    # ── Create Datasets ──────────────────────────────────────────────────
    train_dataset = IAMDataset(
        train_df, char_to_idx,
        augment=config.AUGMENT_TRAIN     # Enable augmentation for training
    )
    val_dataset = IAMDataset(
        val_df, char_to_idx,
        augment=False                     # No augmentation for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True                    # Drop incomplete last batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # ── Build Model ──────────────────────────────────────────────────────
    model = build_model(num_classes, device=device)

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────
    criterion = nn.CTCLoss(blank=config.BLANK_IDX, zero_infinity=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE
    )

    # ── Early Stopping ───────────────────────────────────────────────────
    early_stopping = None
    if config.EARLY_STOPPING:
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    # ── Resume from Checkpoint ───────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt_path = args.checkpoint or os.path.join(
            config.CHECKPOINTS_DIR, "best_model.pth"
        )
        if os.path.exists(ckpt_path):
            checkpoint = load_checkpoint(ckpt_path, device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            print(f"\n[Train] Resumed from epoch {start_epoch} "
                  f"(val_loss: {best_val_loss:.4f})")
        else:
            print(f"\n[Train] No checkpoint found at {ckpt_path}, starting fresh")

    # ── Training Loop ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING STARTED")
    print("=" * 70)
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} "
          f"{'CER':<10} {'WER':<10} {'Word Acc':<10} {'LR':<12}")
    print("─" * 70)

    training_start = time.time()
    history = {"train_loss": [], "val_loss": [], "cer": [], "wer": [], "word_acc": []}

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # ── Validate ─────────────────────────────────────────────────
        val_loss = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # ── Compute Metrics ──────────────────────────────────────────
        avg_cer, avg_wer, word_acc = compute_epoch_metrics(
            model, val_loader, idx_to_char, device
        )

        # ── Learning Rate Schedule ───────────────────────────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Record History ───────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["cer"].append(avg_cer)
        history["wer"].append(avg_wer)
        history["word_acc"].append(word_acc)

        # ── Print Progress ───────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        save_marker = ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics = {
                "cer": avg_cer,
                "wer": avg_wer,
                "word_acc": word_acc,
            }
            save_checkpoint(
                model, optimizer, epoch, val_loss, metrics,
                char_to_idx, idx_to_char
            )
            save_marker = " ★"

        print(f"  {epoch + 1:<8} {train_loss:<14.4f} {val_loss:<14.4f} "
              f"{avg_cer:<10.4f} {avg_wer:<10.4f} {word_acc:<10.2%} "
              f"{current_lr:<12.6f} ({epoch_time:.1f}s){save_marker}")

        # ── Show Predictions Periodically ────────────────────────────
        if (epoch + 1) % config.LOG_INTERVAL == 0 or epoch == 0:
            show_sample_predictions(
                model, val_loader, idx_to_char, device, num_samples=5
            )

        # ── Early Stopping Check ─────────────────────────────────────
        if early_stopping is not None:
            if early_stopping(val_loss, epoch):
                print(f"\n[Train] Early stopping triggered at epoch {epoch + 1}")
                break

    # ── Training Summary ─────────────────────────────────────────────────
    total_time = time.time() - training_start
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time:     {total_time / 60:.1f} minutes")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Epochs trained: {epoch + 1 - start_epoch}")

    # ── Final Evaluation with Best Model ─────────────────────────────────
    print("\n[Train] Final evaluation with best model...")
    best_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")
    if os.path.exists(best_ckpt):
        checkpoint = load_checkpoint(best_ckpt, device)
        model.load_state_dict(checkpoint["model_state_dict"])

        avg_cer, avg_wer, word_acc = compute_epoch_metrics(
            model, val_loader, idx_to_char, device
        )

        print(f"\n  Best Model Performance:")
        print(f"  ─────────────────────────")
        print(f"  CER (Character Error Rate): {avg_cer:.4f}")
        print(f"  WER (Word Error Rate):      {avg_wer:.4f}")
        print(f"  Word Accuracy:              {word_acc:.2%}")

        print(f"\n  Sample Predictions (Best Model):")
        show_sample_predictions(
            model, val_loader, idx_to_char, device, num_samples=15
        )

    # ── Save Training History ────────────────────────────────────────────
    import json
    history_path = os.path.join(config.OUTPUTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] Training history saved to: {history_path}")

    # ── Save Final Model ─────────────────────────────────────────────────
    final_path = os.path.join(config.CHECKPOINTS_DIR, "final_model.pth")
    save_checkpoint(
        model, optimizer, epoch, val_loss,
        {"cer": avg_cer, "wer": avg_wer, "word_acc": word_acc},
        char_to_idx, idx_to_char, filepath=final_path
    )
    print(f"[Train] Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
