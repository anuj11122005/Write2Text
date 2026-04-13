"""Improved training script for higher accuracy."""

import os
import sys
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import IAMDataset
from src.model import CRNN
from src.utils import load_char_mapping, decode_predictions
from src import config


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images, 0), labels


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=config.GRAD_CLIP):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    num_batches = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = [l.to(device) for l in labels]

        preds = model(imgs)
        log_probs = torch.log_softmax(preds, dim=2).permute(1, 0, 2)

        targets = torch.cat(labels)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
        input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = [l.to(device) for l in labels]

            preds = model(imgs)
            log_probs = torch.log_softmax(preds, dim=2).permute(1, 0, 2)

            targets = torch.cat(labels)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def evaluate_accuracy(model, loader, idx_to_char, device):
    """Compute word and character accuracy."""
    model.eval()
    correct_words = 0
    total_words = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_texts = decode_predictions(preds, idx_to_char)

            for pred_text, label_tensor in zip(pred_texts, labels):
                true_text = "".join([idx_to_char[i.item()] for i in label_tensor])

                if pred_text == true_text:
                    correct_words += 1
                total_words += 1

                min_len = min(len(pred_text), len(true_text))
                max_len = max(len(pred_text), len(true_text))
                matches = sum(1 for i in range(min_len) if pred_text[i] == true_text[i])
                correct_chars += matches
                total_chars += max_len

    word_acc = correct_words / total_words if total_words > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0

    return word_acc, char_acc


def main():
    """Main training function with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Load data
    train_path = os.path.join(config.DATASET_PROCESSED, "train.csv")
    val_path = os.path.join(config.DATASET_PROCESSED, "val.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"\nDataset: Train={len(train_df)}, Val={len(val_df)}")

    # Character mapping
    char_to_idx, idx_to_char, num_classes = load_char_mapping()
    print(f"Vocabulary: {len(char_to_idx)} chars")

    # Datasets
    train_dataset = IAMDataset(train_df, char_to_idx, config.IMG_WIDTH, config.IMG_HEIGHT)
    val_dataset = IAMDataset(val_df, char_to_idx, config.IMG_WIDTH, config.IMG_HEIGHT)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model
    model = CRNN(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Training setup
    criterion = nn.CTCLoss(blank=config.BLANK_IDX, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.LR_FACTOR, patience=config.LR_PATIENCE, verbose=True
    )

    # Early stopping
    best_val_loss = float('inf')
    best_word_acc = 0.0
    epochs_no_improve = 0

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("Training Started")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        word_acc, char_acc = evaluate_accuracy(model, val_loader, idx_to_char, device)

        # Scheduler
        scheduler.step(val_loss)

        # Time
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Word: {word_acc:.2%} | Char: {char_acc:.2%} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Save best model (based on word accuracy)
        improved = False
        if word_acc > best_word_acc:
            best_word_acc = word_acc
            improved = True
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if improved:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'word_acc': word_acc,
                'char_acc': char_acc,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
            }
            torch.save(checkpoint, os.path.join(config.CHECKPOINTS_DIR, "best_model.pth"))
            print(f"  [SAVED] Best model (Word Acc: {word_acc:.2%})")

        # Early stopping
        if config.EARLY_STOPPING and epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {epochs_no_improve} epochs)")
            break

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best Word Accuracy: {best_word_acc:.2%}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
