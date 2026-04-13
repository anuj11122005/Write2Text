"""Complete training script for CRNN with diagnostics."""

import os
import sys
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
    """Collate function for DataLoader."""
    images, labels = zip(*batch)
    return torch.stack(images, 0), labels


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = [l.to(device) for l in labels]

        # Forward pass
        preds = model(imgs)  # (batch, time_steps, num_classes)

        # CTC expects (time, batch, classes)
        log_probs = torch.log_softmax(preds, dim=2)
        log_probs = log_probs.permute(1, 0, 2)  # (time, batch, classes)

        # Prepare targets
        targets = torch.cat(labels)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
        input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device)

        # Compute loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
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
            log_probs = torch.log_softmax(preds, dim=2)
            log_probs = log_probs.permute(1, 0, 2)

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

                # Word accuracy
                if pred_text == true_text:
                    correct_words += 1
                total_words += 1

                # Character accuracy
                min_len = min(len(pred_text), len(true_text))
                max_len = max(len(pred_text), len(true_text))
                matches = sum(1 for i in range(min_len) if pred_text[i] == true_text[i])
                correct_chars += matches
                total_chars += max_len

    word_acc = correct_words / total_words if total_words > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0

    return word_acc, char_acc


def show_predictions(model, loader, idx_to_char, device, num_samples=10):
    """Show sample predictions."""
    model.eval()
    print("\nSample Predictions:")
    print("-" * 50)

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= 1:  # Just first batch
                break
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_texts = decode_predictions(preds, idx_to_char)

            for j in range(min(num_samples, len(pred_texts))):
                true_text = "".join([idx_to_char[k.item()] for k in labels[j]])
                match = "✓" if pred_texts[j] == true_text else "✗"
                print(f"{match} Pred: '{pred_texts[j]:<15}' | True: '{true_text}'")


def main():
    """Main training function."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Load data
    train_path = os.path.join(config.DATASET_PROCESSED, "train_clean.csv")
    val_path = os.path.join(config.DATASET_PROCESSED, "val.csv")

    if not os.path.exists(train_path):
        print(f"ERROR: Train file not found: {train_path}")
        print("Run: python scripts/prepare_data.py")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"\nDataset:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")

    # Character mapping
    char_to_idx, idx_to_char, num_classes = load_char_mapping()
    print(f"  Vocabulary size: {len(char_to_idx)}")

    # Datasets and loaders
    train_dataset = IAMDataset(train_df, char_to_idx, config.IMG_WIDTH, config.IMG_HEIGHT)
    val_dataset = IAMDataset(val_df, char_to_idx, config.IMG_WIDTH, config.IMG_HEIGHT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Model
    print("\nModel:")
    model = CRNN(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Verify model dimensions
    dummy_input = torch.randn(2, 1, config.IMG_HEIGHT, config.IMG_WIDTH).to(device)
    dummy_output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {dummy_output.shape}")
    print(f"  Time steps: {dummy_output.shape[1]}, Classes: {dummy_output.shape[2]}")

    # Training setup
    criterion = nn.CTCLoss(blank=config.BLANK_IDX, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training Started")
    print("=" * 60)

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Accuracy
        word_acc, char_acc = evaluate_accuracy(model, val_loader, idx_to_char, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{config.NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Word Acc: {word_acc:.2%} | "
              f"Char Acc: {char_acc:.2%} | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

        # Show predictions every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            show_predictions(model, val_loader, idx_to_char, device, num_samples=5)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Final evaluation
    print("\nFinal Evaluation:")
    checkpoint = torch.load(os.path.join(config.CHECKPOINTS_DIR, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    word_acc, char_acc = evaluate_accuracy(model, val_loader, idx_to_char, device)
    print(f"Best model - Word Accuracy: {word_acc:.2%}, Char Accuracy: {char_acc:.2%}")
    show_predictions(model, val_loader, idx_to_char, device, num_samples=10)


if __name__ == "__main__":
    main()
