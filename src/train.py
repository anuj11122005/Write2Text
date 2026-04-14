"""
=============================================================================
Training Utilities — Train / Validate / Early Stopping
=============================================================================
Core training functions used by the main train.py script.
Separated for modularity and reuse.
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from src import config
from src.utils import decode_predictions, compute_batch_metrics


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait for improvement (default: 15)
        min_delta: Minimum change to qualify as improvement (default: 1e-4)
        verbose: Print messages when stopping (default: True)
    """

    def __init__(self, patience=None, min_delta=1e-4, verbose=True):
        self.patience = patience or config.EARLY_STOPPING_PATIENCE
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False

        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"\n[EarlyStopping] No improvement for {self.patience} epochs. "
                          f"Best val_loss: {self.best_loss:.4f} at epoch {self.best_epoch + 1}")
                return True
            return False


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    """
    Train the model for one epoch.

    Args:
        model: CRNN model
        loader: Training DataLoader
        optimizer: Optimizer
        criterion: CTC Loss function
        device: Computation device
        grad_clip: Gradient clipping value (default from config)

    Returns:
        average_loss: Mean loss over all batches
    """
    if grad_clip is None:
        grad_clip = config.GRAD_CLIP

    model.train()
    total_loss = 0.0
    num_batches = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = [l.to(device) for l in labels]

        # Forward pass
        preds = model(imgs)  # (batch, time_steps, num_classes)

        # CTC requires (time, batch, classes) log-probabilities
        log_probs = torch.log_softmax(preds, dim=2)
        log_probs = log_probs.permute(1, 0, 2)  # (time, batch, classes)

        # Prepare CTC inputs
        targets = torch.cat(labels)
        target_lengths = torch.tensor(
            [len(l) for l in labels], dtype=torch.long, device=device
        )
        input_lengths = torch.full(
            (imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device
        )

        # Compute CTC loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        # Skip batch if loss is inf or nan (corrupted sample)
        if torch.isinf(loss) or torch.isnan(loss):
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    return avg_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Validate the model on the validation set.

    Args:
        model: CRNN model
        loader: Validation DataLoader
        criterion: CTC Loss function
        device: Computation device

    Returns:
        average_loss: Mean validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = [l.to(device) for l in labels]

            preds = model(imgs)
            log_probs = torch.log_softmax(preds, dim=2)
            log_probs = log_probs.permute(1, 0, 2)

            targets = torch.cat(labels)
            target_lengths = torch.tensor(
                [len(l) for l in labels], dtype=torch.long, device=device
            )
            input_lengths = torch.full(
                (imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if not (torch.isinf(loss) or torch.isnan(loss)):
                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    return avg_loss


def compute_epoch_metrics(model, loader, idx_to_char, device):
    """
    Compute CER, WER, and word accuracy for the current model.

    Args:
        model: CRNN model
        loader: DataLoader
        idx_to_char: Index to character mapping
        device: Computation device

    Returns:
        avg_cer, avg_wer, word_acc
    """
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_texts = decode_predictions(preds, idx_to_char)

            for label_tensor in labels:
                true_text = "".join(
                    [idx_to_char.get(i.item(), "") for i in label_tensor]
                )
                all_trues.append(true_text)

            all_preds.extend(pred_texts)

    avg_cer, avg_wer, word_acc = compute_batch_metrics(all_preds, all_trues)
    return avg_cer, avg_wer, word_acc