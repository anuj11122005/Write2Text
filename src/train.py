"""Training utilities for CRNN."""

import torch
from torch.nn.utils import clip_grad_norm_

from src.utils import decode_predictions


def train(model, loader, optimizer, criterion, device="cpu", scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = [l.to(device) for l in labels]

        preds = model(imgs)

        # CTC expects (T, N, C)
        log_probs = torch.log_softmax(preds, dim=2)
        log_probs = log_probs.permute(1, 0, 2)

        targets = torch.cat(labels)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
        input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, criterion, device="cpu"):
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


def evaluate_batch(model, imgs, labels, idx_to_char, device="cpu"):
    """Evaluate a single batch and return predictions."""
    model.eval()
    with torch.no_grad():
        imgs = imgs.to(device)
        preds = model(imgs)
        pred_texts = decode_predictions(preds, idx_to_char)

        true_texts = ["".join([idx_to_char[i.item()] for i in label_tensor]) for label_tensor in labels]

    return pred_texts, true_texts