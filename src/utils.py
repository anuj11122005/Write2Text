"""Utility functions for CRNN."""

import json
import os
import torch

from src import config


def load_char_mapping():
    """Load character mapping from prepared data.

    Returns:
        char_to_idx: dict mapping char to index
        idx_to_char: dict mapping index to char
        num_classes: total number of classes including blank
    """
    mapping_path = os.path.join(config.DATASET_PROCESSED, "char_mapping.json")

    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        char_to_idx = mapping['char_to_idx']
        idx_to_char = {int(k): v for k, v in mapping['idx_to_char'].items()}
        num_classes = mapping['num_classes']
    else:
        # Fallback: build from scratch (for backwards compatibility)
        raise FileNotFoundError(f"Character mapping not found at {mapping_path}. "
                                  f"Run: python scripts/prepare_data.py")

    return char_to_idx, idx_to_char, num_classes


def build_char_mapping(labels):
    """Build character to index mapping from labels (deprecated, use load_char_mapping).

    Args:
        labels: List of strings

    Returns:
        char_to_idx: dict mapping char to index (1-indexed, 0 reserved for CTC blank)
        idx_to_char: dict mapping index to char
        num_classes: total number of classes including blank
    """
    chars = sorted(list(set("".join(labels))))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    num_classes = len(chars) + 1  # +1 for CTC blank
    return char_to_idx, idx_to_char, num_classes


def decode_predictions(preds, idx_to_char, blank_idx=0):
    """CTC Greedy Decoding.

    Args:
        preds: Model output tensor of shape (batch, time, classes) or (time, batch, classes)
        idx_to_char: Dict mapping indices to characters
        blank_idx: Index for CTC blank token

    Returns:
        List of decoded strings
    """
    if preds.dim() == 3:
        if preds.size(0) < preds.size(1):
            # (time, batch, classes) format
            preds = preds.argmax(2).permute(1, 0)
        else:
            # (batch, time, classes) format
            preds = preds.argmax(2)

    texts = []
    for pred in preds:
        prev = -1
        text = ""
        for p in pred:
            p = p.item()
            if p != prev and p != blank_idx:
                text += idx_to_char.get(p, "")
            prev = p
        texts.append(text)

    return texts


def compute_accuracy(model, loader, idx_to_char, device="cpu"):
    """Compute Character Error Rate (CER) and Word Error Rate (WER).

    Returns:
        char_acc: Character-level accuracy
        word_acc: Word-level accuracy
    """
    model.eval()
    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_texts = decode_predictions(preds, idx_to_char)

            for pred_text, label_tensor in zip(pred_texts, labels):
                true_text = "".join([idx_to_char[i.item()] for i in label_tensor])

                # Character-level accuracy
                min_len = min(len(pred_text), len(true_text))
                max_len = max(len(pred_text), len(true_text))
                matches = sum(1 for i in range(min_len) if pred_text[i] == true_text[i])
                correct_chars += matches
                total_chars += max_len

                # Word-level accuracy
                total_words += 1
                if pred_text == true_text:
                    correct_words += 1

    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    word_acc = correct_words / total_words if total_words > 0 else 0

    return char_acc, word_acc
