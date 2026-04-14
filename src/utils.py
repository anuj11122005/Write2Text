"""
=============================================================================
Utility Functions — Decoding, Metrics, and Helpers
=============================================================================
Contains:
    - Character mapping loader
    - CTC Greedy Decoding
    - CTC Beam Search Decoding
    - Character Error Rate (CER) computation
    - Word Error Rate (WER) computation
    - Model save/load helpers
    - Accuracy evaluation
"""

import json
import os
import torch
import numpy as np

from src import config


# =============================================================================
# Character Mapping
# =============================================================================

def load_char_mapping(mapping_path=None):
    """
    Load character mapping from a previously saved JSON file.

    Args:
        mapping_path: Path to char_mapping.json (default: from config)

    Returns:
        char_to_idx: dict mapping char → index
        idx_to_char: dict mapping index → char
        num_classes: total number of classes (including CTC blank)
    """
    if mapping_path is None:
        mapping_path = os.path.join(config.DATASET_PROCESSED, "char_mapping.json")

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Character mapping not found at: {mapping_path}\n"
            f"Run data preparation first:\n"
            f"  python src/data_loader.py\n"
            f"  OR python scripts/prepare_data.py"
        )

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    char_to_idx = mapping["char_to_idx"]
    idx_to_char = {int(k): v for k, v in mapping["idx_to_char"].items()}
    num_classes = mapping["num_classes"]

    return char_to_idx, idx_to_char, num_classes


def build_char_mapping(labels):
    """
    Build character mapping from a list of label strings.

    Index 0 is reserved for the CTC blank token.

    Args:
        labels: List of strings

    Returns:
        char_to_idx, idx_to_char, num_classes
    """
    chars = sorted(list(set("".join(labels))))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    num_classes = len(chars) + 1  # +1 for CTC blank
    return char_to_idx, idx_to_char, num_classes


# =============================================================================
# CTC Greedy Decoding
# =============================================================================

def decode_predictions(preds, idx_to_char, blank_idx=0):
    """
    CTC Greedy Decoding — takes argmax at each timestep, collapses repeated
    characters, and removes blanks.

    Algorithm:
        1. Take argmax at each timestep
        2. Remove consecutive duplicates
        3. Remove blank tokens
        4. Map indices to characters

    Args:
        preds: Model output tensor
               Shape: (batch, time, classes) or (time, batch, classes)
        idx_to_char: Dict mapping index → character
        blank_idx: Index of CTC blank token (default: 0)

    Returns:
        List of decoded strings
    """
    # Model output is always (batch, time, classes)
    if preds.dim() == 3:
        preds = preds.argmax(dim=2)  # → (batch, time)

    texts = []
    for seq in preds:
        prev_idx = -1
        chars = []
        for idx in seq:
            idx = idx.item()
            # Skip if same as previous (CTC collapse) or if blank
            if idx != prev_idx and idx != blank_idx:
                char = idx_to_char.get(idx, "")
                if char:
                    chars.append(char)
            prev_idx = idx
        texts.append("".join(chars))

    return texts


# =============================================================================
# CTC Beam Search Decoding
# =============================================================================

def beam_search_decode(log_probs, idx_to_char, blank_idx=0, beam_width=10):
    """
    CTC Beam Search Decoding — explores multiple hypotheses for better accuracy.

    This is a simplified prefix beam search that maintains the top-K candidates
    at each timestep. Significantly better than greedy decoding for noisy outputs.

    Args:
        log_probs: Log probabilities from model, shape (time, classes) — single sample
        idx_to_char: Dict mapping index → character
        blank_idx: Index of CTC blank token
        beam_width: Number of beams to keep (default: 10)

    Returns:
        Best decoded string
    """
    T, C = log_probs.shape
    probs = np.exp(log_probs)  # Convert log-probs to probs

    # Each beam: (prefix_string, probability)
    beams = [("", 1.0)]

    for t in range(T):
        new_beams = {}

        for prefix, prob in beams:
            for c in range(C):
                p = probs[t, c]
                new_prob = prob * p

                if c == blank_idx:
                    # Blank extends the same prefix
                    key = prefix
                elif prefix and idx_to_char.get(c, "") == prefix[-1]:
                    # Same character as last → collapse (CTC rule)
                    key = prefix
                else:
                    # New character
                    key = prefix + idx_to_char.get(c, "")

                if key in new_beams:
                    new_beams[key] = new_beams[key] + new_prob
                else:
                    new_beams[key] = new_prob

        # Keep top-K beams
        sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
        beams = sorted_beams[:beam_width]

    # Return best beam
    return beams[0][0] if beams else ""


def beam_search_decode_batch(preds, idx_to_char, blank_idx=0, beam_width=10):
    """
    Apply beam search decoding to a batch of predictions.

    Args:
        preds: Model output tensor, shape (batch, time, classes)
        idx_to_char: Dict mapping index → character
        blank_idx: CTC blank index
        beam_width: Number of beams

    Returns:
        List of decoded strings
    """
    # Model output is always (batch, time, classes)
    # Convert to log probabilities
    log_probs = torch.log_softmax(preds, dim=2)

    texts = []
    for i in range(log_probs.size(0)):
        sample_log_probs = log_probs[i].cpu().numpy()
        text = beam_search_decode(
            sample_log_probs, idx_to_char,
            blank_idx=blank_idx, beam_width=beam_width
        )
        texts.append(text)

    return texts


# =============================================================================
# Metrics — CER and WER
# =============================================================================

def edit_distance(s1, s2):
    """
    Compute Levenshtein (edit) distance between two sequences.

    Uses dynamic programming with O(m*n) time and O(min(m,n)) space.

    Args:
        s1, s2: Two sequences (strings or lists)

    Returns:
        Integer edit distance
    """
    m, n = len(s1), len(s2)

    # Ensure s1 is the shorter sequence for space optimization
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m

    # DP with single row optimization
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for j in range(1, n + 1):
        curr[0] = j
        for i in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[i] = prev[i - 1]
            else:
                curr[i] = 1 + min(prev[i - 1], prev[i], curr[i - 1])
        prev, curr = curr, prev

    return prev[m]


def compute_cer(predicted, ground_truth):
    """
    Compute Character Error Rate (CER).

    CER = edit_distance(predicted, ground_truth) / len(ground_truth)

    A CER of 0.0 means perfect prediction.
    A CER of 1.0 means every character is wrong.

    Args:
        predicted: Predicted string
        ground_truth: True string

    Returns:
        Float CER value
    """
    if len(ground_truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    dist = edit_distance(predicted, ground_truth)
    return dist / len(ground_truth)


def compute_wer(predicted, ground_truth):
    """
    Compute Word Error Rate (WER).

    WER = edit_distance(predicted_words, ground_truth_words) / len(ground_truth_words)

    For single-word predictions (IAM words dataset), this becomes:
        0.0 if predicted == ground_truth, else 1.0

    Args:
        predicted: Predicted string
        ground_truth: True string

    Returns:
        Float WER value
    """
    pred_words = predicted.split()
    true_words = ground_truth.split()

    if len(true_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    dist = edit_distance(pred_words, true_words)
    return dist / len(true_words)


def compute_batch_metrics(pred_texts, true_texts):
    """
    Compute CER and WER over a batch of predictions.

    Args:
        pred_texts: List of predicted strings
        true_texts: List of ground truth strings

    Returns:
        avg_cer: Average Character Error Rate
        avg_wer: Average Word Error Rate
        word_accuracy: Fraction of exactly correct predictions
    """
    assert len(pred_texts) == len(true_texts), "Prediction and truth lists must match"

    total_cer = 0.0
    total_wer = 0.0
    correct_words = 0

    for pred, true in zip(pred_texts, true_texts):
        total_cer += compute_cer(pred, true)
        total_wer += compute_wer(pred, true)
        if pred == true:
            correct_words += 1

    n = len(pred_texts)
    avg_cer = total_cer / n if n > 0 else 0.0
    avg_wer = total_wer / n if n > 0 else 0.0
    word_accuracy = correct_words / n if n > 0 else 0.0

    return avg_cer, avg_wer, word_accuracy


# =============================================================================
# Model Save / Load
# =============================================================================

def save_checkpoint(model, optimizer, epoch, val_loss, metrics, char_to_idx,
                    idx_to_char, filepath=None):
    """
    Save a training checkpoint with all necessary state.

    Args:
        model: The CRNN model
        optimizer: The optimizer
        epoch: Current epoch number
        val_loss: Validation loss
        metrics: Dict with metrics (cer, wer, word_acc, char_acc)
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        filepath: Save path (default: checkpoints/best_model.pth)
    """
    if filepath is None:
        filepath = os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "metrics": metrics,
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "num_classes": len(char_to_idx) + 1,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    return filepath


def load_checkpoint(filepath, device=None):
    """
    Load a training checkpoint.

    Args:
        filepath: Path to checkpoint .pth file
        device: Device to map tensors to

    Returns:
        checkpoint: Dict with all saved state
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Convert idx_to_char keys back to int
    if "idx_to_char" in checkpoint:
        checkpoint["idx_to_char"] = {
            int(k): v for k, v in checkpoint["idx_to_char"].items()
        }

    return checkpoint


# =============================================================================
# Evaluation Helper
# =============================================================================

def evaluate_model(model, loader, idx_to_char, device, use_beam_search=False,
                   beam_width=10):
    """
    Evaluate model on a DataLoader and compute all metrics.

    Args:
        model: CRNN model
        loader: DataLoader
        idx_to_char: Index to character mapping
        device: Computation device
        use_beam_search: Whether to use beam search (slower but better)
        beam_width: Beam width for beam search

    Returns:
        avg_cer: Average Character Error Rate
        avg_wer: Average Word Error Rate
        word_acc: Word-level accuracy
    """
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs)

            if use_beam_search:
                pred_texts = beam_search_decode_batch(
                    preds, idx_to_char, beam_width=beam_width
                )
            else:
                pred_texts = decode_predictions(preds, idx_to_char)

            # Decode true labels to strings
            for label_tensor in labels:
                true_text = "".join(
                    [idx_to_char.get(i.item(), "") for i in label_tensor]
                )
                all_trues.append(true_text)

            all_preds.extend(pred_texts)

    avg_cer, avg_wer, word_acc = compute_batch_metrics(all_preds, all_trues)

    return avg_cer, avg_wer, word_acc


def show_sample_predictions(model, loader, idx_to_char, device, num_samples=10,
                            use_beam_search=False):
    """
    Print sample predictions vs ground truth for visual inspection.

    Args:
        model: CRNN model
        loader: DataLoader
        idx_to_char: Index to character mapping
        device: Computation device
        num_samples: Number of samples to show
        use_beam_search: Whether to use beam search
    """
    model.eval()
    samples_shown = 0

    print("\n" + "-" * 55)
    print(f"  {'Status':<8} {'Predicted':<20} {'Ground Truth':<20}")
    print("-" * 55)

    with torch.no_grad():
        for imgs, labels in loader:
            if samples_shown >= num_samples:
                break

            imgs = imgs.to(device)
            preds = model(imgs)

            if use_beam_search:
                pred_texts = beam_search_decode_batch(preds, idx_to_char)
            else:
                pred_texts = decode_predictions(preds, idx_to_char)

            for j in range(min(len(pred_texts), num_samples - samples_shown)):
                true_text = "".join(
                    [idx_to_char.get(k.item(), "") for k in labels[j]]
                )
                match = "  [Y]" if pred_texts[j] == true_text else "  [N]"
                print(f"  {match:<8} {pred_texts[j]:<20} {true_text:<20}")
                samples_shown += 1

    print("-" * 55)
