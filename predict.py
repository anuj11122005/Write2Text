"""
=============================================================================
Predict.py — Inference Script for Write2Text CRNN
=============================================================================
Supports three prediction modes:
    1. Single image → text prediction
    2. Batch prediction from CSV file
    3. Batch prediction from a directory of images

Features:
    - Greedy decoding (fast)
    - Beam search decoding (more accurate)
    - Confidence scores
    - Output to console and/or JSON file

Usage:
    python predict.py --image path/to/image.png
    python predict.py --csv path/to/test.csv
    python predict.py --dir path/to/images/
    python predict.py --image img.png --beam_search
    python predict.py --csv test.csv --output results.json
"""

import os
import sys
import glob
import argparse
import json
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src import config
from src.dataset import IAMDataset, collate_fn
from src.model import CRNN
from src.utils import (
    decode_predictions, beam_search_decode_batch,
    load_checkpoint, compute_cer, compute_wer, compute_batch_metrics
)


def predict_single_image(model, image_path, char_to_idx, idx_to_char, device,
                         use_beam_search=False, beam_width=10):
    """
    Predict text from a single handwritten word image.

    Args:
        model: Trained CRNN model
        image_path: Path to the image file
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        device: Computation device
        use_beam_search: Use beam search decoding
        beam_width: Beam width for beam search

    Returns:
        predicted_text: The recognized text string
    """
    # Create a temporary dataset with just this image
    df = pd.DataFrame([{"image_path": image_path, "label": ""}])
    dataset = IAMDataset(df, char_to_idx, augment=False)

    img, _ = dataset[0]
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        preds = model(img)

        if use_beam_search:
            pred_texts = beam_search_decode_batch(
                preds, idx_to_char, beam_width=beam_width
            )
        else:
            pred_texts = decode_predictions(preds, idx_to_char)

    return pred_texts[0]


def predict_batch_csv(model, csv_path, char_to_idx, idx_to_char, device,
                      use_beam_search=False, beam_width=10):
    """
    Predict text for all images in a CSV file.

    CSV must have columns: image_path, label

    Args:
        model: Trained CRNN model
        csv_path: Path to CSV file
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        device: Computation device
        use_beam_search: Use beam search decoding
        beam_width: Beam width

    Returns:
        results: List of dicts with predictions and metrics
    """
    df = pd.read_csv(csv_path)
    dataset = IAMDataset(df, char_to_idx, augment=False)
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    results = []
    sample_idx = 0

    model.eval()
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

            for j, pred_text in enumerate(pred_texts):
                if sample_idx < len(dataset.df):
                    true_text = "".join(
                        [idx_to_char.get(k.item(), "") for k in labels[j]]
                    )
                    cer = compute_cer(pred_text, true_text)

                    results.append({
                        "image_path": dataset.df.iloc[sample_idx]["image_path"],
                        "predicted": pred_text,
                        "actual": true_text,
                        "correct": pred_text == true_text,
                        "cer": round(cer, 4)
                    })
                    sample_idx += 1

    return results


def predict_directory(model, dir_path, char_to_idx, idx_to_char, device,
                      use_beam_search=False, beam_width=10):
    """
    Predict text for all images in a directory.

    Supports: .png, .jpg, .jpeg, .bmp, .tiff

    Args:
        model: Trained CRNN model
        dir_path: Path to directory containing images
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        device: Computation device
        use_beam_search: Use beam search
        beam_width: Beam width

    Returns:
        results: List of dicts with predictions
    """
    # Find all image files
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(dir_path, ext)))

    if not image_files:
        print(f"[Predict] No images found in: {dir_path}")
        return []

    print(f"[Predict] Found {len(image_files)} images in: {dir_path}")

    results = []
    model.eval()

    for img_path in sorted(image_files):
        try:
            pred_text = predict_single_image(
                model, img_path, char_to_idx, idx_to_char, device,
                use_beam_search=use_beam_search, beam_width=beam_width
            )
            results.append({
                "image_path": img_path,
                "predicted": pred_text,
            })
        except Exception as e:
            print(f"  [Warning] Failed to process {img_path}: {e}")
            results.append({
                "image_path": img_path,
                "predicted": "[ERROR]",
                "error": str(e)
            })

    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Predict text from handwritten word images"
    )
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINTS_DIR, "best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str,
                        help="Path to a single image for prediction")
    parser.add_argument("--csv", type=str,
                        help="Path to CSV file with image paths and labels")
    parser.add_argument("--dir", type=str,
                        help="Path to directory of images")
    parser.add_argument("--output", type=str,
                        help="Path to save predictions as JSON")
    parser.add_argument("--beam_search", action="store_true",
                        help="Use beam search decoding (slower but more accurate)")
    parser.add_argument("--beam_width", type=int, default=config.BEAM_WIDTH,
                        help=f"Beam width (default: {config.BEAM_WIDTH})")
    parser.add_argument("--num_show", type=int, default=20,
                        help="Number of predictions to display (default: 20)")

    args = parser.parse_args()

    # ── Device Setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Device: {device}")

    # ── Load Checkpoint ──────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("Train a model first: python train.py")
        sys.exit(1)

    checkpoint = load_checkpoint(args.checkpoint, device)
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    num_classes = checkpoint.get("num_classes", len(char_to_idx) + 1)

    # ── Load Model ───────────────────────────────────────────────────────
    model = CRNN(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    metrics = checkpoint.get("metrics", {})

    print(f"[Predict] Loaded checkpoint:")
    print(f"  Epoch:         {epoch}")
    print(f"  Val loss:      {val_loss}")
    if metrics:
        print(f"  CER:           {metrics.get('cer', '?'):.4f}")
        print(f"  Word accuracy: {metrics.get('word_acc', '?'):.2%}")
    print(f"  Decoding:      {'Beam Search (width={})'.format(args.beam_width) if args.beam_search else 'Greedy'}")

    # ── Run Prediction ───────────────────────────────────────────────────
    print("\n" + "=" * 60)

    if args.image:
        # ── Single Image ─────────────────────────────────────────────
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)

        pred_text = predict_single_image(
            model, args.image, char_to_idx, idx_to_char, device,
            use_beam_search=args.beam_search, beam_width=args.beam_width
        )

        print(f"  Image:      {args.image}")
        print(f"  Prediction: \"{pred_text}\"")
        print("=" * 60)

    elif args.csv:
        # ── Batch from CSV ───────────────────────────────────────────
        if not os.path.exists(args.csv):
            print(f"[ERROR] CSV not found: {args.csv}")
            sys.exit(1)

        results = predict_batch_csv(
            model, args.csv, char_to_idx, idx_to_char, device,
            use_beam_search=args.beam_search, beam_width=args.beam_width
        )

        # Compute overall metrics
        pred_texts = [r["predicted"] for r in results]
        true_texts = [r["actual"] for r in results]
        avg_cer, avg_wer, word_acc = compute_batch_metrics(pred_texts, true_texts)

        correct = sum(1 for r in results if r["correct"])
        total = len(results)

        print(f"  Results: {correct}/{total} correct ({word_acc:.2%})")
        print(f"  Average CER: {avg_cer:.4f}")
        print(f"  Average WER: {avg_wer:.4f}")
        print("=" * 60)

        # Show sample predictions
        print(f"\n  {'Status':<8} {'Predicted':<20} {'Actual':<20} {'CER':<8}")
        print("  " + "─" * 55)

        for r in results[:args.num_show]:
            status = "✓" if r["correct"] else "✗"
            print(f"  {status:<8} {r['predicted']:<20} {r['actual']:<20} {r['cer']:<8.4f}")

        if len(results) > args.num_show:
            print(f"\n  ... and {len(results) - args.num_show} more predictions")

    elif args.dir:
        # ── Batch from Directory ─────────────────────────────────────
        if not os.path.isdir(args.dir):
            print(f"[ERROR] Directory not found: {args.dir}")
            sys.exit(1)

        results = predict_directory(
            model, args.dir, char_to_idx, idx_to_char, device,
            use_beam_search=args.beam_search, beam_width=args.beam_width
        )

        print(f"\n  {'#':<5} {'File':<35} {'Prediction':<20}")
        print("  " + "─" * 55)
        for i, r in enumerate(results[:args.num_show]):
            filename = os.path.basename(r["image_path"])
            print(f"  {i+1:<5} {filename:<35} {r['predicted']:<20}")

    else:
        print("[ERROR] Specify --image, --csv, or --dir")
        parser.print_help()
        sys.exit(1)

    # ── Save Results to JSON ─────────────────────────────────────────────
    if args.output and (args.csv or args.dir):
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Predict] Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
