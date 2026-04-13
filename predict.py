"""Prediction script for trained CRNN model."""

import os
import sys
import argparse
import cv2
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import IAMDataset
from src.model import CRNN
from src.utils import decode_predictions
from src import config


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images, 0), labels


def predict_single(model, image_path, char_to_idx, idx_to_char, device):
    """Predict text from a single image."""
    # Create dummy dataframe for single image
    df = pd.DataFrame([{'image_path': image_path, 'label': ''}])
    dataset = IAMDataset(df, char_to_idx, config.IMG_WIDTH, config.IMG_HEIGHT)

    img, _ = dataset[0]
    img = img.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(img)
        pred_text = decode_predictions(pred, idx_to_char)[0]

    return pred_text


def predict_batch(model, csv_path, char_to_idx, idx_to_char, device):
    """Predict on a batch of images from CSV."""
    df = pd.read_csv(csv_path)
    dataset = IAMDataset(df, char_to_idx, config.IMG_WIDTH, config.IMG_HEIGHT)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    results = []
    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_texts = decode_predictions(preds, idx_to_char)

            for i, pred_text in enumerate(pred_texts):
                idx = len(results)
                if idx < len(df):
                    true_text = df.iloc[idx]['label']
                    results.append({
                        'image_path': df.iloc[idx]['image_path'],
                        'predicted': pred_text,
                        'actual': true_text,
                        'correct': pred_text == true_text
                    })

    return results


def main():
    parser = argparse.ArgumentParser(description='Predict text from handwritten images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--csv', type=str, help='Path to CSV with images')
    parser.add_argument('--output', type=str, help='Output file for predictions')

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    num_classes = len(char_to_idx) + 1

    # Load model
    model = CRNN(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Val loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"Word accuracy: {checkpoint.get('word_acc', 'unknown'):.2%}")
    print("-" * 50)

    # Predict
    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

        pred_text = predict_single(model, args.image, char_to_idx, idx_to_char, device)
        print(f"Image: {args.image}")
        print(f"Prediction: '{pred_text}'")

    elif args.csv:
        # Batch prediction
        if not os.path.exists(args.csv):
            print(f"Error: CSV not found: {args.csv}")
            sys.exit(1)

        results = predict_batch(model, args.csv, char_to_idx, idx_to_char, device)

        # Display results
        correct = sum(1 for r in results if r['correct'])
        total = len(results)

        print(f"\nPredictions ({correct}/{total} correct = {correct/total:.2%}):")
        print("-" * 50)

        for i, r in enumerate(results[:20]):  # Show first 20
            status = "✓" if r['correct'] else "✗"
            print(f"{status} Pred: '{r['predicted']:<15}' | True: '{r['actual']}'")

        if len(results) > 20:
            print(f"... and {len(results) - 20} more")

        # Save to file if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved predictions to: {args.output}")

    else:
        print("Error: Specify either --image or --csv")
        parser.print_help()


if __name__ == "__main__":
    main()
