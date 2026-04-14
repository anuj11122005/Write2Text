"""
=============================================================================
PyTorch Dataset — IAM Handwriting Words
=============================================================================
Custom Dataset class that loads images, applies preprocessing + augmentation,
and encodes labels into numerical sequences for CTC training.

Preprocessing pipeline:
    1. Load grayscale image
    2. Aspect-ratio-preserving padding (white border)
    3. Resize to fixed dimensions (128 x 32)
    4. Invert (text becomes bright on dark background)
    5. Normalize to [-1, 1]
    6. Optional augmentation (rotation, noise)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src import config


class IAMDataset(Dataset):
    """
    PyTorch Dataset for IAM handwritten word images.

    Args:
        df: DataFrame with columns ['image_path', 'label']
        char_to_idx: Dict mapping characters to integer indices
        img_width: Target image width (default: 128)
        img_height: Target image height (default: 32)
        augment: Whether to apply data augmentation (default: False)
    """

    def __init__(self, df, char_to_idx, img_width=None, img_height=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.char_to_idx = char_to_idx
        self.img_width = img_width or config.IMG_WIDTH
        self.img_height = img_height or config.IMG_HEIGHT
        self.augment = augment

        # Pre-filter: remove rows with labels containing unknown characters
        valid_mask = self.df["label"].apply(
            lambda lbl: all(c in self.char_to_idx for c in str(lbl))
        )
        before = len(self.df)
        self.df = self.df[valid_mask].reset_index(drop=True)
        removed = before - len(self.df)
        if removed > 0:
            print(f"[Dataset] Removed {removed} samples with unknown characters")

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, path):
        """Resolve image path — handles relative and absolute paths."""
        if os.path.exists(path):
            return path

        # Try resolving relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Strip leading ../ or ./
        clean_path = path
        while clean_path.startswith("../"):
            clean_path = clean_path[3:]
        if clean_path.startswith("./"):
            clean_path = clean_path[2:]

        resolved = os.path.join(project_root, clean_path)
        if os.path.exists(resolved):
            return resolved

        return path  # Return original — will raise error in preprocess

    def preprocess(self, image_path):
        """
        Full preprocessing pipeline for a single image.

        Steps:
            1. Load as grayscale
            2. Pad to preserve aspect ratio
            3. Resize to target dimensions
            4. Invert colors (text bright, background dark)
            5. Normalize to [-1, 1]
            6. Apply augmentation (if enabled)

        Args:
            image_path: Path to the image file

        Returns:
            Tensor of shape (1, H, W) — single channel
        """
        path = self._resolve_path(image_path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Cannot load image: {path}")



        # ── Strict Padding-to-Ratio Preprocessing ────────────────────────
        h, w = img.shape
        target_ratio = self.img_width / self.img_height
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Image is too wide → pad top/bottom to reach ratio
            new_h = int(w / target_ratio)
            pad_top = (new_h - h) // 2
            pad_bottom = new_h - h - pad_top
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, 
                                   cv2.BORDER_CONSTANT, value=255)
        else:
            # Image is too tall → pad left/right to reach ratio
            new_w = int(h * target_ratio)
            pad_left = (new_w - w) // 2
            pad_right = new_w - w - pad_left
            img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, 
                                   cv2.BORDER_CONSTANT, value=255)

        # Now resize to fixed dimensions WITHOUT stretching
        img = cv2.resize(img, (self.img_width, self.img_height), 
                         interpolation=cv2.INTER_AREA)
        
        # Standard Normalization
        img = img.astype(np.float32) / 255.0
        img = 1.0 - img
        img = (img - 0.5) / 0.5

        # ── Augmentation (training only) ─────────────────────────────────
        if self.augment:
            img = self._apply_augmentation(img)

        # Convert to tensor: (1, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img_tensor

    def _apply_augmentation(self, img):
        """
        Apply random augmentations to the image.

        Augmentations:
            - Small random rotation (±2 degrees)
            - Gaussian noise
            - Random brightness shift
        """
        h, w = img.shape

        # Random rotation (small angle)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-config.AUG_ROTATION, config.AUG_ROTATION)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=0.0)

        # Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, config.AUG_NOISE_STD, img.shape).astype(np.float32)
            img = np.clip(img + noise, -1.0, 1.0)

        # Random brightness shift
        if np.random.random() < 0.3:
            shift = np.random.uniform(-0.1, 0.1)
            img = np.clip(img + shift, -1.0, 1.0)

        return img

    def encode_label(self, text):
        """
        Encode a text string into a sequence of integer indices.

        Unknown characters are skipped (should not happen after filtering).

        Args:
            text: Label string

        Returns:
            Tensor of shape (label_length,) with dtype long
        """
        encoded = []
        for c in text:
            if c in self.char_to_idx:
                encoded.append(self.char_to_idx[c])
            # Skip unknown characters silently
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            img: Tensor of shape (1, H, W)
            label: Tensor of shape (label_length,)

        Raises:
            Exception if image cannot be loaded (caught by DataLoader)
        """
        row = self.df.iloc[idx]
        img = self.preprocess(row["image_path"])
        label = self.encode_label(str(row["label"]))
        return img, label


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Since labels have variable length, we cannot stack them into a single tensor.
    Instead, we return images as a stacked tensor and labels as a list of tensors.

    Args:
        batch: List of (image_tensor, label_tensor) tuples

    Returns:
        images: Tensor of shape (batch_size, 1, H, W)
        labels: List of label tensors
    """
    # Filter out None entries (from failed image loads)
    batch = [(img, lbl) for img, lbl in batch if img is not None]

    if len(batch) == 0:
        return torch.zeros(1, 1, config.IMG_HEIGHT, config.IMG_WIDTH), [torch.zeros(1, dtype=torch.long)]

    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(labels)