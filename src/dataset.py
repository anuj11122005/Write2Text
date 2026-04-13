import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class IAMDataset(Dataset):
    def __init__(self, df, char_to_idx, img_width=128, img_height=32):
        self.df = df.reset_index(drop=True)
        self.char_to_idx = char_to_idx
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, path):
        """Resolve image path relative to project root."""
        if os.path.exists(path):
            return path

        # Handle paths relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Remove leading ../ or ./
        if path.startswith('../'):
            path = path[3:]
        elif path.startswith('./'):
            path = path[2:]

        resolved = os.path.join(project_root, path)
        if os.path.exists(resolved):
            return resolved

        return path  # Return original if nothing worked

    def preprocess(self, path):
        path = self._resolve_path(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Invalid image: {path}")

        # Add padding to preserve aspect ratio
        h, w = img.shape
        target_ratio = self.img_width / self.img_height
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Image is too wide, pad height
            new_h = int(w / target_ratio)
            pad_top = (new_h - h) // 2
            pad_bottom = new_h - h - pad_top
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)
        else:
            # Image is too tall, pad width
            new_w = int(h * target_ratio)
            pad_left = (new_w - w) // 2
            pad_right = new_w - w - pad_left
            img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)

        # Resize to target size
        img = cv2.resize(img, (self.img_width, self.img_height))

        # Normalize: white background (255) -> 0, black text (0) -> 1
        img = img.astype(np.float32) / 255.0
        img = 1.0 - img  # Invert: now text is 1, background is 0

        # Standardize
        img = (img - 0.5) / 0.5

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img

    def encode_label(self, text):
        return torch.tensor(
            [self.char_to_idx[c] for c in text],
            dtype=torch.long
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = self.preprocess(row['image_path'])
        label = self.encode_label(row['label'])

        return img, label