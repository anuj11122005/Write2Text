import torch
from torch.utils.data import Dataset
import cv2

class IAMDataset(Dataset):
    def __init__(self, df, char_to_idx, img_width=128, img_height=32):
        self.df = df.reset_index(drop=True)
        self.char_to_idx = char_to_idx
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return len(self.df)

def preprocess(self, path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 🔒 HARD GUARANTEE
    if img is None:
        raise ValueError(f"Invalid image: {path}")

    img = cv2.resize(img, (self.img_width, self.img_height))

    img = img / 255.0
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