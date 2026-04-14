"""
=============================================================================
CRNN Model — Convolutional Recurrent Neural Network
=============================================================================
Architecture for handwritten text recognition:

    Input: (batch, 1, 32, 128) — grayscale word images

    CNN Feature Extractor:
        Conv2D blocks with BatchNorm, ReLU, MaxPool
        Progressively: 1 → 64 → 128 → 256 → 512 channels
        Height reduced: 32 → 16 → 8 → 4 → 2
        Width preserved for sequence: 128 → 64 → 32 → 32 → 32

    Map-to-Sequence:
        Reshape (batch, 512, 2, 32) → (batch, 32, 1024)
        Each column of the feature map becomes a timestep

    RNN Sequence Modeling:
        2-layer Bidirectional LSTM (256 hidden per direction)
        Output: (batch, 32, 512)

    Output:
        Dense + LogSoftmax → (batch, 32, num_classes)
        32 timesteps, each predicting a character probability
"""

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """
    Single Bidirectional LSTM layer with a linear projection.

    This module wraps nn.LSTM with bidirectional=True and adds
    a linear layer to project the concatenated forward/backward
    outputs to the desired output size.

    Args:
        input_size: Size of input features
        hidden_size: Number of LSTM units per direction
        output_size: Size of the linear projection output
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)

        Returns:
            Tensor of shape (batch, seq_len, output_size)
        """
        recurrent, _ = self.rnn(x)
        output = self.linear(recurrent)
        return output


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for text recognition.

    Architecture:
        CNN → Map-to-Sequence → BiLSTM → Dense → Softmax

    The CNN extracts visual features from the image.
    The feature map columns are treated as a sequence.
    BiLSTM layers model temporal/sequential dependencies.
    Dense layer outputs character probabilities at each timestep.

    Args:
        num_classes: Number of output classes (vocabulary size + 1 for CTC blank)
        rnn_hidden: LSTM hidden size per direction (default: 256)

    Input:
        images: Tensor of shape (batch, 1, 32, 128)

    Output:
        log_probs: Tensor of shape (batch, 32, num_classes)
    """

    def __init__(self, num_classes, rnn_hidden=256):
        super(CRNN, self).__init__()
        self.num_classes = num_classes

        # ── CNN Feature Extractor ────────────────────────────────────────
        # Each block: Conv2D → BatchNorm → ReLU → MaxPool
        # Designed so height reduces to 2 while width reduces to 32

        self.cnn = nn.Sequential(
            # Block 1: (1, 32, 128) → (64, 16, 64)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),             # /2 both dims

            # Block 2: (64, 16, 64) → (128, 8, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),             # /2 both dims

            # Block 3: (128, 8, 32) → (256, 4, 32)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),   # /2 height only

            # Block 4: (256, 4, 32) → (512, 2, 32)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),   # /2 height only

            # Optional Block 5: Additional conv for more capacity
            # (512, 2, 32) → (512, 2, 32)  — no pooling
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── RNN Sequence Modeling ────────────────────────────────────────
        # After CNN: (batch, 512, 1, 31) → flatten to (batch, 31, 512)
        # Two stacked BiLSTM layers with dropout between them

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, rnn_hidden, rnn_hidden),
            nn.Dropout(0.3),
            BidirectionalLSTM(rnn_hidden, rnn_hidden, num_classes),
        )

        # ── Weight Initialization ────────────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using best practices:
            - Conv2D: Kaiming (He) initialization
            - BatchNorm: weight=1, bias=0
            - Linear: Xavier uniform
            - LSTM: Xavier for input weights, Orthogonal for hidden weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to 1 (helps LSTM remember)
                        n = param.size(0)
                        param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x):
        """
        Forward pass through the CRNN.

        Args:
            x: Input images, shape (batch, 1, 32, 128)

        Returns:
            Output predictions, shape (batch, time_steps, num_classes)
        """
        # ── CNN ──────────────────────────────────────────────────────────
        features = self.cnn(x)  # (batch, 512, 1, 31)

        # ── Map to Sequence ──────────────────────────────────────────────
        # Squeeze height, permute so width becomes sequence dimension
        b, c, h, w = features.size()
        features = features.view(b, c * h, w)   # (batch, 512*h, width)
        features = features.permute(0, 2, 1)     # (batch, width, 512*h)

        # ── RNN ──────────────────────────────────────────────────────────
        output = self.rnn(features)  # (batch, width, num_classes)

        return output


def build_model(num_classes, rnn_hidden=None, device=None):
    """
    Factory function to build and return the CRNN model.

    Args:
        num_classes: Number of output classes (vocab size + 1 for CTC blank)
        rnn_hidden: LSTM hidden size (default from config)
        device: Device to place model on

    Returns:
        model: CRNN model on the specified device
    """
    from src import config as cfg

    if rnn_hidden is None:
        rnn_hidden = cfg.RNN_HIDDEN
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRNN(num_classes=num_classes, rnn_hidden=rnn_hidden).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] CRNN Architecture:")
    print(f"  Num classes:       {num_classes}")
    print(f"  RNN hidden:        {rnn_hidden} (x2 bidirectional)")
    print(f"  Total params:      {total_params:,}")
    print(f"  Trainable params:  {trainable_params:,}")
    print(f"  Device:            {device}")

    # Verify dimensions with dummy input
    dummy = torch.randn(2, 1, cfg.IMG_HEIGHT, cfg.IMG_WIDTH).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Input shape:       {tuple(dummy.shape)}")
    print(f"  Output shape:      {tuple(out.shape)}")
    print(f"  Time steps:        {out.shape[1]}")

    return model
