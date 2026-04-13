"""CRNN Model for Handwritten Text Recognition."""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Convolutional RNN for text recognition.

    Architecture:
        Input: (batch, 1, 32, 128) - grayscale images
        CNN: 1->64->128->256->512 channels
        RNN: 2-layer bidirectional LSTM
        Output: (batch, time_steps=32, num_classes)
    """

    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.num_classes = num_classes

        # CNN feature extraction
        self.cnn = nn.Sequential(
            # Block 1: (1, 32, 128) -> (64, 16, 64)
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (64, 16, 64) -> (128, 8, 32)
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (128, 8, 32) -> (256, 4, 32) - pool height only
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # /2 height, same width

            # Block 4: (256, 4, 32) -> (512, 2, 32) - pool height only
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # /2 height, same width
        )

        # After CNN: (batch, 512, 2, 32)
        # For RNN: (batch, 32, 512*2) = (batch, 32, 1024)
        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # (batch, 512, 2, 32)

        # Reshape for RNN: (batch, channels, height, width) -> (batch, width, features)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(b, w, c * h)  # (batch, 32, 1024)

        # RNN
        x, _ = self.rnn(x)

        # Output layer
        x = self.fc(x)  # (batch, 32, num_classes)

        return x
