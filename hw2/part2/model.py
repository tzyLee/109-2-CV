import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # (N, 1, 28, 28) -> (N, 6, 24, 24)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (N, 6, 12, 12)
            nn.Conv2d(6, 16, 5),  # (N, 6, 12, 12) -> (N, 16, 8, 8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (N, 16, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        out = self.flatten(self.cnn(x))
        return self.fc(out)

    def name(self):
        return "ConvNet"


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 5),  # (N, 1, 28, 28) -> (N, 64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (N, 64, 12, 12)
            nn.Conv2d(64, 32, 5),  # (N, 64, 12, 12) -> (N, 32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (N, 32, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.fc(self.flatten(self.cnn(x)))

    def name(self):
        return "MyNet"