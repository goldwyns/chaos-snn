# src.models.cnn_lstm_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMBaseline(nn.Module):
    def __init__(self, n_channels, hidden=64):
        super().__init__()

        self.conv = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, C, T)
        x = F.relu(self.conv(x))          # (B, 64, T)
        x = x.permute(0, 2, 1)            # (B, T, 64)
        out, _ = self.lstm(x)             # (B, T, H)
        h = out[:, -1]                    # last timestep
        logits = self.fc(h).squeeze(-1)
        prob = torch.sigmoid(logits)
        return prob, logits
