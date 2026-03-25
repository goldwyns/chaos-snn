#run_baseline.py
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class CHBMITNPZDataset(Dataset):
    def __init__(self, npz_dir, patients, use="X"):
        self.X, self.y = [], []

        for p in patients:
            data = np.load(os.path.join(npz_dir, f"{p}.npz"))
            self.X.append(data[use])
            self.y.append(data["y"])

        self.X = np.concatenate(self.X).astype(np.float32)
        self.y = np.concatenate(self.y).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

# --------------------------------------------------
# Models
# --------------------------------------------------
class CNNBaseline(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 64, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)


class CNNLSTMBaseline(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(C, 64, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.cnn(x)              # (B, F, T')
        x = x.transpose(1, 2)        # (B, T', F)
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1])).squeeze(-1)

# --------------------------------------------------
# Train + Eval
# --------------------------------------------------
def train_and_eval(model, train_loader, val_loader, device, epochs=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    best_auc = 0.0

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                p = model(x.to(device))
                ys.append(y.numpy())
                ps.append(p.cpu().numpy())

        auc = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
        best_auc = max(best_auc, auc)
        print(f"Epoch {ep+1}: AUC={auc:.3f}")

    return best_auc

# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patients = args.patients.split(",")

    ds = CHBMITNPZDataset(args.npz_dir, patients, use="X")
    n = len(ds)
    train_ds, val_ds = torch.utils.data.random_split(ds, [int(0.8*n), n-int(0.8*n)])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=128)

    C = ds.X.shape[1]

    results = {}

    for name, model in {
        "cnn": CNNBaseline(C),
        "cnn_lstm": CNNLSTMBaseline(C)
    }.items():
        print(f"\n▶ Training {name.upper()}")
        model.to(device)
        auc = train_and_eval(model, train_loader, val_loader, device, args.epochs)
        results[name] = auc

    os.makedirs("analysis/baselines", exist_ok=True)
    with open("analysis/baselines/baseline_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n📊 BASELINE RESULTS")
    for k, v in results.items():
        print(f"{k}: AUC={v:.3f}")

# --------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    main(args)
