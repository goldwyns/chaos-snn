# train_baseline_npz

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.datasets.chbmit_npz import CHBMITNPZDataset
from src.models.cnn_baseline import CNNBaseline
from src.models.cnn_lstm_baseline import CNNLSTMBaseline

def train(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        p, logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p, _ = model(x)
            y_true.extend(y.numpy())
            y_pred.extend(p.cpu().numpy())
    return roc_auc_score(y_true, y_pred)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patients = args.patients.split(",")

    ds = CHBMITNPZDataset(args.npz_dir, patients)
    n_val = int(0.2 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=128)

    C = ds.X.shape[1]

    if args.model == "cnn":
        model = CNNBaseline(C)
    else:
        model = CNNLSTMBaseline(C)

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, AUC={auc:.3f}")
        best_auc = max(best_auc, auc)

    print(f"✅ Best AUC = {best_auc:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--model", choices=["cnn", "cnn_lstm"], default="cnn")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    main(args)
