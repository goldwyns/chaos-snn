# src/train_chbmit_npz.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json

from src.models.chaos_snn import ChaosSNNSeizureDetector


# ==========================================================
# Dataset
# ==========================================================
class CHBMITNPZDataset(Dataset):
    """
    Loads precomputed NPZ files with keys:
      - X_raw : (N, C, T)
      - U     : (N, C, T)  [chaos-modulated]
      - y     : (N,)
    """

    def __init__(self, npz_dir, patients, use="U"):
        super().__init__()
        assert use in ["X_raw", "U"]
        self.use = use

        X_all, y_all = [], []

        for p in patients:
            path = os.path.join(npz_dir, f"{p}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            data = np.load(path)
            X_all.append(data[use].astype(np.float32))
            y_all.append(data["y"].astype(np.float32))

        self.X = np.concatenate(X_all, axis=0)
        self.y = np.concatenate(y_all, axis=0)

        print(
            f"✅ NPZ dataset loaded: N={len(self.y)}, "
            f"C={self.X.shape[1]}, T={self.X.shape[2]}, "
            f"pos={int((self.y==1).sum())}, neg={int((self.y==0).sum())}"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


# ==========================================================
# Training
# ==========================================================
def train_chbmit_npz(
    npz_dir,
    patients,
    regime="edge",
    batch_size=128,
    n_epochs=10,
    lr=1e-4,
    val_split=0.2,
    use_amp=True,
    num_workers=0,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🖥 Device: {device}")
    print(f"🧠 Regime: {regime}")
    print(f"👥 Patients: {patients}")

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------
    dataset = CHBMITNPZDataset(
        npz_dir=npz_dir,
        patients=patients,
        use="U",          # 🔥 IMPORTANT: chaos-modulated input
    )

    n_total = len(dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # ------------------------------------------------------
    # Inspect shape
    # ------------------------------------------------------
    x0, _ = dataset[0]
    C, T = x0.shape
    fs = T / 5.0   # window_size_sec = 5
    dt = 1.0 / fs

    print(f"🧠 Sample shape: C={C}, T={T}")
    print(f"📏 Inferred fs ≈ {fs:.2f} Hz, dt={dt:.5f}s")

    # ------------------------------------------------------
    # Chaos-SNN parameters (UNCHANGED from previous datasets)
    # ------------------------------------------------------
    if regime == "edge":
        alpha, beta, gamma, lam = 3.5, 0.2, 0.5, 0.7
        alpha_rec = 1.0
    elif regime == "chaos_off":
        alpha, beta, gamma, lam = 0.0, 0.0, 0.0, 1.0
        alpha_rec = 0.8
    else:
        raise ValueError("Unknown regime")

    model = ChaosSNNSeizureDetector(
        n_channels=C,
        n_hidden=128,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lam=lam,
        v_th=1.0,
        alpha_rec=alpha_rec,
        dt=dt,
        tau_mem=20e-3,
        tau_syn=10e-3,
    ).to(device)

    # ------------------------------------------------------
    # Loss (class imbalance)
    # ------------------------------------------------------
    y_all = np.array(dataset.y)
    n_pos = (y_all == 1).sum()
    n_neg = (y_all == 0).sum()
    pos_weight = max(n_neg / max(n_pos, 1), 1.0)

    print(f"⚖ pos={n_pos}, neg={n_neg}, pos_weight={pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device != "cpu"))

    # ------------------------------------------------------
    # Training loop
    # ------------------------------------------------------
    best_auc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/chbmit_{regime}_npz_best.pt"

    val_auc_list = []
    best_auc = 0.0
    
    for epoch in range(1, n_epochs + 1):
    
        # ================== TRAIN ==================
        model.train()
        train_loss = 0.0
    
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
    
            optimizer.zero_grad()
    
            with torch.amp.autocast("cuda", enabled=(use_amp and device != "cpu")):
                p, extra = model(x)
                loss = criterion(extra["logits"], y)
    
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    
            train_loss += loss.item() * x.size(0)
    
        train_loss /= n_train
    
    
        # ================== VALIDATION ==================
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
    
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(use_amp and device != "cpu")):
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x = x.to(device)
                y = y.to(device)
    
                p, extra = model(x)
                loss = criterion(extra["logits"], y)
    
                val_loss += loss.item() * x.size(0)
                y_true.append(y.cpu())
                y_pred.append(p.cpu())
    
        val_loss /= n_val
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
    
        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = 0.5
    
        val_auc_list.append(float(auc))
    
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"AUC={auc:.3f}"
        )
        
        # ================== CHECKPOINT ==================
        if auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "auc": auc,
                    "epoch": epoch,
                    "patients": patients,
                    "regime": regime,
                },
                ckpt_path,
            )
            print(f"💾 Saved new best model → {ckpt_path}")
    
    
    # ================== SAVE TRAINING LOG ==================
    log = {
        "val_auc": val_auc_list,
        "best_auc": best_auc,
        "epochs": n_epochs,
        "regime": regime,
        "patients": patients,
    }
    
    os.makedirs("analysis/logs", exist_ok=True)
    log_name = f"analysis/logs/chbmit_{regime}_npz.json"
    
    with open(log_name, "w") as f:
        json.dump(log, f, indent=2)
    
    print("📄 Training log saved to:", log_name)
    print(f"✅ Finished training. Best AUC={best_auc:.3f}")


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--patients", required=True, help="comma-separated")
    parser.add_argument("--regime", default="edge")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    train_chbmit_npz(
        npz_dir=args.npz_dir,
        patients=args.patients.split(","),
        regime=args.regime,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )
