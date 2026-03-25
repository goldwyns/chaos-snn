# src/train_chbmit.py
"""
Train script for CHB-MIT (uses CHBMITWindowDataset above).
Stable training loop with AMP, balanced sampler option, proper checkpointing.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.datasets.chbmit import CHBMITWindowDataset
from src.models.chaos_snn import ChaosSNNSeizureDetector

def compute_grad_and_weight_norms(model):
    g_sq = 0.0; cnt = 0
    w_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data.norm(2).item()
            g_sq += g*g; cnt += 1
        w = p.data.norm(2).item()
        w_sq += w*w
    return (g_sq**0.5 if cnt>0 else 0.0), (w_sq**0.5)

def train_chbmit(
    base_dir,
    patient_ids=None,
    batch_size=128,
    n_epochs=10,
    lr=1e-4,
    val_split=0.2,
    device=None,
    ph_minutes=10,
    window_size_sec=5.0,
    overlap_sec=2.5,
    regime="edge",
    num_workers=2,
    pin_memory=False,
    use_amp=True,
    sampler_balancing=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Patients:", patient_ids)

    ds = CHBMITWindowDataset(
        base_dir=base_dir,
        patient_ids=patient_ids,
        ph_minutes=ph_minutes,
        window_size_sec=window_size_sec,
        overlap_sec=overlap_sec,
        downsample_sfreq=None,   # keep native fs
        chaos_precompute=True,
        chaos_params={"alpha":3.5,"beta":0.2,"gamma":0.5,"lam":0.7,"rng_seed":42},
        verbose=True
    )

    if len(ds) == 0:
        print("No windows. Check dataset paths/params.")
        return

    # quick sample to infer shapes
    x0, y0, m0 = ds[0]
    C, T = x0.shape
    fs = T / window_size_sec
    dt = 1.0 / fs
    print(f"CHB sample shape: C={C}, T={T}, inferred fs~={fs:.2f}Hz, dt={dt:.5f}s")

    # train/val split
    n_total = len(ds)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val
    indices = np.arange(n_total)
    rng = np.random.default_rng(seed=1234)
    rng.shuffle(indices)
    train_idx = indices[:n_train]; val_idx = indices[n_train:]

    # samplers
    if sampler_balancing:
        # create weighted sampler over train subset using labels
        labels = np.array([ds[i][1] for i in train_idx])
        class_counts = np.bincount(labels.astype(int))
        # avoid zero division
        class_counts = np.where(class_counts==0, 1, class_counts)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader([ds[i] for i in train_idx], batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader([ds[i] for i in train_idx], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader([ds[i] for i in val_idx], batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers//2), pin_memory=pin_memory)

    # build model parameters by regime
    if regime == "edge":
        alpha, beta, gamma, lam, alpha_rec = 3.5, 0.2, 0.5, 0.7, 1.0
    elif regime == "chaos_off":
        alpha, beta, gamma, lam, alpha_rec = 0.0, 0.0, 0.0, 1.0, 0.8
    elif regime == "sub":
        alpha, beta, gamma, lam, alpha_rec = 3.5, 0.2, 0.5, 0.7, 0.7
    elif regime == "super":
        alpha, beta, gamma, lam, alpha_rec = 3.5, 0.2, 0.5, 0.7, 1.2
    else:
        alpha, beta, gamma, lam, alpha_rec = 3.5, 0.2, 0.5, 0.7, 1.0

    model = ChaosSNNSeizureDetector(
        n_channels=C, n_hidden=128,
        alpha=alpha, beta=beta, gamma=gamma, lam=lam,
        v_th=1.0, alpha_rec=alpha_rec, dt=dt,
        tau_mem=20e-3, tau_syn=10e-3
    ).to(device)

    # loss + optimizer
    labels_all = np.array([ds[i][1] for i in range(len(ds))])
    n_pos = int((labels_all==1).sum()); n_neg = int((labels_all==0).sum())
    pos_weight = torch.tensor([max(1.0, n_neg/(n_pos+1e-9))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"chaos_snn_chbmit_{regime}_best.pt")

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device!="cpu"))

    for epoch in range(1, n_epochs+1):

    # ------------------------ TRAIN ------------------------
        model.train()
        running_loss = 0.0
        last_grad_norm = 0.0
        last_weight_norm = 0.0
    
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
    
            optimizer.zero_grad()
    
            # Mixed precision forward
            with torch.amp.autocast("cuda", enabled=(use_amp and device != "cpu")):
                p, extra = model(x_batch)
                logits = extra["logits"]
                loss = criterion(logits, y_batch)
    
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
            # Update weights
            scaler.step(optimizer)
            scaler.update()
    
            # Track norms
            grad_norm, weight_norm = compute_grad_and_weight_norms(model)
            last_grad_norm = grad_norm
            last_weight_norm = weight_norm
    
            running_loss += loss.item() * x_batch.size(0)
    
        train_loss = running_loss / max(1, n_train)
    
        # ------------------------ VALIDATION ------------------------
        model.eval()
        val_loss_total = 0.0
        y_true = []
        y_prob = []
    
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
    
                with torch.amp.autocast("cuda", enabled=(use_amp and device != "cpu")):
                    p, extra = model(x_batch)
                    logits = extra["logits"]
                    loss = criterion(logits, y_batch)
    
                val_loss_total += loss.item() * x_batch.size(0)
    
                y_true.extend(y_batch.cpu().numpy().tolist())
                y_prob.extend(p.detach().cpu().numpy().tolist())
    
        val_loss = val_loss_total / max(1, n_val)
    
        # AUC
        try:
            auc = roc_auc_score(np.array(y_true), np.array(y_prob))
        except Exception:
            auc = 0.0
    
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, AUC={auc:.4f}, "
            f"grad_norm={last_grad_norm:.4e}, weight_norm={last_weight_norm:.4e}"
        )
    
        # ------------------------ CHECKPOINT ------------------------
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": auc,
                    "regime": regime,
                },
                ckpt_path,
            )
            print(f"💾 Saved improved checkpoint: {ckpt_path}")
    
    print(f"Finished. Best AUC={best_val_auc:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to CHB-MIT dataset folder")
    parser.add_argument("--patients", type=str, default="",
                        help="Comma-separated patient IDs (empty = all)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    # Parse patient list
    if args.patients.strip() == "":
        patient_list = None
    else:
        patient_list = [p.strip() for p in args.patients.split(",")]

    train_chbmit(
        base_dir=args.base_dir,
        patient_ids=patient_list,
        batch_size=args.batch,
        n_epochs=args.epochs,
        lr=args.lr,
        val_split=0.2,
        ph_minutes=10,
        window_size_sec=5.0,
        overlap_sec=2.5,
        regime="edge",
        num_workers=2,
        pin_memory=True,
        use_amp=True,
        sampler_balancing=True,
    )
